from deap import base, creator, tools, algorithms
from multiprocessing import Pool
import random
import numpy as np

#重要注释解释：
#种群和个体初始化：在遗传算法中，种群是潜在解决方案的集合，而个体则代表一个具体的解决方案。本代码通过tools.initRepeat函数初始化种群和个体，其中个体的基因由随机生成的调整力组成。
#适应度评估：evalAdjustment函数用于评估每个个体的适应度。适应度是根据个体的基因（即索力调整方案）对应的结构表现来计算的，包括计算调整后索力的误差和对违反约束条件的个体施加惩罚。
#算子注册：mate（交叉）、mutate（变异）和select（选择）算子是遗传算法中核心的进化操作。通过这些操作，算法能够在种群中产生新个体，并逐代改进解决方案。
#多进程加速：利用multiprocessing模块的Pool类来加速适应度评估过程，特别是当处理大型种群或复杂评估函数时。
#遗传算法执行：algorithms.eaSimple是DEAP库提供的一个简单遗传算法实现。cxpb和mutpb分别代表交叉和变异的概率，ngen表示进化的代数。

# 检查creator中是否已定义FitnessMin和Individual，避免重复定义
if not hasattr(creator, "FitnessMin"):
    # 创建一个适应度类，适应度越小越好
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    # 创建个体类，个体是基因列表，包含适应度信息
    creator.create("Individual", list, fitness=creator.FitnessMin)

# 评估个体适应度的函数
def evalAdjustment(individual, initial_forces, target_forces, influence_matrix, max_force, min_adjustment, max_adjustment, tolerance, lower_bounds, upper_bounds, selected_metric='max_error_percent', threshold=40):
    PENALTY = 1e3
    
    # 根据影响矩阵和个体基因计算调整后的索力
    adjusted_forces = initial_forces + np.dot(influence_matrix, individual)
    
    penalties = 0  # 初始化罚分
    
    # 如果调整后的索力超出最大值或小于0，则增加罚分
    if any(adjusted_forces > max_force) or any(adjusted_forces < 0):
        penalties += PENALTY
    
    # 检查个体的每个基因是否在下限和上限之间
    for gene, lb, ub in zip(individual, lower_bounds, upper_bounds):
        if gene < lb or gene > ub:
            penalties += PENALTY  # 对于超出范围的基因施加罚分
    
    # 如果个体基因值超出调整范围，则增加罚分
    if any(x < min_adjustment or x > max_adjustment for x in individual):
        penalties += PENALTY
    
    # 应用阈值调整（将绝对值小于阈值的基因置为0）
    adjusted_individual = [value if abs(value) > threshold else 0 for value in individual]
    
    # 根据阈值调整后的个体基因计算最终的调整索力
    adjusted_forces_after_threshold = initial_forces + np.dot(influence_matrix, adjusted_individual)
    
    # 计算索力调整的误差
    error = np.abs((adjusted_forces_after_threshold - target_forces) / target_forces)
    
    # 如果误差超出容忍范围，则增加罚分
    if any(error > tolerance):
        penalties += PENALTY

    error_percent = (adjusted_forces_after_threshold - target_forces) / target_forces  # 计算误差
    
    # 计算最大误差百分比，同时考虑正误差和负误差
    max_positive_error_percent = np.max(error_percent)
    max_negative_error_percent = np.min(error_percent)
    max_error_percent = max_positive_error_percent if abs(max_positive_error_percent) >= abs(max_negative_error_percent) else max_negative_error_percent
    max_abs_error_percent = np.max(np.abs(error_percent))  # 计算最大绝对误差百分比

    mse = np.mean(np.square(error))  # 计算误差的均方误差

    # 根据选择的指标返回适应度
    if selected_metric == 'mse':
        return (mse + penalties,)
    elif selected_metric == 'max_error_percent':
        return (max_abs_error_percent + penalties,)


# 运行遗传算法的主函数
def run_genetic_algorithm(initial_forces, target_forces, influence_matrix, population_size, crossover_rate, mutation_rate, min_adjustment, max_adjustment, max_force, tolerance, ngen,lower_bounds, upper_bounds,selected_metric):
    toolbox = base.Toolbox()
    # 注册属性生成函数，生成调整力的浮点数
    toolbox.register("attr_float", np.random.uniform, min_adjustment, max_adjustment)

    # 注册个体和种群的生成函数
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(initial_forces))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册评估函数，包括了额外参数的传递
    toolbox.register("evaluate", evalAdjustment, initial_forces=initial_forces, target_forces=target_forces, 
                     influence_matrix=influence_matrix, max_force=max_force, min_adjustment=min_adjustment, 
                     max_adjustment=max_adjustment, tolerance=tolerance,lower_bounds=lower_bounds,upper_bounds=upper_bounds,selected_metric=selected_metric)

    # 注册交叉、变异和选择算子
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=-500, up=500, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 生成初始种群
    population = toolbox.population(n=population_size)
    
    # 使用多进程来加速计算
    pool = Pool()
    toolbox.register("map", pool.map)

    # 执行遗传算法
    final_pop, log = algorithms.eaSimple(population, toolbox, cxpb=crossover_rate, mutpb=mutation_rate, 
                                         ngen=ngen, stats=None, verbose=True)

    # 关闭进程池
    pool.close()
    pool.join()

    # 选择并返回最佳个体及其适应度
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values

    return best_individual, best_fitness, final_pop, log
