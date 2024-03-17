import numpy as np
from deap import base, creator, tools, algorithms

# 初始化参数
# 初始化索力、目标索力和影响矩阵
initial_forces = np.array([1536.6, 1695.7, 1748.9, 1787.6, 1769.0, 1550.0, 1854.5, 1615.9, 1670.0, 1747.9, 1750.6])
target_forces = np.array([1438, 1617, 1614, 1651, 1651, 1701, 1701, 1703, 1701, 1700, 1704])
influence_matrix = np.array([
    [-100, 28, 19, 11, 5, 1, -1, -2, -1, -1, 0],
    [16, -100, 25, 18, 9, 3, 0, -1, -1, -1, 0],
    [9, 19, -100, 24, 16, 8, 3, 0, -1, -1, 0],
    [4, 11, 20, -100, 24, 15, 8, 2, 0, -1, 0],
    [2, 6, 14, 24, -100, 24, 16, 8, 2, 0, 0],
    [0, 3, 8, 15, 24, -100, 24, 15, 7, 2, 1],
    [0, 0, 3, 8, 15, 23, -100, 23, 14, 7, 2],
    [-1, -1, 0, 3, 7, 15, 23, -100, 22, 12, 4],
    [-1, -1, -1, 0, 2, 8, 15, 23, -100, 18, 8],
    [0, -1, -1, -1, 0, 3, 8, 15, 22, -100, 14],
    [0, 0, 0, -1, 0, 1, 3, 8, 15, 19, -100]
])*(-0.01)

POPULATION_SIZE = 1000
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
GENERATIONS = 100

MAX_FORCE = 3000
TOLERANCE = 0.05  # 5%

# 定义问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -300, 300)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(initial_forces))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评价函数
def evalAdjustment(individual):
    adjusted_forces = initial_forces + np.dot(influence_matrix, individual)
    penalties = 0

    # 约束：索力不能超过最大值或变为负值
    if any(adjusted_forces > MAX_FORCE) or any(adjusted_forces < 0):
        penalties += 1e6

    # 误差计算
    error = np.abs((adjusted_forces - target_forces) / target_forces)
    
    # 约束：误差控制在-5%到+5%
    if any(error > TOLERANCE):
        penalties += 1e6

    return np.sum(error) + penalties,

toolbox.register("evaluate", evalAdjustment)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def print_adjustment_details(best_individual):
    print("最优调整方案分析:")
    simulated_adjusted_forces = initial_forces.copy()  # 初始索力，将根据每步调整进行更新

    for step, adjustment in enumerate(best_individual):
        if adjustment != 0:  # 考虑到可能不是每个基因都会导致调整
            # 模拟这一步的调整
            simulated_adjusted_forces += influence_matrix[:, step] * adjustment
            error_percentages = (simulated_adjusted_forces - target_forces) / target_forces * 100

            print(f"\n步骤 {step+1}: 索 {step+1} 调整值: {adjustment:.2f}kN")
            print("调整后的所有索的索力和误差百分比:")
            for i, (force, error) in enumerate(zip(simulated_adjusted_forces, error_percentages), start=1):
                print(f"索 {i}: 调整后索力: {force:.2f}kN, 误差百分比: {error:.2f}%")

    # 最终所有索的平均误差百分比
    final_error_percentages = (simulated_adjusted_forces - target_forces) / target_forces * 100
    total_error = np.mean(np.abs(final_error_percentages))
    print(f"\n最终所有索的平均误差百分比: {total_error:.2f}%")

# 运行遗传算法
def main():
    population = toolbox.population(n=POPULATION_SIZE)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    
    # algorithms.eaSimple现在返回更新后的种群和一个日志对象
    final_pop, log = algorithms.eaSimple(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE, 
                                         ngen=GENERATIONS, stats=stats, verbose=True)
    
    return final_pop, log, stats  # 确保返回这三个值

if __name__ == "__main__":
    final_pop, log, stats = main()
    best_individual = tools.selBest(final_pop, 1)[0]
    print_adjustment_details(best_individual)

