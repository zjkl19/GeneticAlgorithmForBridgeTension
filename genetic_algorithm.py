from deap import base, creator, tools, algorithms
from multiprocessing import Pool
import random
import numpy as np

# 检查creator中是否已定义FitnessMin和Individual，如果已定义，则不重复创建
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

# 将evalAdjustment函数移到模块顶层
def evalAdjustment(individual, initial_forces, target_forces, influence_matrix, max_force, min_adjustment, max_adjustment, tolerance):
    adjusted_forces = initial_forces + np.dot(influence_matrix, individual)
    penalties = 0
    if any(adjusted_forces > max_force) or any(adjusted_forces < 0):
        penalties += 1e6
    #if abs(individual[0]) > 20 or abs(individual[1]) > 10:
    #    penalties += 1e6
    error = np.abs((adjusted_forces - target_forces) / target_forces)
    if any(x < min_adjustment or x > max_adjustment for x in individual):
        penalties += 1e6
    if any(error > tolerance):
        penalties += 1e6
    mse = np.mean(np.square(error))
    return (mse + penalties,)

def run_genetic_algorithm(initial_forces, target_forces, influence_matrix, population_size, crossover_rate, mutation_rate, min_adjustment, max_adjustment, max_force, tolerance, ngen):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, min_adjustment, max_adjustment)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(initial_forces))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 修改toolbox.evaluate的注册，以包括额外的参数
    toolbox.register("evaluate", evalAdjustment, initial_forces=initial_forces, target_forces=target_forces, 
                     influence_matrix=influence_matrix, max_force=max_force, min_adjustment=min_adjustment, 
                     max_adjustment=max_adjustment, tolerance=tolerance)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    
    pool = Pool()
    toolbox.register("map", pool.map)

    final_pop, log = algorithms.eaSimple(population, toolbox, cxpb=crossover_rate, mutpb=mutation_rate, 
                                         ngen=ngen, stats=None, verbose=True)

    pool.close()
    pool.join()

    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values

    return best_individual, best_fitness, final_pop, log