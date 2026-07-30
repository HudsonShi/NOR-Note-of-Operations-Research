# -*- coding: utf-8 -*-
"""
纯 Python 实现：遗传算法求解连续函数最优化

问题：minimize the Rastrigin function（经典的多峰测试函数）
  f(x) = 10n + sum_i [x_i^2 - 10 * cos(2π x_i)],  x_i ∈ [-5.12, 5.12]
  global minimum: f(0,...,0) = 0

编码：实数编码，交叉 = BLX-α (blend crossover)，变异 = 高斯扰动
选择：锦标赛选择 (tournament selection)
"""

import random
import math
from typing import List, Tuple


def rastrigin(x: List[float]) -> float:
    """Rastrigin 函数：全局最优在 x=0，值为 0。"""
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def tournament_select(population, fitness, k=3):
    """锦标赛选择：随机选 k 个，返回最优者。"""
    idx = random.sample(range(len(population)), k)
    best = min(idx, key=lambda i: fitness[i])
    return population[best][:]


def blend_crossover(p1, p2, alpha=0.5):
    """BLX-α 交叉：子代在 [min - α*d, max + α*d] 区间随机生成。"""
    child = []
    for a, b in zip(p1, p2):
        lo, hi = min(a, b), max(a, b)
        d = hi - lo
        child.append(random.uniform(lo - alpha * d, hi + alpha * d))
    return child


def gaussian_mutation(x, sigma=0.1, mutation_rate=0.1):
    """高斯变异：以 mutation_rate 的概率扰动每个基因。"""
    x_mut = x[:]
    for i in range(len(x)):
        if random.random() < mutation_rate:
            x_mut[i] += random.gauss(0, sigma)
    return x_mut


def clamp(x, bounds=(-5.12, 5.12)):
    return [max(bounds[0], min(bounds[1], xi)) for xi in x]


def genetic_algorithm(dim=5, pop_size=50, generations=200,
                      crossover_rate=0.8, mutation_rate=0.1,
                      elite_size=2, seed=42):
    """
    参数：
      dim: 维度
      pop_size: 种群大小
      generations: 迭代代数
      crossover_rate: 交叉概率
      mutation_rate: 变异率
      elite_size: 精英个体数（直接保留到下一代）
    """
    random.seed(seed)
    bounds = (-5.12, 5.12)

    # 初始化种群
    population = [[random.uniform(*bounds) for _ in range(dim)]
                  for _ in range(pop_size)]

    best_solution = None
    best_fitness = float('inf')

    for gen in range(generations):
        fitness = [rastrigin(ind) for ind in population]

        # 精英保留
        elite_idx = sorted(range(pop_size), key=lambda i: fitness[i])[:elite_size]
        elites = [population[i][:] for i in elite_idx]

        # 更新全局最优
        if fitness[elite_idx[0]] < best_fitness:
            best_fitness = fitness[elite_idx[0]]
            best_solution = population[elite_idx[0]][:]

        # 生成下一代
        next_pop = elites[:]  # 保留精英
        while len(next_pop) < pop_size:
            p1 = tournament_select(population, fitness)
            p2 = tournament_select(population, fitness)

            if random.random() < crossover_rate:
                child = blend_crossover(p1, p2)
            else:
                child = p1[:]

            child = gaussian_mutation(child, mutation_rate=mutation_rate)
            child = clamp(child, bounds)
            next_pop.append(child)

        population = next_pop[:pop_size]

        if gen % 40 == 0:
            print(f"Gen {gen:4d}: best = {best_fitness:.6f}")

    return best_solution, best_fitness


if __name__ == "__main__":
    dim = 10
    sol, val = genetic_algorithm(dim=dim, pop_size=100, generations=200)

    print(f"\nDimension: {dim}")
    print(f"Global optimum: 0.0 (at x=0)")
    print(f"GA found: f(x*) = {val:.6f}")
    print(f"x* = {[round(xi, 4) for xi in sol]}")
