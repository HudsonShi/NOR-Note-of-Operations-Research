# -*- coding: utf-8 -*-
"""
纯 Python 实现：模拟退火求解 TSP

无外部依赖，仅需 Python 标准库。
展示模拟退火的核心思想：Metropolis 准则 + 冷却调度。

邻域移动：2-opt swap（反转子路径中的一段）。
"""

import random
import math
import time


def total_distance(tour, dist):
    """计算路径总长度。"""
    return sum(dist[tour[i]][tour[(i + 1) % len(tour)]]
               for i in range(len(tour)))


def two_opt_swap(tour):
    """随机 2-opt 移动：反转 tour[i:j] 一段。"""
    i, j = sorted(random.sample(range(len(tour)), 2))
    if j - i <= 2:   # 太短没意义，重新来
        return two_opt_swap(tour)
    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
    return new_tour


def simulated_annealing_tsp(points, T0=1000, alpha=0.995,
                            T_min=1e-3, max_iter=50000):
    """
    参数：
      T0: 初始温度
      alpha: 冷却率 T_{k+1} = alpha * T_k
      T_min: 终止温度
      max_iter: 每温度下的最大迭代次数
    """
    n = len(points)

    # 距离矩阵
    dist = [[math.dist(points[i], points[j]) for j in range(n)]
            for i in range(n)]

    # 初始解：随机排列
    current = list(range(n))
    random.shuffle(current)
    current_dist = total_distance(current, dist)
    best_tour, best_dist = current[:], current_dist

    T = T0
    iterations = 0

    while T > T_min:
        for _ in range(max_iter):
            candidate = two_opt_swap(current)
            candidate_dist = total_distance(candidate, dist)
            delta = candidate_dist - current_dist

            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)

            if accept:
                current, current_dist = candidate, candidate_dist
                if current_dist < best_dist:
                    best_tour, best_dist = current[:], current_dist

        T *= alpha
        iterations += 1

    return best_tour, best_dist


if __name__ == "__main__":
    import random as _r
    _r.seed(42)

    # 随机生成 30 个城市
    points = [(_r.uniform(0, 100), _r.uniform(0, 100)) for _ in range(30)]

    t0 = time.time()
    tour, dist = simulated_annealing_tsp(points)
    elapsed = time.time() - t0

    print(f"Tour length: {dist:.2f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Tour: {' → '.join(str(t) for t in tour)} → {tour[0]}")
