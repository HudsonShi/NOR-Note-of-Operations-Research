# -*- coding: utf-8 -*-
"""
MIP 示例：旅行商问题 (TSP) — Miller–Tucker–Zemlin 公式

MTZ 是经典的紧凑 TSP MIP 公式，通过辅助变量 u_i 消除子环。

变量：
  x_{ij} ∈ {0, 1}：是否从 i 直接到 j
  u_i ∈ [1, n]：节点 i 在路径中的访问顺序（从 1 开始）

子环消除约束 (MTZ)：
  u_i - u_j + n * x_{ij} ≤ n - 1    ∀ i ≠ j, i, j ≥ 2
  1 ≤ u_i ≤ n
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_tsp_mtz(n=10, seed=123):
    """求解随机 TSP 实例。"""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, size=(n, 2))

    # 距离矩阵
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = np.linalg.norm(points[i] - points[j])

    model = gp.Model("TSP_MTZ")

    # 变量
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, lb=1, ub=n, name="u")

    # 目标：最小化总距离
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j),
        GRB.MINIMIZE
    )

    # 每个节点恰入一次、恰出一次
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in range(n) if j != i) == 1 for i in range(n)),
        name="out"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in range(n) if i != j) == 1 for j in range(n)),
        name="in"
    )

    # MTZ 子环消除
    model.addConstrs(
        (u[i] - u[j] + n * x[i, j] <= n - 1
         for i in range(1, n) for j in range(1, n) if i != j),
        name="mtz"
    )

    model.optimize()

    # -------- 提取路径 --------
    if model.status == GRB.OPTIMAL:
        print(f"\nOptimal length = {model.ObjVal:.2f}")
        # 从节点 0 出发沿 x 走一圈
        tour = [0]
        visited = {0}
        while len(tour) < n:
            i = tour[-1]
            for j in range(n):
                if j not in visited and x[i, j].X > 0.5:
                    tour.append(j)
                    visited.add(j)
                    break
        tour.append(0)
        print(f"Tour: {' → '.join(str(t) for t in tour)}")

    return model


if __name__ == "__main__":
    solve_tsp_mtz(n=10)
