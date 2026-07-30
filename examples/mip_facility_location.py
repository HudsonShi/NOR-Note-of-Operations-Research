# -*- coding: utf-8 -*-
"""
MIP 示例：设施选址问题 (Uncapacitated Facility Location)

问题描述：
  有 m 个候选仓库位置，n 个客户。
  每个仓库有固定开设成本 f_j，从仓库 j 服务客户 i 的运输成本为 c_ij。
  每个客户必须被恰好一个仓库服务。
  目标：选择开设哪些仓库并分配客户，使总成本最小。

MIP 形式：
  min  sum_j f_j * y_j  +  sum_i sum_j c_ij * x_ij
  s.t. sum_j x_ij = 1           (每个客户被服务)
       x_ij ≤ y_j               (只有开设的仓库才能服务)
       x_ij ∈ {0, 1}, y_j ∈ {0, 1}
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_facility_location(m=5, n=15, seed=42):
    """求解随机生成的设施选址问题。"""
    rng = np.random.default_rng(seed)

    # 生成随机数据
    f_j   = rng.integers(50, 200, size=m)      # 开设固定成本
    loc_w = rng.uniform(0, 100, size=(m, 2))   # 仓库坐标
    loc_c = rng.uniform(0, 100, size=(n, 2))   # 客户坐标

    # 运输成本 = 欧氏距离
    c = np.zeros((m, n))
    for j in range(m):
        for i in range(n):
            c[j, i] = np.linalg.norm(loc_w[j] - loc_c[i])

    # -------- 建模 --------
    model = gp.Model("facility_location")

    # 变量
    y = model.addVars(m, vtype=GRB.BINARY, name="y")   # 是否开设仓库 j
    x = model.addVars(m, n, vtype=GRB.BINARY, name="x") # 客户 i 由仓库 j 服务

    # 目标
    model.setObjective(
        gp.quicksum(f_j[j] * y[j] for j in range(m)) +
        gp.quicksum(c[j, i] * x[j, i] for j in range(m) for i in range(n)),
        GRB.MINIMIZE
    )

    # 约束 (1)：每个客户恰好被一个仓库服务
    model.addConstrs(
        (gp.quicksum(x[j, i] for j in range(m)) == 1 for i in range(n)),
        name="demand"
    )

    # 约束 (2)：只有开设的仓库才能服务
    model.addConstrs(
        (x[j, i] <= y[j] for j in range(m) for i in range(n)),
        name="link"
    )

    model.optimize()

    # -------- 输出 --------
    print(f"\nStatus: {model.status}")
    print(f"Total cost = {model.ObjVal:.2f}")

    opened = [j for j in range(m) if y[j].X > 0.5]
    print(f"Opened facilities: {opened} "
          f"(fixed cost = {sum(f_j[j] for j in opened):.0f})")

    for j in opened:
        served = [i for i in range(n) if x[j, i].X > 0.5]
        transport = sum(c[j, i] for i in served)
        print(f"  Facility {j}: serves {len(served)} customers, "
              f"transport cost = {transport:.2f}")

    return model


if __name__ == "__main__":
    solve_facility_location(m=5, n=15)
