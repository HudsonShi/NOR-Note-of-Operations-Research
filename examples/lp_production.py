# -*- coding: utf-8 -*-
"""
LP 示例：生产计划问题 (Production Planning)

问题描述：
  一家工厂生产两种产品 A 和 B。
  每种产品需要三种资源：原材料、工时、机器时间。
  约束：
    - 原材料总量 ≤ 100 单位
    - 工时总量 ≤ 80 小时
    - 机器时间总量 ≤ 90 小时
  目标：最大化总利润。

LP 标准形：
  max  40 x_A + 50 x_B
  s.t. 2 x_A + 3 x_B ≤ 100   (原材料)
       4 x_A + 2 x_B ≤ 80    (工时)
       3 x_A + 3 x_B ≤ 90    (机器)
       x_A, x_B ≥ 0
"""

import gurobipy as gp
from gurobipy import GRB


def solve_primal():
    """求解原始 LP 并输出最优解。"""
    m = gp.Model("production_planning")

    # 决策变量
    x_A = m.addVar(name="x_A", lb=0)
    x_B = m.addVar(name="x_B", lb=0)

    # 目标：最大化利润
    m.setObjective(40 * x_A + 50 * x_B, GRB.MAXIMIZE)

    # 约束
    m.addConstr(2 * x_A + 3 * x_B <= 100, "raw_material")
    m.addConstr(4 * x_A + 2 * x_B <= 80,  "labor")
    m.addConstr(3 * x_A + 3 * x_B <= 90,  "machine")

    m.optimize()

    print("\n===== Primal Solution =====")
    print(f"Status: {m.status} ({GRB.OPTIMAL=})")
    print(f"x_A = {x_A.X:.2f}, x_B = {x_B.X:.2f}")
    print(f"Optimal profit = {m.ObjVal:.2f}")

    # --- Shadow Prices (dual variables) ---
    print("\n===== Shadow Prices =====")
    for c in m.getConstrs():
        print(f"  {c.ConstrName}: dual = {c.Pi:.4f}  "
              f"(slack = {c.Slack:.2f}, RHS = {c.RHS:.0f})")

    return m


def solve_dual():
    """显式构造并求解对偶问题。"""
    m = gp.Model("production_dual")

    # 对偶变量（无约束，因为 primal 是 ≤ 且变量 ≥ 0 → dual 变量 ≥ 0 且无约束）
    y1 = m.addVar(name="y_raw",    lb=0)   # 对应原材料约束
    y2 = m.addVar(name="y_labor",  lb=0)   # 对应工时约束
    y3 = m.addVar(name="y_machine",lb=0)   # 对应机器约束

    # 对偶目标：min 100 y1 + 80 y2 + 90 y3
    m.setObjective(100 * y1 + 80 * y2 + 90 * y3, GRB.MINIMIZE)

    # 对偶约束：A^T y ≥ c
    m.addConstr(2 * y1 + 4 * y2 + 3 * y3 >= 40, "dual_A")
    m.addConstr(3 * y1 + 2 * y2 + 3 * y3 >= 50, "dual_B")

    m.optimize()

    print("\n===== Dual Solution =====")
    print(f"y_raw = {y1.X:.4f}, y_labor = {y2.X:.4f}, y_machine = {y3.X:.4f}")
    print(f"Optimal dual value = {m.ObjVal:.4f}")

    return m


if __name__ == "__main__":
    primal = solve_primal()
    dual   = solve_dual()

    # --- 强对偶验证 ---
    print(f"\nPrimal obj = {primal.ObjVal:.4f}, Dual obj = {dual.ObjVal:.4f}")
    print(f"Strong duality gap = {abs(primal.ObjVal - dual.ObjVal):.2e}")
