# Examples — Python + Gurobi 运筹优化代码

| 文件 | 类型 | 内容 |
|---|---|---|
| `lp_production.py` | Gurobi LP | 生产计划：原始问题 + 对偶显式构造，含影子价格与强对偶验证 |
| `mip_facility_location.py` | Gurobi MIP | 设施选址：固定成本 + 运输成本，二元变量 + 分配约束 |
| `tsp_mtz.py` | Gurobi MIP | TSP：Miller–Tucker–Zemlin 子环消除公式 |
| `simulated_annealing_tsp.py` | 纯 Python | 模拟退火解 TSP：Metropolis 准则 + 2-opt 邻域 |
| `genetic_algorithm.py` | 纯 Python | 遗传算法：Rastrigin 函数，锦标赛选择 + BLX-α 交叉 + 高斯变异 |

## 依赖

- **Gurobi 例**：需安装 `gurobipy` 并配置 license
- **纯 Python 例**：无外部依赖，Python 3.8+ 即可运行
