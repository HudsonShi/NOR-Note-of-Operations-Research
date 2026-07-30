# NOR — Notes of Operations Research

运筹学优化算法笔记与课件，涵盖线性规划、非线性规划、凸优化，以及运筹优化在机器学习和大型语言模型中的应用。

## 内容概览

| 章节 | 主题 | 幻灯片数 |
|---|---|---|
| **Part 1** | 线性规划 (Linear Programming) — 标准形、单纯形法、对偶理论、灵敏度分析、内点法 | ~12 |
| **Part 2** | 非线性规划 (Nonlinear Programming) — KKT 条件、梯度下降 / 牛顿法 / BFGS、SQP、信赖域方法 | ~10 |
| **Part 2.5** | 启发式与元启发式算法 (Heuristics & Metaheuristics) — 贪心 / 局部搜索、模拟退火、遗传算法、PSO、禁忌搜索、蚁群算法 | ~8 |
| **Part 3** | 凸优化 (Convex Optimization) — 凸集与凸函数、对偶理论、近端梯度 / Nesterov 加速、ADMM、SDP | ~12 |
| **Part 5** | GPU / CUDA 并行计算与 ML 优化加速 — 混合精度、梯度累积、分布式训练、显存优化、FlashAttention | ~8 |
| **Part 4** | 运筹优化在机器学习与大模型中的应用 — SGD/Adam、分布式训练 (ZeRO)、LoRA/PEFT、RLHF/DPO、量化推理 | ~11 |

共计 **61 张 Beamer 幻灯片**，LaTeX (`ctexbeamer`) + XeLaTeX 编译。

## 编译

```bash
xelatex optimization_beamer.tex   # 运行两遍以稳定目录和交叉引用
```

环境要求：TeX Live 2023+（含 `ctex`, `ctexbeamer`, `fontspec`）。示例代码依赖见 [examples/README.md](examples/README.md)。

## 文件说明

| 文件 | 说明 |
|---|---|
| `optimization_beamer.tex` | Beamer 主文件 |
| `optimization_beamer.pdf` | 编译输出（61 页） |

## 许可

MIT License

## 代码示例 (`examples/`)

| 文件 | 类型 | 内容 |
|---|---|---|
| `lp_production.py` | Gurobi LP | 生产计划：原始 + 对偶，含影子价格与强对偶验证 |
| `mip_facility_location.py` | Gurobi MIP | 设施选址：固定成本 + 运输成本 |
| `tsp_mtz.py` | Gurobi MIP | TSP：Miller–Tucker–Zemlin 子环消除公式 |
| `simulated_annealing_tsp.py` | 纯 Python | 模拟退火解 TSP：Metropolis 准则 + 2-opt |
| `genetic_algorithm.py` | 纯 Python | 遗传算法：Rastrigin，锦标赛选择 + BLX-α + 高斯变异 |

详见 [examples/README.md](examples/README.md)。
