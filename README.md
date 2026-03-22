# 模拟退火算法求解 TSP（SAA）

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Algorithm" src="https://img.shields.io/badge/Algorithm-Simulated%20Annealing-ff6f61">
  <img alt="Problem" src="https://img.shields.io/badge/Problem-TSP-4caf50">
</p>

## 目录

- [项目概览](#项目概览)
- [实验结果汇总](#实验结果汇总)
  - [个性化实验参数](#个性化实验参数)
  - [参数对比结果（v1.0，指数降温 α 对比）](#参数对比结果v10指数降温-α-对比)
  - [结论分析](#结论分析)
- [当前支持功能（v1.7+）](#当前支持功能v17)
- [项目结构](#项目结构)
- [核心实现片段](#核心实现片段)
  - [邻域构造（统一接口）](#邻域构造统一接口)
  - [Metropolis 接受准则](#metropolis-接受准则)
  - [温度更新策略（四种）](#温度更新策略四种)
- [快速开始](#快速开始)
- [后续改进方向](#后续改进方向)

---

## 项目概览

本项目使用**模拟退火算法（Simulated Annealing, SA）**求解旅行商问题（TSP），并围绕不同降温系数进行参数对比实验，分析收敛速度与解质量之间的权衡关系。

---

## 实验结果汇总

### 个性化实验参数

| 参数 | 值 |
| --- | --- |
| 学号 | 125130024341 |
| 随机种子（后5位） | 24341 |
| 城市数量 N | 50 |
| 坐标范围 | 1000 × 1000 |
| 初始温度 T₀ | 1820 |
| 终止温度 T_final | 1.0 |
| 每温度内循环次数 | 200 |

### 参数对比结果（v1.0，指数降温 α 对比）

| 退火系数 α | 外循环迭代次数 | 最优路径长度 | 耗时（秒） |
| --- | --- | --- | --- |
| 0.85 | 47 | 6012.33 | 0.09 |
| 0.92 | 91 | 5824.29 | 0.21 |
| 0.99 | 747 | 5487.02 | 1.51 |

### 结论分析

- `α = 0.99`：最优路径最短（`5487.02`），搜索最充分，但耗时最长。
- `α = 0.85`：迭代轮次最少、运行最快，但路径质量最差。
- `α = 0.92`：在速度与质量之间表现居中。
- 总体规律：**退火越慢（α 越大）→ 搜索越充分 → 更接近全局最优**。

---

## 当前支持功能（v1.7+）

- **邻域方法**：`2-opt`（O(1) 增量）、`swap`（O(1) 增量）、`insert`（整路重算）
- **降温策略**：`exponential`（指数）、`linear`（线性）、`logarithmic`（对数）、`adaptive`（自适应）
- **配置化切换**：通过 `config.py` 中的 `NEIGHBOR_METHOD` 与 `COOLING_STRATEGY` 选择策略，无需改源码

---

## 项目结构

| 文件名 | 说明 |
| --- | --- |
| `config.py` | 实验参数配置（问题规模、算法超参数、邻域方法、降温策略） |
| `tsp_simulated_annealing.py` | 主算法实现（含四种降温策略、三种邻域方法、O(1) 增量计算） |
| `tsp_result_alpha085.png` | α=0.85 单组实验四子图（路径、长度、温度、接受率） |
| `tsp_result_alpha092.png` | α=0.92 单组实验四子图（路径、长度、温度、接受率） |
| `tsp_result_alpha099.png` | α=0.99 单组实验四子图（路径、长度、温度、接受率） |
| `tsp_comparison.png` | 三组参数对比四子图（最优长度、当前长度、温度、接受率） |
| `LOG.md` | 实验日志 |

---

## 核心实现片段

### 邻域构造（统一接口）

```python
# 2-opt：O(1) 增量计算
new_tour, delta_E = move_2opt(tour, dist_matrix)

# swap：O(1) 增量计算（非相邻情形）
new_tour, delta_E = move_swap(tour, dist_matrix)

# insert：整路重算
new_tour, delta_E = move_insert(tour, dist_matrix)
```

### Metropolis 接受准则

```python
if delta_E < 0:
    # 优于当前解 -> 无条件接受
    current_tour, current_len = new_tour, new_len
else:
    # 劣于当前解 -> 以 Boltzmann 概率接受
    if np.random.random() < np.exp(-delta_E / T):
        current_tour, current_len = new_tour, new_len
```

### 温度更新策略（四种）

```python
if T_d == 'exponential':
    T *= alpha  # 指数降温
elif T_d == 'linear':
    T -= linear_step  # 线性降温
elif T_d == 'logarithmic':
    T = T0 / np.log(outer_iter + 2)  # 对数降温
elif T_d == 'adaptive':
    # 自适应降温（按接受率调节）
    acc_rate = accepted / inner_iter
    if acc_rate > 0.6:
        T *= 0.90
    elif acc_rate < 0.2:
        T *= 0.98
    else:
        T *= 0.95
T = max(T, T_final)  # 温度下界保护
```

---

## 快速开始

```bash
python tsp_simulated_annealing.py
```

如需调参，请编辑 `config.py` 中的相关配置（如 `ALPHAS`、`INNER_ITER`、`NEIGHBOR_METHOD`、`COOLING_STRATEGY`）。

---

## 后续改进方向

- 多次独立运行统计（10~30 次，记录 `mean/std/best/worst`）
- 提前停止准则（连续无改进轮数阈值）
- 初始解优化（最近邻贪心初解 vs 随机初解）
- 邻域混合策略（`2-opt + insert` 自适应切换）
- 实验报告 PDF 撰写与提交（截止：3月30日 23:59:59）
