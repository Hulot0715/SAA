# ============================================================
# TSP 模拟退火算法 — 初始参数配置文件
# ============================================================
# 修改此文件中的参数即可调整实验设置，无需改动主程序逻辑。

# 随机种子（取学号后5位）
SEED: int = 24341

# 城市总数
N_CITIES: int = 50

# 坐标范围（生成 COORD_RANGE x COORD_RANGE 的平面）
COORD_RANGE: int = 1000

# 学号后两位（动态计算，用于初始温度）
ID_LAST2: int = SEED % 100

# 初始温度
T0: float = 1000 + ID_LAST2 * 20

# 终止温度
T_FINAL: float = 1.0

# 对比实验的退火系数列表
ALPHAS: list = [0.85, 0.92, 0.99]

# 每个温度下的内循环迭代次数
INNER_ITER: int = 200

# 提前停止超参数：连续若干外循环无最优解改进则停止
PATIENCE: int = 44

# 邻域解生成方法（可选：'2-opt'、'swap'、'insert'）
NEIGHBOR_METHOD: str = '2-opt'

# 降温策略（可选：'exponential'、'linear'、'logarithmic'、'adaptive'）
COOLING_STRATEGY: str = 'exponential'

# 初始解生成方法（可选：'random'、'nearest_neighbor'）
#   - 'random'：随机全排列初始解
#   - 'nearest_neighbor'：最邻近贪心初始解
INITIAL_TOUR_METHOD: str = 'random'
