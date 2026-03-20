import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from typing import List, Tuple

# ============================================================
# 0. 初始参数配置（从外部配置文件导入，修改参数在 config.py）
# ============================================================
from config import (
    SEED,
    N_CITIES,
    COORD_RANGE,
    ID_LAST2,
    T0,
    T_FINAL,
    ALPHAS,
    INNER_ITER,
)

# 配置中文字体（用于画图）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 城市坐标生成
# ============================================================
def generate_cities(seed: int, n: int = 50) -> np.ndarray:
    """
    以学号后5位为随机种子，生成 n 座城市的坐标。
    坐标范围：[0, 1000] x [0, 1000]
    """
    np.random.seed(seed)
    x = np.random.uniform(0, COORD_RANGE, n)
    y = np.random.uniform(0, COORD_RANGE, n)
    return np.column_stack((x, y)) # 返回城市坐标数组


# ============================================================
# 2. 城市距离计算（欧式距离）
# ============================================================
def calc_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """
    计算城市间欧氏距离矩阵
    输入:
        cities: 城市坐标数组
    输出:
        dist_matrix: 城市间欧氏距离矩阵
    """
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :] # 利用广播机制计算两两城市间的坐标差
    return np.sqrt((diff ** 2).sum(axis=2)) # 对最后一轴求平方和再开根：(N,N,2) -> (N,N)，即欧氏距离矩阵


def tour_length(tour: List[int], dist_matrix: np.ndarray) -> float:
    """
    计算路径总长度
    输入：
        tour: 路径
        dist_matrix: 城市间欧氏距离矩阵
    输出：
        total: 路径总长度
    """
    total = 0.0
    n = len(tour)
    for i in range(n): # 遍历路径中的每个城市
        total += dist_matrix[tour[i]][tour[(i + 1) % n]] # tour[(i + 1) % n让路径首尾相连，形成闭合回路。
    return total


# ============================================================
# 3. 邻域解生成（三种实现方式：2-opt逆转、随机交换、插入法）
# ============================================================
def neighbor_2opt(tour: List[int]) -> List[int]:
    """
    2-opt 邻域：随机选取两个位置 i < j，逆转 tour[i..j] 段。
    这是最经典的 TSP 邻域操作。
    输入：
        tour: 路径
    输出：
        new_tour: 新的路径
    """
    new_tour = tour.copy()
    i, j = sorted(np.random.choice(len(tour), 2, replace=False)) # 随机选取两个位置 i < j
    new_tour[i:j + 1] = new_tour[i:j + 1][::-1] # 逆转 tour[i..j] 段
    return new_tour


def neighbor_swap(tour: List[int]) -> List[int]:
    """
    随机交换两城市位置
    输入：
        tour: 路径
    输出：
        new_tour: 新的路径
    """
    new_tour = tour.copy()
    i, j = np.random.choice(len(tour), 2, replace=False)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def neighbor_insert(tour: List[int]) -> List[int]:
    """
    插入法：随机取出一座城市，插入到随机位置
    输入：
        tour: 路径
    输出：
        new_tour: 新的路径
    """
    new_tour = tour.copy()
    i = np.random.randint(len(tour))
    city = new_tour.pop(i)
    j = np.random.randint(len(new_tour))
    new_tour.insert(j, city)
    return new_tour


# ============================================================
# 4. 模拟退火核心求解函数
# ============================================================
def simulated_annealing(
    cities: np.ndarray,
    dist_matrix: np.ndarray,
    alpha: float,
    T0: float = T0,
    T_final: float = T_FINAL,
    inner_iter: int = INNER_ITER,
    rng_seed: int = SEED
) -> Tuple[List[int], float, dict]:
    """
    模拟退火算法求解 TSP。

    Args:
        cities      : 城市坐标数组
        dist_matrix : 距离矩阵
        alpha       : 退火系数（冷却率）
        T0          : 初始温度
        T_final     : 终止温度
        inner_iter  : 每个温度的内循环迭代次数
        rng_seed    : 随机种子

    Returns:
        best_tour   : 最优路径（城市索引列表）
        best_len    : 最优路径长度
        history     : 优化过程记录字典
    """
    np.random.seed(rng_seed)
    n = len(cities)

    # 初始解：随机全排列
    current_tour = list(range(n))
    np.random.shuffle(current_tour)
    current_len = tour_length(current_tour, dist_matrix)

    best_tour = current_tour.copy()
    best_len  = current_len

    history = {
        'best_lengths'    : [],
        'current_lengths' : [],
        'temperatures'    : [],
        'acceptance_rates': [],
        'iterations'      : [],
    }

    T = T0
    outer_iter = 0

    # 外循环：温度逐步降低
    while T > T_final:
        outer_iter += 1
        accepted = 0

        # 内循环：热平衡（Metropolis 采样）
        for _ in range(inner_iter):
            # 生成邻域解（2-opt 逆转法）
            new_tour = neighbor_2opt(current_tour)
            new_len  = tour_length(new_tour, dist_matrix)

            delta_E = new_len - current_len

            # Metropolis 接受准则
            if delta_E < 0:
                # 优于当前解 -> 无条件接受
                current_tour = new_tour
                current_len  = new_len
                accepted += 1
                # 更新全局最优解
                if new_len < best_len:
                    best_tour = new_tour.copy()
                    best_len  = new_len
            else:
                # 劣于当前解 -> 以 Boltzmann 概率接受
                prob = np.exp(-delta_E / T)
                if np.random.random() < prob:
                    current_tour = new_tour
                    current_len  = new_len
                    accepted += 1

        # 记录本轮数据
        history['best_lengths'].append(best_len)
        history['current_lengths'].append(current_len)
        history['temperatures'].append(T)
        history['acceptance_rates'].append(accepted / inner_iter)
        history['iterations'].append(outer_iter)

        # 降温
        T *= alpha

    return best_tour, best_len, history


# ============================================================
# 5. 可视化：路线图 + 收敛曲线
# ============================================================
def plot_single_result(cities: np.ndarray, tour: List[int],
                       history: dict, alpha: float, best_len: float,
                       save_path: str = None):
    """
    绘制单次实验结果：
      左图：最终最优路线图（城市点 + 连线）
      右图：收敛曲线（横轴迭代次数，纵轴路径长度）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'模拟退火 TSP  alpha={alpha}  |  学号: 125130024341  |  最优路径长度: {best_len:.2f}',
        fontsize=13, fontweight='bold'
    )

    # 图1：最优路线图
    tour_closed = tour + [tour[0]]
    coords = cities[tour_closed]
    ax1.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1.2, alpha=0.7, label='路线')
    ax1.scatter(cities[:, 0], cities[:, 1], c='tomato', s=60, zorder=5)
    ax1.scatter(cities[tour[0], 0], cities[tour[0], 1],
                c='limegreen', s=160, marker='*', zorder=6, label='起点')
    for idx, (x, y) in enumerate(cities):
        ax1.annotate(str(idx), (x, y), fontsize=6,
                     ha='center', va='center', color='black')
    ax1.set_title(f'最终最优路线图（alpha={alpha}）', fontsize=12)
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, COORD_RANGE + 20)
    ax1.set_ylim(-20, COORD_RANGE + 20)

    # 图2：收敛曲线
    iters = history['iterations']
    ax2.plot(iters, history['best_lengths'], 'b-', linewidth=2, label='最优路径长度')
    ax2.plot(iters, history['current_lengths'], 'r-', alpha=0.4,
             linewidth=0.8, label='当前路径长度')
    ax2.set_title(f'收敛曲线（alpha={alpha}）', fontsize=12)
    ax2.set_xlabel('迭代次数（外循环）')
    ax2.set_ylabel('路径长度')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  图表已保存：{save_path}')
    plt.show()


def plot_comparison(histories: dict, alphas: List[float]):
    """
    绘制三组退火系数的对比汇总图。
    """
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        '模拟退火 TSP — 退火系数 alpha 参数对比实验\n学号: 125130024341',
        fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 上行：各自的收敛曲线
    for col, (alpha, color) in enumerate(zip(alphas, colors)):
        ax = fig.add_subplot(gs[0, col])
        h = histories[alpha]
        ax.plot(h['iterations'], h['best_lengths'],
                color=color, linewidth=2, label=f'alpha={alpha}')
        ax.set_title(f'alpha = {alpha}', fontsize=12)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('最优路径长度')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    # 下行：汇总对比
    ax_all = fig.add_subplot(gs[1, :])
    for alpha, color in zip(alphas, colors):
        h = histories[alpha]
        final_best = h['best_lengths'][-1]
        ax_all.plot(h['iterations'], h['best_lengths'],
                    color=color, linewidth=2,
                    label=f'alpha={alpha}  最终最优={final_best:.2f}')
    ax_all.set_title('三组退火系数收敛对比', fontsize=12)
    ax_all.set_xlabel('迭代次数（外循环）')
    ax_all.set_ylabel('最优路径长度')
    ax_all.legend(fontsize=11)
    ax_all.grid(True, alpha=0.3)

    plt.savefig('tsp_comparison.png', dpi=150, bbox_inches='tight')
    print('对比图已保存：tsp_comparison.png')
    plt.show()


# ============================================================
# 6. 主程序入口
# ============================================================
if __name__ == '__main__':
    print('=' * 65)
    print('  TSP 模拟退火算法实验')
    print(f'  城市数量 N = {N_CITIES}   坐标范围: {COORD_RANGE}x{COORD_RANGE}')
    print(f'  初始温度 T0 = 1000 + {ID_LAST2}x20 = {T0}')
    print(f'  终止温度 T_final = {T_FINAL}')
    print(f'  每温度内循环次数 = {INNER_ITER}')
    print('=' * 65)

    # 生成个性化城市坐标
    cities      = generate_cities(SEED, N_CITIES)
    dist_matrix = calc_distance_matrix(cities)

    print('\n城市坐标生成完毕（前5座城市）：')
    for i in range(5):
        print(f'  城市 {i:2d}: X={cities[i, 0]:.2f}  Y={cities[i, 1]:.2f}')

    # 三组参数对比实验
    results   = {}
    histories = {}

    print('\n' + '-' * 65)
    print('  开始参数对比实验（alpha = 0.85 / 0.92 / 0.99）')
    print('-' * 65)

    for alpha in ALPHAS:
        print(f'\n[ alpha = {alpha} ]')
        t_start = time.time()

        best_tour, best_len, history = simulated_annealing(
            cities      = cities,
            dist_matrix = dist_matrix,
            alpha       = alpha,
            T0          = T0,
            T_final     = T_FINAL,
            inner_iter  = INNER_ITER,
            rng_seed    = SEED
        )

        elapsed     = time.time() - t_start
        total_outer = history['iterations'][-1]

        print(f'  外循环迭代次数 : {total_outer}')
        print(f'  最优路径长度   : {best_len:.2f}')
        print(f'  耗时           : {elapsed:.2f} 秒')

        results[alpha]   = (best_tour, best_len)
        histories[alpha] = history

        # 每个 alpha 单独保存一张路线+收敛图
        save_name = f'tsp_result_alpha{str(alpha).replace(".", "")}.png'
        plot_single_result(cities, best_tour, history, alpha, best_len, save_path=save_name)

    # 汇总对比图
    print('\n' + '-' * 65)
    print('  生成参数对比汇总图...')
    plot_comparison(histories, ALPHAS)

    # 打印结果汇总表
    print('\n' + '=' * 65)
    print('  实验结果汇总')
    print('=' * 65)
    print(f'  {"退火系数 alpha":<16} {"外循环迭代次数":<16} {"最优路径长度"}')
    print('  ' + '-' * 48)
    for alpha in ALPHAS:
        _, best_len = results[alpha]
        total_iter  = histories[alpha]['iterations'][-1]
        print(f'  {alpha:<16} {total_iter:<16} {best_len:.2f}')
    print('=' * 65)
    print('\n结论：退火越慢（alpha 越大）-> 迭代次数越多，路径越短（更优）')
    print('所有图表已保存，请检查当前目录下的 PNG 文件。')
