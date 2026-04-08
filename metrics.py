"""
评估指标模块
实现OSPA和OSPA(2)指标
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class OSPAMetric:
    """
    OSPA (Optimal Sub-Pattern Assignment) 指标
    
    参考文献：
    D. Schuhmacher, B.-T. Vo, and B.-N. Vo, 
    "A consistent metric for performance evaluation of multi-object filters," 
    IEEE Trans. Signal Process., vol. 56, no. 8, pp. 3447–3457, 2008.
    """
    def __init__(self, c=1.0, p=1):
        """
        Args:
            c: cutoff distance (截断距离)
            p: order parameter (阶参数)
        """
        self.c = c
        self.p = p
    
    def __call__(self, X, Y):
        """
        计算两个点集之间的OSPA距离
        
        Args:
            X: numpy array of shape [n, d] - 估计的目标状态
            Y: numpy array of shape [m, d] - 真实的目标状态
        
        Returns:
            ospa_dist: OSPA距离
            ospa_loc: 定位误差部分
            ospa_card: 基数误差部分
        """
        n = len(X)
        m = len(Y)
        
        # 空集情况
        if n == 0 and m == 0:
            return 0.0, 0.0, 0.0
        elif n == 0:
            return self.c, 0.0, self.c
        elif m == 0:
            return self.c, 0.0, self.c
        
        # 计算成对距离矩阵
        D = cdist(X, Y, metric='euclidean')
        
        # 截断距离
        D = np.minimum(D, self.c)
        
        # 使用匈牙利算法求解最优分配
        if n <= m:
            # 估计数 <= 真实数
            row_ind, col_ind = linear_sum_assignment(D)
            loc_error = D[row_ind, col_ind].sum()
            card_penalty = (m - n) * self.c ** self.p
        else:
            # 估计数 > 真实数
            row_ind, col_ind = linear_sum_assignment(D.T)
            loc_error = D[col_ind, row_ind].sum()
            card_penalty = (n - m) * self.c ** self.p
        
        # OSPA距离
        ospa_dist = (1.0 / max(n, m) * (loc_error ** self.p + card_penalty)) ** (1.0 / self.p)
        
        # 分解为定位和基数误差
        ospa_loc = (1.0 / max(n, m) * loc_error ** self.p) ** (1.0 / self.p)
        ospa_card = (1.0 / max(n, m) * card_penalty) ** (1.0 / self.p)
        
        return ospa_dist, ospa_loc, ospa_card


class OSPA2Metric:
    """
    OSPA(2) 指标 - 用于轨迹评估
    
    参考文献：
    M. Beard, B. T. Vo, and B.-N. Vo, 
    "OSPA(2): Using the OSPA metric to evaluate multi-target tracking performance," 
    in 2017 International Conference on Control, Automation and Information Sciences (ICCAIS). 
    IEEE, 2017, pp. 86-91.
    """
    def __init__(self, c=1.0, p=1, win_len=10):
        """
        Args:
            c: cutoff distance
            p: order parameter
            win_len: window length for trajectory comparison
        """
        self.c = c
        self.p = p
        self.win_len = win_len
        self.ospa = OSPAMetric(c=c, p=p)
    
    def __call__(self, X_traj, Y_traj):
        """
        计算两组轨迹之间的OSPA(2)距离
        
        Args:
            X_traj: list of trajectories (each trajectory is [T, d])
                    估计的轨迹
            Y_traj: list of trajectories (each trajectory is [T, d])
                    真实的轨迹
        
        Returns:
            ospa2_dist: OSPA(2)距离
            ospa2_loc: 定位误差部分
            ospa2_card: 基数误差部分
        """
        n = len(X_traj)
        m = len(Y_traj)
        
        if n == 0 and m == 0:
            return 0.0, 0.0, 0.0
        elif n == 0:
            return self.c, 0.0, self.c
        elif m == 0:
            return self.c, 0.0, self.c
        
        # 计算轨迹间的距离矩阵
        D = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                D[i, j] = self._trajectory_distance(X_traj[i], Y_traj[j])
        
        # 截断
        D = np.minimum(D, self.c)
        
        # 匈牙利算法
        if n <= m:
            row_ind, col_ind = linear_sum_assignment(D)
            loc_error = D[row_ind, col_ind].sum()
            card_penalty = (m - n) * self.c ** self.p
        else:
            row_ind, col_ind = linear_sum_assignment(D.T)
            loc_error = D[col_ind, row_ind].sum()
            card_penalty = (n - m) * self.c ** self.p
        
        # OSPA(2)距离
        ospa2_dist = (1.0 / max(n, m) * (loc_error ** self.p + card_penalty)) ** (1.0 / self.p)
        ospa2_loc = (1.0 / max(n, m) * loc_error ** self.p) ** (1.0 / self.p)
        ospa2_card = (1.0 / max(n, m) * card_penalty) ** (1.0 / self.p)
        
        return ospa2_dist, ospa2_loc, ospa2_card
    
    def _trajectory_distance(self, traj1, traj2):
        """
        计算两条轨迹之间的距离
        使用时间平均的OSPA距离
        
        Args:
            traj1: [T1, d]
            traj2: [T2, d]
        
        Returns:
            distance: 轨迹距离
        """
        T1 = len(traj1)
        T2 = len(traj2)
        
        if T1 != T2:
            # 如果长度不同，返回最大距离
            return self.c
        
        # 对每个时间步计算OSPA距离
        distances = []
        for t in range(T1):
            d = np.linalg.norm(traj1[t] - traj2[t])
            d = min(d, self.c)
            distances.append(d)
        
        # 返回平均距离
        return np.mean(distances)


class TrackingMetrics:
    """跟踪性能综合评估"""
    def __init__(self, c=1.0, p=1):
        self.ospa = OSPAMetric(c=c, p=p)
        self.ospa2 = OSPA2Metric(c=c, p=p)
        
        # 统计量
        self.reset()
    
    def reset(self):
        """重置统计"""
        self.ospa_dists = []
        self.ospa_locs = []
        self.ospa_cards = []
        self.ospa2_dists = []
        self.ospa2_locs = []
        self.ospa2_cards = []
    
    def update_ospa(self, X, Y):
        """更新OSPA统计"""
        dist, loc, card = self.ospa(X, Y)
        self.ospa_dists.append(dist)
        self.ospa_locs.append(loc)
        self.ospa_cards.append(card)
        return dist, loc, card
    
    def update_ospa2(self, X_traj, Y_traj):
        """更新OSPA(2)统计"""
        dist, loc, card = self.ospa2(X_traj, Y_traj)
        self.ospa2_dists.append(dist)
        self.ospa2_locs.append(loc)
        self.ospa2_cards.append(card)
        return dist, loc, card
    
    def get_ospa_stats(self):
        """获取OSPA统计结果"""
        if len(self.ospa_dists) == 0:
            return {'mean': 0.0, 'std': 0.0, 'loc_mean': 0.0, 'card_mean': 0.0}
        
        return {
            'mean': np.mean(self.ospa_dists),
            'std': np.std(self.ospa_dists),
            'loc_mean': np.mean(self.ospa_locs),
            'card_mean': np.mean(self.ospa_cards)
        }
    
    def get_ospa2_stats(self):
        """获取OSPA(2)统计结果"""
        if len(self.ospa2_dists) == 0:
            return {'mean': 0.0, 'std': 0.0, 'loc_mean': 0.0, 'card_mean': 0.0}
        
        return {
            'mean': np.mean(self.ospa2_dists),
            'std': np.std(self.ospa2_dists),
            'loc_mean': np.mean(self.ospa2_locs),
            'card_mean': np.mean(self.ospa2_cards)
        }


if __name__ == "__main__":
    # 测试OSPA指标
    print("Testing OSPA metric...")
    
    ospa = OSPAMetric(c=1.0, p=1)
    
    # 测试案例1：相同数量的点
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    Y = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
    
    dist, loc, card = ospa(X, Y)
    print(f"Test 1 - OSPA distance: {dist:.4f}, Loc: {loc:.4f}, Card: {card:.4f}")
    
    # 测试案例2：不同数量的点
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    Y = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
    
    dist, loc, card = ospa(X, Y)
    print(f"Test 2 - OSPA distance: {dist:.4f}, Loc: {loc:.4f}, Card: {card:.4f}")
    
    # 测试案例3：空集
    X = np.array([])
    Y = np.array([[0.1, 0.1], [1.1, 1.1]])
    
    X = X.reshape(0, 2)
    dist, loc, card = ospa(X, Y)
    print(f"Test 3 - OSPA distance: {dist:.4f}, Loc: {loc:.4f}, Card: {card:.4f}")
    
    # 测试OSPA(2)
    print("\nTesting OSPA(2) metric...")
    
    ospa2 = OSPA2Metric(c=1.0, p=1)
    
    # 创建两条轨迹
    traj1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    traj2 = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1]])
    traj3 = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    
    X_traj = [traj1, traj2]
    Y_traj = [traj2, traj3]
    
    dist, loc, card = ospa2(X_traj, Y_traj)
    print(f"OSPA(2) distance: {dist:.4f}, Loc: {loc:.4f}, Card: {card:.4f}")
    
    print("\nMetrics test passed!")
