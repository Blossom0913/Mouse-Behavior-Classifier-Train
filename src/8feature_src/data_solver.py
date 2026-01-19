import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from scipy.stats import skew

def safe_skew(data, eps=1e-6):
    """安全计算偏度"""
    if np.var(data) < eps:  # 方差小于阈值时返回0
        return 0.0
    return skew(data, bias=False)


class OnlineStats:
    """数值稳定的在线统计量计算"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0

    def update(self, x):
        n1 = self.n
        self.n += 1
        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term = delta * delta_n * n1
        self.mean += delta_n
        self.M3 += term * delta_n2 * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def skewness(self):
        if self.n < 3 or self.variance < 1e-6:
            return 0.0
        return (self.M3 / self.n) / (self.variance**1.5)


# # 渐进式训练
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, 
#     max_lr=1e-3,
#     steps_per_epoch=len(train_loader),
#     epochs=200,
#     pct_start=0.3
# )

def data_load(coords_path,labels_path):


    coords_df = pd.read_excel(coords_path)
    labels_df = pd.read_excel(labels_path)


    coords_df = coords_df.interpolate(method='linear', axis=0)
    coords_df = coords_df.bfill()
    coords_df = coords_df.ffill()

    # 确保特征和标签数据完全对齐
    coords_df = coords_df.dropna()
    labels_df = labels_df[labels_df['time'].isin(coords_df.index)]

    print("coords describe:", coords_df.describe())

    labels_df = labels_df[labels_df['time'].isin(coords_df.index)]

    return coords_df, labels_df

def create_sequences_velo(X, y, seq_length):

    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        # 原始坐标特征
        raw_features = X[i:i+seq_length]
        # 计算速度（一阶差分）
        velocity = np.diff(raw_features, axis=0, prepend=raw_features[0:1])
        # 计算加速度（二阶差分）
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        # 合并特征 [原始坐标, 速度, 加速度]
        combined_features = np.concatenate([raw_features, velocity, acceleration], axis=1)
        X_seq.append(combined_features)
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)


def create_sequences(X, y, seq_length):

    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

def load_mouse_data(feature_path, label_path):
    """加载并预处理小鼠行为数据"""
    # 读取特征数据（跳过首行标题）
    features = pd.read_excel(feature_path, header=None, skiprows=1, engine='openpyxl')
    # 读取标签数据（假设最后一列为标签）
    labels = pd.read_excel(label_path, header=None, skiprows=1, engine='openpyxl')
    
    # 数据清洗
    features = features.interpolate(method='linear').bfill().ffill()
    labels = labels.iloc[:, -1].values  # 假设标签在最后一列
    
    # 转换为numpy数组
    X = features.values.astype(np.float32)
    y = labels.astype(int)
    
    # 打印数据统计
    print("特征数据统计:")
    print(pd.DataFrame(X).describe())
    print("\n标签分布:")
    print(pd.Series(y).value_counts(normalize=True))
    
    return X, y

def create_mouse_sequences(X, y, seq_length=64, 
                          window_sizes=[16, 32],
                          diff_orders=[1, 2]):
    """
    小鼠行为序列特征生成
    输入维度: (n_samples, 8)
    列索引对应:
    0: 小鼠1头速度
    1: 小鼠2头速度
    2: 小鼠1身体速度
    3: 小鼠2身体速度
    4: 小鼠12头间距
    5: 头-尾距离
    6: 小鼠12身体间距
    7: 头-尾夹角
    """
    # 特征分组定义
    HEAD_SPEED_COLS = [0, 1]      # 头部速度
    BODY_SPEED_COLS = [2, 3]      # 身体速度
    DISTANCE_COLS = [4, 5, 6]     # 距离特征
    ANGLE_COL = [7]               # 角度特征
    
    # 初始化标准化器
    scalers = {
        'head_speed': RobustScaler(quantile_range=(5, 95)),
        'body_speed': RobustScaler(quantile_range=(5, 95)),
        'distance': StandardScaler(),
        'angle': MinMaxScaler(feature_range=(-np.pi, np.pi))
    }

    # 滑动窗口统计函数
    def sliding_stats(data, window_size, stat_func):
        if window_size > data.shape[0]:
            return np.zeros_like(data)
            
        padded = np.pad(data, ((window_size-1,0), (0,0)), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, data.shape[1]))
        
        # 修正后的聚合逻辑
        result = stat_func(windows, axis=-1)
        
        # 处理多维度情况
        if result.ndim == 3:
            result = result.reshape(result.shape[0], -1)
        return result[:len(data)]

    sequences = []
    labels_seq = []
    
    for i in range(len(X) - seq_length):
        window = X[i:i+seq_length]
        
        # ===== 特征工程 =====
        feature_groups = []
        
        # 1. 原始特征
        feature_groups.append(window.copy())
        
        # 2. 差分特征
        diff_features = []
        for order in diff_orders:
            diff = window.copy()
            for _ in range(order):
                diff = np.diff(diff, axis=0, prepend=diff[[0], :])
            diff_features.append(diff)
        feature_groups.extend(diff_features)
        
        # 3. 统计特征
        stat_features = []
        for w_size in window_sizes:
            # 分组统计计算
            for group, cols in [
                ('head_speed', HEAD_SPEED_COLS),
                ('body_speed', BODY_SPEED_COLS),
                ('distance', DISTANCE_COLS),
                ('angle', ANGLE_COL)
            ]:
                group_data = window[:, cols]
                
                stats_group = [
                    sliding_stats(group_data, w_size, np.mean),
                    sliding_stats(group_data, w_size, np.std),
                    sliding_stats(group_data, w_size, np.max),
                    sliding_stats(group_data, w_size, np.min),
                    sliding_stats(group_data, w_size, stats.skew)
                ]
                stat_features.extend(stats_group)
        
        # 4. 时序特征
        time_stamp = np.linspace(0, 1, seq_length)[:, None]
        time_feat = np.concatenate([
            np.sin(2 * np.pi * time_stamp),
            np.cos(2 * np.pi * time_stamp)
        ], axis=1)
        feature_groups.append(time_feat)
        
        # ===== 特征拼接 =====
        combined = np.concatenate(feature_groups + stat_features, axis=1)
        
        # ===== 分组标准化 =====
        start_idx = 0
        for group, cols in [
            ('head_speed', HEAD_SPEED_COLS),
            ('body_speed', BODY_SPEED_COLS),
            ('distance', DISTANCE_COLS),
            ('angle', ANGLE_COL)
        ]:
            n_cols = len(cols) * (1 + len(diff_orders))
            end_idx = start_idx + n_cols
            if n_cols > 0:
                combined[:, start_idx:end_idx] = scalers[group].fit_transform(combined[:, start_idx:end_idx])
            start_idx = end_idx
        
        sequences.append(combined)
        labels_seq.append(y[i + seq_length - 1])

    return np.array(sequences), np.array(labels_seq)

# 使用示例
if __name__ == "__main__":
    # 文件路径
    feature_path = "data/train_batch5/merged_feature.xlsx"
    label_path = "data/train_batch5/merged_label.xlsx"
    
    # 加载数据
    X_raw, y_raw = load_mouse_data(feature_path, label_path)
    
    # 生成序列
    X_seq, y_seq = create_mouse_sequences(
        X_raw, y_raw,
        seq_length=16,
        window_sizes=[4, 8],
        diff_orders=[1, 2]
    )
    
    print("\n处理后的数据维度:")
    print(f"样本数: {X_seq.shape[0]}")
    print(f"序列长度: {X_seq.shape[1]}")
    print(f"特征维度: {X_seq.shape[2]}")