"""
Data Loader for Mouse Behavior Classification
整合特征提取和标签解析，提供完整的数据加载流程
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path

try:
    from .feature_extraction import DLCFeatureExtractor, get_feature_names
    from .label_parser import AnnotationParser, parse_all_annotations, get_class_names
except ImportError:
    from feature_extraction import DLCFeatureExtractor, get_feature_names
    from label_parser import AnnotationParser, parse_all_annotations, get_class_names


class MouseBehaviorDataset(Dataset):
    """小鼠行为分类数据集"""
    
    def __init__(self, features, labels, scaler=None, fit_scaler=False):
        """
        Args:
            features: 特征数组 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
            scaler: StandardScaler实例
            fit_scaler: 是否fit scaler (仅训练集为True)
        """
        # 过滤无效标签 (标签为-1的样本)
        valid_mask = labels >= 0
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # 标准化
        if scaler is None:
            scaler = StandardScaler()
        
        if fit_scaler:
            self.features = torch.FloatTensor(scaler.fit_transform(features))
        else:
            self.features = torch.FloatTensor(scaler.transform(features))
        
        self.labels = torch.LongTensor(labels)
        self.scaler = scaler
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_dataset(csv_folder, annot_folder, experiment='behavior', cache_dir='./cache'):
    """
    准备完整数据集
    
    Args:
        csv_folder: DLC CSV文件夹路径
        annot_folder: 标注文件夹路径
        experiment: 'behavior' (3类) 或 'aggression' (7类)
        cache_dir: 缓存目录
    
    Returns:
        X: 特征数组 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        feature_names: 特征名称列表
        class_info: 类别信息字典
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    feature_cache = cache_dir / 'features.pkl'
    
    # =========================================================================
    # Step 1: 提取特征
    # =========================================================================
    print("=" * 60)
    print("Step 1: Extracting features from DLC files...")
    print("=" * 60)
    
    if feature_cache.exists():
        print(f"Loading cached features from {feature_cache}")
        features_df = pd.read_pickle(feature_cache)
        # 重建n_frames_dict
        n_frames_dict = features_df.groupby('video_id')['frame'].max().to_dict()
        n_frames_dict = {k: v + 1 for k, v in n_frames_dict.items()}
    else:
        extractor = DLCFeatureExtractor(fps=30)
        features_df, n_frames_dict = extractor.extract_from_folder(csv_folder, feature_cache)
    
    print(f"Total features shape: {features_df.shape}")
    
    # =========================================================================
    # Step 2: 解析标签
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"Step 2: Parsing annotations for {experiment} experiment...")
    print("=" * 60)
    
    all_labels = parse_all_annotations(annot_folder, n_frames_dict, experiment)
    
    # =========================================================================
    # Step 3: 合并特征和标签
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Merging features and labels...")
    print("=" * 60)
    
    X_list = []
    y_list = []
    
    # 特征列（排除video_id和frame）
    feature_cols = [c for c in features_df.columns if c not in ['video_id', 'frame']]
    
    for video_id in sorted(all_labels.keys()):
        video_features = features_df[features_df['video_id'] == video_id][feature_cols].values
        video_labels = all_labels[video_id]
        
        # 确保长度一致
        min_len = min(len(video_features), len(video_labels))
        if min_len > 0:
            X_list.append(video_features[:min_len])
            y_list.append(video_labels[:min_len])
    
    if not X_list:
        raise ValueError("No valid data found! Check your data paths.")
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    print(f"Combined X shape: {X.shape}")
    print(f"Combined y shape: {y.shape}")
    
    # =========================================================================
    # Step 4: 过滤无效标签
    # =========================================================================
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\nAfter filtering invalid labels:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # =========================================================================
    # Step 5: 类别信息
    # =========================================================================
    class_names = get_class_names(experiment)
    n_classes = len(class_names)
    
    class_counts = np.bincount(y.astype(int), minlength=n_classes)
    
    print(f"\nClass distribution:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        pct = 100 * count / len(y) if len(y) > 0 else 0
        print(f"  Class {i} ({name}): {count:,} samples ({pct:.1f}%)")
    
    class_info = {
        'n_classes': n_classes,
        'class_names': class_names,
        'class_counts': class_counts.tolist()
    }
    
    # =========================================================================
    # Step 6: 处理NaN
    # =========================================================================
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"\nFound {nan_count} NaN values, using median imputation...")
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        print("✓ NaN values imputed")
    
    print(f"\n✓ Dataset prepared successfully!")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Classes: {n_classes}")
    
    return X, y, feature_cols, class_info


def create_data_loaders(X, y, batch_size=256, test_size=0.3, val_size=0.5, random_state=42):
    """
    创建训练/验证/测试 DataLoader
    
    Args:
        X: 特征数组
        y: 标签数组
        batch_size: 批量大小
        test_size: 测试集比例 (从总数据中)
        val_size: 验证集比例 (从临时集中)
        random_state: 随机种子
    
    Returns:
        train_loader: 训练DataLoader
        val_loader: 验证DataLoader
        test_loader: 测试DataLoader
        scaler: StandardScaler实例
    """
    # 划分数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 创建数据集
    train_dataset = MouseBehaviorDataset(X_train, y_train, fit_scaler=True)
    val_dataset = MouseBehaviorDataset(X_val, y_val, scaler=train_dataset.scaler)
    test_dataset = MouseBehaviorDataset(X_test, y_test, scaler=train_dataset.scaler)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset.scaler


def create_numpy_splits(X, y, test_size=0.3, val_size=0.5, random_state=42):
    """
    创建numpy数组的数据分割（用于非PyTorch模型）
    
    Returns:
        splits: dict containing X_train, X_val, X_test, y_train, y_val, y_test
        scaler: StandardScaler实例
    """
    # 划分数据
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    return splits, scaler


# ============== 测试代码 ==============
if __name__ == "__main__":
    print("Data Loader Module")
    print("=" * 50)
    print("\nThis module provides:")
    print("  - prepare_dataset(): Load and prepare full dataset")
    print("  - create_data_loaders(): Create PyTorch DataLoaders")
    print("  - create_numpy_splits(): Create numpy array splits")
    print("  - MouseBehaviorDataset: PyTorch Dataset class")
    print("\n✓ Data loader module ready!")
