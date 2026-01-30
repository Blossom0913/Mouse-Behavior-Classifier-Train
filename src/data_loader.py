"""
Data Loader for Mouse Behavior Classification
整合特征提取和标签解析，提供完整的数据加载流程
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
        cache_dir: 缓存目录路径
    
    Returns:
        X: 特征数组 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        video_ids: 视频ID数组 (n_samples,)
        feature_names: 特征名称列表
        class_info: 类别信息字典
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    feature_cache = cache_dir / 'features.pkl'
    
    
    
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
    
    
    
    all_labels = parse_all_annotations(annot_folder, n_frames_dict, experiment)
    
    
    
    videos_with_annotations = set()
    for video_id, labels in all_labels.items():
        if np.any(labels >= 0):  # 至少有一个有效标注
            videos_with_annotations.add(video_id)
    
    total_videos = len(np.unique(list(all_labels.keys())))
    valid_videos = len(videos_with_annotations)
    invalid_videos = total_videos - valid_videos
    
    
    if invalid_videos > 0:
        invalid_list = sorted([v for v in all_labels.keys() if v not in videos_with_annotations])
        print(f"  Excluded videos: {invalid_list}")
    
    
    
    X_list = []
    y_list = []
    video_id_list = []  # 记录每个样本对应的video_id
    
    # 特征列（排除video_id和frame）
    feature_cols = [c for c in features_df.columns if c not in ['video_id', 'frame']]
    
    for video_id in sorted(videos_with_annotations):
        video_features = features_df[features_df['video_id'] == video_id][feature_cols].values
        video_labels = all_labels[video_id]
        
        # 确保长度一致
        min_len = min(len(video_features), len(video_labels))
        if min_len > 0:
            X_list.append(video_features[:min_len])
            y_list.append(video_labels[:min_len])
            video_id_list.append(np.full(min_len, video_id))  # 记录每帧对应的video_id
    
    if not X_list:
        raise ValueError("No valid data found! Check your data paths.")
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    video_ids = np.concatenate(video_id_list)  # 合并video_ids
    
    
    
    # =========================================================================
    # Step 5: 过滤无效标签
    # =========================================================================
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]
    video_ids = video_ids[valid_mask]  # 同步过滤video_ids
    
    
    
    # =========================================================================
    # Step 6: 类别信息
    # =========================================================================
    class_names = get_class_names(experiment)
    n_classes = len(class_names)
    
    class_counts = np.bincount(y.astype(int), minlength=n_classes)
    
    
    
    class_info = {
        'n_classes': n_classes,
        'class_names': class_names,
        'class_counts': class_counts.tolist()
    }
    
    # =========================================================================
    # Step 7: 处理NaN
    # =========================================================================
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"\nFound {nan_count} NaN values, using median imputation...")
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        print("✓ NaN values imputed")
    
    
    
    return X, y, video_ids, feature_cols, class_info


def create_video_level_splits(
    X,
    y,
    video_ids,
    test_size=0.3,
    val_size=0.5,
    random_state=42,
    train_video_count=None,
    val_video_count=None,
    test_video_count=None,
):
    """
    创建视频级别的数据分割，防止时间序列泄漏
    
    Args:
        X: 特征数组 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        video_ids: 视频ID数组 (n_samples,) - 每个样本对应的视频ID
        test_size: 测试集比例
        val_size: 验证集比例 (从临时集中)
        random_state: 随机种子
        train_video_count: 训练集视频数（优先于比例）
        val_video_count: 验证集视频数（优先于比例）
        test_video_count: 测试集视频数（优先于比例）
    
    Returns:
        splits: dict containing train/val/test indices and video sets
    """
    unique_videos = np.unique(video_ids)
    rng = np.random.default_rng(random_state)
    print(f"\n{'='*60}")
    print(f"Video-Level Split (Preventing Temporal Leakage)")
    print(f"{'='*60}")
    total_videos = len(unique_videos)
    print(f"Total unique videos: {total_videos}")
    
    using_fixed_counts = (
        train_video_count is not None
        and val_video_count is not None
        and test_video_count is not None
    )
    
    if using_fixed_counts:
        required = train_video_count + val_video_count + test_video_count
        if required > total_videos:
            raise ValueError(
                f"Requested {required} videos (train {train_video_count}, val {val_video_count}, "
                f"test {test_video_count}) but only {total_videos} available"
            )
        perm = rng.permutation(unique_videos)
        train_videos = set(perm[:train_video_count])
        val_videos = set(perm[train_video_count:train_video_count + val_video_count])
        test_videos = set(
            perm[train_video_count + val_video_count:train_video_count + val_video_count + test_video_count]
        )
        
        train_idx = np.where(np.isin(video_ids, list(train_videos)))[0]
        val_idx = np.where(np.isin(video_ids, list(val_videos)))[0]
        test_idx = np.where(np.isin(video_ids, list(test_videos)))[0]
    else:
        # Step 1: Split videos into train+val vs test
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss1.split(X, y, groups=video_ids))
        
        # Step 2: Split train+val videos into train vs val
        X_tv = X[train_val_idx]
        y_tv = y[train_val_idx]
        video_ids_tv = video_ids[train_val_idx]
        
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx_rel, val_idx_rel = next(gss2.split(X_tv, y_tv, groups=video_ids_tv))
        
        # Convert relative indices to absolute
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        
        # Get video sets
        train_videos = set(video_ids[train_idx])
        val_videos = set(video_ids[val_idx])
        test_videos = set(video_ids[test_idx])
    
    # Verify no video overlap
    overlap_train_val = train_videos & val_videos
    overlap_train_test = train_videos & test_videos
    overlap_val_test = val_videos & test_videos
    
    assert len(overlap_train_val) == 0, f"Video overlap between train and val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Video overlap between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Video overlap between val and test: {overlap_val_test}"
    
    
    
    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'train_videos': train_videos,
        'val_videos': val_videos,
        'test_videos': test_videos
    }


def verify_split_integrity(splits, video_ids, y, verbose=False):
    """验证分割完整性；默认静默，仅在 verbose=True 时打印详细信息。"""

    train_videos = splits['train_videos']
    val_videos = splits['val_videos']
    test_videos = splits['test_videos']

    overlap_train_val = train_videos & val_videos
    overlap_train_test = train_videos & test_videos
    overlap_val_test = val_videos & test_videos

    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    unique_classes = np.unique(y)

    report = {
        'video_overlap_pass': len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0,
        'overlap_train_val': list(overlap_train_val),
        'overlap_train_test': list(overlap_train_test),
        'overlap_val_test': list(overlap_val_test),
        'train_counts': np.bincount(y_train.astype(int), minlength=len(unique_classes)),
        'val_counts': np.bincount(y_val.astype(int), minlength=len(unique_classes)),
        'test_counts': np.bincount(y_test.astype(int), minlength=len(unique_classes)),
    }

    # samples per video
    for split_name, split_videos, split_idx in [
        ('train', train_videos, train_idx),
        ('val', val_videos, val_idx),
        ('test', test_videos, test_idx)
    ]:
        video_ids_split = video_ids[split_idx]
        samples_per_video = {vid: np.sum(video_ids_split == vid) for vid in split_videos}
        report[f'{split_name}_video_lengths'] = samples_per_video

    

    return report


def create_data_loaders(
    X,
    y,
    video_ids,
    batch_size=256,
    test_size=0.3,
    val_size=0.5,
    random_state=42,
    train_video_count=None,
    val_video_count=None,
    test_video_count=None,
):
    """
    创建训练/验证/测试 DataLoader (使用视频级别分割)
    
    Args:
        X: 特征数组
        y: 标签数组
        video_ids: 视频ID数组
        batch_size: 批量大小
        test_size: 测试集比例 (从总数据中)
        val_size: 验证集比例 (从临时集中)
        random_state: 随机种子
        train_video_count: 训练集视频数（优先于比例）
        val_video_count: 验证集视频数（优先于比例）
        test_video_count: 测试集视频数（优先于比例）
    
    Returns:
        train_loader: 训练DataLoader
        val_loader: 验证DataLoader
        test_loader: 测试DataLoader
        scaler: StandardScaler实例
        splits: 分割信息字典
    """
    # 使用视频级别分割
    splits = create_video_level_splits(
        X,
        y,
        video_ids,
        test_size,
        val_size,
        random_state,
        train_video_count,
        val_video_count,
        test_video_count,
    )
    
    # 验证分割完整性
    verify_split_integrity(splits, video_ids, y)
    
    # 提取数据
    X_train = X[splits['train_idx']]
    X_val = X[splits['val_idx']]
    X_test = X[splits['test_idx']]
    
    y_train = y[splits['train_idx']]
    y_val = y[splits['val_idx']]
    y_test = y[splits['test_idx']]
    
    # 创建数据集
    train_dataset = MouseBehaviorDataset(X_train, y_train, fit_scaler=True)
    val_dataset = MouseBehaviorDataset(X_val, y_val, scaler=train_dataset.scaler)
    test_dataset = MouseBehaviorDataset(X_test, y_test, scaler=train_dataset.scaler)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, train_dataset.scaler, splits


def create_numpy_splits(
    X,
    y,
    video_ids,
    test_size=0.3,
    val_size=0.5,
    random_state=42,
    train_video_count=None,
    val_video_count=None,
    test_video_count=None,
):
    """
    创建numpy数组的数据分割（用于非PyTorch模型，使用视频级别分割）
    
    Args:
        X: 特征数组
        y: 标签数组
        video_ids: 视频ID数组
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        train_video_count: 训练集视频数（优先于比例）
        val_video_count: 验证集视频数（优先于比例）
        test_video_count: 测试集视频数（优先于比例）
    
    Returns:
        splits: dict containing X_train, X_val, X_test, y_train, y_val, y_test
        scaler: StandardScaler实例
        split_info: 分割信息字典
    """
    # 使用视频级别分割
    split_info = create_video_level_splits(
        X,
        y,
        video_ids,
        test_size,
        val_size,
        random_state,
        train_video_count,
        val_video_count,
        test_video_count,
    )
    
    # 验证分割完整性
    verify_split_integrity(split_info, video_ids, y)
    
    # 提取数据
    X_train = X[split_info['train_idx']]
    X_val = X[split_info['val_idx']]
    X_test = X[split_info['test_idx']]
    
    y_train = y[split_info['train_idx']]
    y_val = y[split_info['val_idx']]
    y_test = y[split_info['test_idx']]
    
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
    
    return splits, scaler, split_info




# ============== 测试代码 ==============
if __name__ == "__main__":
    
    
    # 设置路径
    csv_folder = '../data/dlc_csv'
    annot_folder = '../data/annotations'
    cache_dir = '../data/cache'
    
    # Step 1: 准备数据集
    print("\n[Step 1] Loading dataset...")
    X, y, video_ids, feature_names, class_info = prepare_dataset(
        csv_folder=csv_folder,
        annot_folder=annot_folder,
        experiment='behavior',  # 3分类
        cache_dir=cache_dir
    )
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(X):,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Unique videos: {len(np.unique(video_ids))}")
    print(f"  Classes: {class_info['n_classes']}")
    print(f"  Class names: {class_info['class_names']}")
    
    # Step 2: 创建视频级别分割的DataLoaders
    print(f"\n[Step 2] Creating video-level splits...")
    train_loader, val_loader, test_loader, scaler, splits = create_data_loaders(
        X, y, video_ids,
        batch_size=256,
        test_size=0.2,  # 20% for test
        val_size=0.25,  # 25% of remaining for val (20% of total)
        random_state=42
    )
    
    print(f"\nDataLoader Summary:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    
    train_batch = next(iter(train_loader))
    
   
    train_vids = splits['train_videos']
    val_vids = splits['val_videos']
    test_vids = splits['test_videos']
    
    all_vids = train_vids | val_vids | test_vids
    total_vids = len(all_vids)
    
    
    
