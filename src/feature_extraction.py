"""
Feature Extraction from DeepLabCut CSV files
从DLC多动物追踪结果中提取动态特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


class DLCFeatureExtractor:
    """从DeepLabCut输出提取动态特征"""
    
    def __init__(self, fps=30, smooth_sigma=1.0, likelihood_threshold=0.6):
        """
        Args:
            fps: 视频帧率，用于计算速度
            smooth_sigma: 高斯平滑参数
            likelihood_threshold: 低于此阈值的点被视为不可靠
        """
        self.fps = fps
        self.smooth_sigma = smooth_sigma
        self.likelihood_threshold = likelihood_threshold
        
        # 身体部位索引 (每个个体5个部位)
        self.bodyparts = ['top', 'body', 'tail', 'tail-m', 'tail-e']
    
    def load_dlc_csv(self, csv_path):
        """
        加载DLC的CSV文件
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            coords: (n_frames, 2, 5, 2) 坐标数组 [frames, individuals, bodyparts, xy]
            likelihood: (n_frames, 2, 5) 置信度数组
            n_frames: 帧数
        """
        # 读取多层表头 (scorer, individuals, bodyparts, coords)
        df = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
        
        # 转换为numpy数组
        data = df.values.astype(np.float32)
        n_frames = len(data)
        
        # 重塑为 (n_frames, 2 individuals, 5 bodyparts, 3 coords)
        # 3 = [x, y, likelihood]
        reshaped = data.reshape(n_frames, 2, 5, 3)
        
        # 分离坐标和置信度
        coords = reshaped[:, :, :, :2]       # (n_frames, 2, 5, 2)
        likelihood = reshaped[:, :, :, 2]    # (n_frames, 2, 5)
        
        return coords, likelihood, n_frames
    
    def preprocess_coords(self, coords, likelihood):
        """
        预处理：低置信度点插值填充
        
        Args:
            coords: (n_frames, 2, 5, 2) 坐标数组
            likelihood: (n_frames, 2, 5) 置信度数组
            
        Returns:
            coords_clean: 处理后的坐标数组
        """
        coords_clean = coords.copy()
        
        for ind in range(2):  # 2个个体
            for bp in range(5):  # 5个身体部位
                mask = likelihood[:, ind, bp] < self.likelihood_threshold
                if mask.any():
                    # 线性插值
                    for dim in range(2):  # x, y
                        values = coords_clean[:, ind, bp, dim]
                        valid_idx = np.where(~mask)[0]
                        invalid_idx = np.where(mask)[0]
                        if len(valid_idx) > 1:
                            values[invalid_idx] = np.interp(
                                invalid_idx, valid_idx, values[valid_idx]
                            )
                        coords_clean[:, ind, bp, dim] = values
        
        return coords_clean
    
    def compute_speed(self, coords):
        """
        计算速度 (像素/秒)
        
        Args:
            coords: (n_frames, 2) 表示 x, y 坐标
            
        Returns:
            speed: (n_frames,) 速度数组
        """
        # 平滑
        if self.smooth_sigma > 0:
            coords = gaussian_filter1d(coords, sigma=self.smooth_sigma, axis=0)
        
        # 计算帧间位移
        diff = np.diff(coords, axis=0)
        speed = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2) * self.fps
        
        # 补齐第一帧
        speed = np.concatenate([[speed[0]], speed])
        
        return speed
    
    def compute_distance(self, coords1, coords2):
        """计算两点间距离"""
        return np.sqrt(np.sum((coords1 - coords2)**2, axis=1))
    
    def compute_angle(self, p1, p2):
        """计算两点连线与水平轴的角度 (弧度)"""
        diff = p2 - p1
        return np.arctan2(diff[:, 1], diff[:, 0])
    
    def extract_features(self, csv_path):
        """
        从单个CSV文件提取所有特征
        
        Args:
            csv_path: DLC CSV文件路径
            
        Returns:
            features_df: 特征DataFrame
            n_frames: 帧数
        """
        # 加载数据
        coords, likelihood, n_frames = self.load_dlc_csv(csv_path)
        
        # 预处理
        coords = self.preprocess_coords(coords, likelihood)
        
        # 提取各身体部位坐标
        # Individual 1
        top1 = coords[:, 0, 0, :]      # (n_frames, 2)
        body1 = coords[:, 0, 1, :]
        tail1 = coords[:, 0, 2, :]
        tail_m1 = coords[:, 0, 3, :]
        tail_e1 = coords[:, 0, 4, :]
        
        # Individual 2
        top2 = coords[:, 1, 0, :]
        body2 = coords[:, 1, 1, :]
        tail2 = coords[:, 1, 2, :]
        tail_m2 = coords[:, 1, 3, :]
        tail_e2 = coords[:, 1, 4, :]
        
        features = {}
        
        # ============== 速度特征 (4个) ==============
        features['top1_speed'] = self.compute_speed(top1)
        features['top2_speed'] = self.compute_speed(top2)
        features['body1_speed'] = self.compute_speed(body1)
        features['body2_speed'] = self.compute_speed(body2)
        
        # ============== 距离特征 (4个) ==============
        features['top_distance'] = self.compute_distance(top1, top2)
        features['body_distance'] = self.compute_distance(body1, body2)
        features['top1_tail2_distance'] = self.compute_distance(top1, tail2)
        features['top2_tail1_distance'] = self.compute_distance(top2, tail1)
        
        # ============== 角度特征 (2个) ==============
        features['angle_top1_tail1'] = self.compute_angle(top1, tail_e1)
        features['angle_top2_tail2'] = self.compute_angle(top2, tail_e2)
        
        # ============== 原始坐标特征 (20个) ==============
        # Individual 1
        features['top1_x'] = top1[:, 0]
        features['top1_y'] = top1[:, 1]
        features['body1_x'] = body1[:, 0]
        features['body1_y'] = body1[:, 1]
        features['tail1_x'] = tail1[:, 0]
        features['tail1_y'] = tail1[:, 1]
        features['tail_mid1_x'] = tail_m1[:, 0]
        features['tail_mid1_y'] = tail_m1[:, 1]
        features['tail_end1_x'] = tail_e1[:, 0]
        features['tail_end1_y'] = tail_e1[:, 1]
        
        # Individual 2
        features['top2_x'] = top2[:, 0]
        features['top2_y'] = top2[:, 1]
        features['body2_x'] = body2[:, 0]
        features['body2_y'] = body2[:, 1]
        features['tail2_x'] = tail2[:, 0]
        features['tail2_y'] = tail2[:, 1]
        features['tail_mid2_x'] = tail_m2[:, 0]
        features['tail_mid2_y'] = tail_m2[:, 1]
        features['tail_end2_x'] = tail_e2[:, 0]
        features['tail_end2_y'] = tail_e2[:, 1]
        
        # ============== 交互特征 (4个) ==============
        features['relative_angle'] = features['angle_top1_tail1'] - features['angle_top2_tail2']
        features['speed_ratio'] = (features['top1_speed'] + 1) / (features['top2_speed'] + 1)
        features['approach_speed'] = -np.gradient(features['top_distance'])
        features['body_speed_diff'] = np.abs(features['body1_speed'] - features['body2_speed'])
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features)
        
        return features_df, n_frames
    
    def extract_from_folder(self, csv_folder, save_path=None):
        """
        从文件夹提取所有CSV文件的特征
        
        Args:
            csv_folder: CSV文件夹路径
            save_path: 保存合并后特征的路径 (可选)
            
        Returns:
            combined_features: 合并后的特征DataFrame
            n_frames_dict: {video_id: n_frames} 字典
        """
        csv_folder = Path(csv_folder)
        all_features = []
        n_frames_dict = {}
        
        # 查找所有DLC CSV文件
        csv_files = sorted(csv_folder.glob("*DLC*.csv"))
        
        if not csv_files:
            print(f"Warning: No DLC CSV files found in {csv_folder}")
            return pd.DataFrame(), {}
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            # 提取视频ID (例如 "6DLC_xxx.csv" -> 6)
            try:
                video_id = int(csv_file.stem.split('DLC')[0])
            except ValueError:
                print(f"  Warning: Cannot parse video ID from {csv_file.name}")
                continue
            
            try:
                features, n_frames = self.extract_features(csv_file)
                features['video_id'] = video_id
                features['frame'] = np.arange(n_frames)
                all_features.append(features)
                n_frames_dict[video_id] = n_frames
                print(f"  ✓ Video {video_id}: {n_frames} frames, {features.shape[1]} features")
            except Exception as e:
                print(f"  ✗ Video {video_id}: {e}")
                continue
        
        if not all_features:
            return pd.DataFrame(), {}
        
        # 合并所有特征
        combined = pd.concat(all_features, ignore_index=True)
        
        if save_path:
            combined.to_pickle(save_path)
            print(f"\nSaved features to {save_path}")
        
        return combined, n_frames_dict


def get_feature_names():
    """获取所有特征名称（不包括video_id和frame）"""
    return [
        # 速度特征
        'top1_speed', 'top2_speed', 'body1_speed', 'body2_speed',
        # 距离特征
        'top_distance', 'body_distance', 'top1_tail2_distance', 'top2_tail1_distance',
        # 角度特征
        'angle_top1_tail1', 'angle_top2_tail2',
        # 坐标特征 - Individual 1
        'top1_x', 'top1_y', 'body1_x', 'body1_y', 'tail1_x', 'tail1_y',
        'tail_mid1_x', 'tail_mid1_y', 'tail_end1_x', 'tail_end1_y',
        # 坐标特征 - Individual 2
        'top2_x', 'top2_y', 'body2_x', 'body2_y', 'tail2_x', 'tail2_y',
        'tail_mid2_x', 'tail_mid2_y', 'tail_end2_x', 'tail_end2_y',
        # 交互特征
        'relative_angle', 'speed_ratio', 'approach_speed', 'body_speed_diff'
    ]


# ============== 测试代码 ==============
if __name__ == "__main__":
    print("Feature Extraction Module")
    print("=" * 50)
    print(f"Total features: {len(get_feature_names())}")
    print("\nFeature categories:")
    print("  - Speed features: 4")
    print("  - Distance features: 4")
    print("  - Angle features: 2")
    print("  - Coordinate features: 20")
    print("  - Interaction features: 4")
    print("\n✓ Feature extraction module ready!")
