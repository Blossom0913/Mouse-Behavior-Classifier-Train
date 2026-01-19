import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from data_solver import *

def data_load(coords_path, labels_path):
    """正确实现数据加载函数"""
    try:
        # 读取坐标数据（假设是特征数据）
        coords_df = pd.read_excel(coords_path, engine='openpyxl')
        
        # 读取标签数据（需要确保索引对齐）
        labels_df = pd.read_excel(labels_path, engine='openpyxl')
        
        # 对齐数据索引（重要！）
        assert len(coords_df) == len(labels_df), "特征数据和标签数据长度不一致"
        return coords_df, labels_df
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        return None, None

# 修正后的数据预处理流程
def prepare_data(coords_path, labels_path, seq_length=64):
    # 加载原始数据
    coords_df, labels_df = data_load(coords_path, labels_path)
    
    # 数据校验
    if coords_df is None or labels_df is None:
        raise ValueError("数据加载失败，请检查文件路径和格式")
    
    # 特征处理
    X = coords_df.values.astype(np.float32)
    
    # 标签处理（使用LabelEncoder替代OneHotEncoder）
    le = LabelEncoder()
    y = le.fit_transform(labels_df['label'].values)
    
    # 打印类别分布
    unique, counts = np.unique(y, return_counts=True)
    print("原始类别分布：")
    for cls, cnt in zip(unique, counts):
        print(f"类别 {cls}: {cnt} 样本 ({cnt/len(y):.2%})")
    
    # 数据标准化（注意时序数据的正确标准化方式）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 全局标准化
    
    # 创建时序序列（使用之前设计的增强版函数）
    X_seq, y_seq = create_enhanced_sequences(
        X_scaled, y,
        seq_length=seq_length,
        window_sizes=[8, 16],
        diff_orders=[1, 2],
        enable_aug=False  # 验证时不启用增强
    )
    
    # 最终数据检查
    print("\n处理后的数据形状：")
    print(f"特征序列: {X_seq.shape} (样本数, 序列长度, 特征维度)")
    print(f"标签形状: {y_seq.shape}")
    
    return X_seq, y_seq, le.classes_

# 使用示例
if __name__ == "__main__":
    # 文件路径
    coords_path = "data/train_batch5/merged_feature.xlsx"
    labels_path = "data/train_batch5/merged_label.xlsx"
    
    # 执行预处理
    X_sequences, y_labels, class_names = prepare_data(
        coords_path, 
        labels_path,
        seq_length=64
    )
    
    # 数据集拆分（保持时序顺序）
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, 
        y_labels,
        test_size=0.2,
        stratify=y_labels,  # 保持类别分布
        random_state=42
    )