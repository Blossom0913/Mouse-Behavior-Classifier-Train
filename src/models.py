"""
Model Definitions for Mouse Behavior Classification
包含多种模型：MLP, LSTM, CNN, LightGBM, XGBoost, RandomForest, SVM, GMM, HMM
"""

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# PyTorch Models
# =============================================================================

class BehaviorMLP(nn.Module):
    """多层感知机模型"""
    
    def __init__(self, n_features, n_classes, hidden_dims=[256, 128, 64], dropout=0.3):
        """
        Args:
            n_features: 输入特征数
            n_classes: 类别数
            hidden_dims: 隐藏层维度列表
            dropout: Dropout比例
        """
        super().__init__()
        
        layers = []
        in_dim = n_features
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, n_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class BehaviorLSTM(nn.Module):
    """双向LSTM模型"""
    
    def __init__(self, n_features, n_classes, hidden_size=128, num_layers=2, dropout=0.3):
        """
        Args:
            n_features: 输入特征数
            n_classes: 类别数
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        # 取最后一层的正向和反向隐藏状态
        h_n = h_n[-2:].permute(1, 0, 2).reshape(x.size(0), -1)
        return self.fc(h_n)


class BehaviorCNN(nn.Module):
    """1D卷积神经网络模型"""
    
    def __init__(self, n_features, n_classes, dropout=0.5):
        """
        Args:
            n_features: 输入特征数
            n_classes: 类别数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * n_features, 128)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BehaviorTransformer(nn.Module):
    """Transformer模型"""
    
    def __init__(self, n_features, n_classes, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        Args:
            n_features: 输入特征数
            n_classes: 类别数
            d_model: Transformer维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.input_embed = nn.Linear(n_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.input_embed(x) + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.classifier(x)


# =============================================================================
# Sklearn Model Wrappers
# =============================================================================

def create_lightgbm_model(n_classes, use_gpu=False):
    """创建LightGBM模型参数"""
    params = {
        'objective': 'multiclass',
        'num_class': n_classes,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1
    }
    
    if use_gpu:
        params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        })
    
    return params


def create_xgboost_model(n_classes, use_gpu=False):
    """创建XGBoost模型参数"""
    params = {
        'objective': 'multi:softmax',
        'num_class': n_classes,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'verbosity': 0
    }
    
    if use_gpu:
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
    
    return params


def create_random_forest_model(n_classes, random_state=42):
    """创建RandomForest模型"""
    from sklearn.ensemble import RandomForestClassifier
    
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )


def create_svm_model(random_state=42):
    """创建SVM模型"""
    from sklearn.svm import SVC
    
    return SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=random_state,
        probability=True
    )


def create_gmm_model(n_components=10, random_state=42):
    """创建GMM模型"""
    from sklearn.mixture import GaussianMixture
    
    return GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=10,
        covariance_type='full'
    )


# =============================================================================
# Model Factory
# =============================================================================

def get_pytorch_model(model_name, n_features, n_classes, **kwargs):
    """
    获取PyTorch模型实例
    
    Args:
        model_name: 模型名称 ('mlp', 'lstm', 'cnn', 'transformer')
        n_features: 输入特征数
        n_classes: 类别数
        **kwargs: 额外参数
        
    Returns:
        model: PyTorch模型实例
    """
    model_name = model_name.lower()
    
    if model_name == 'mlp':
        return BehaviorMLP(n_features, n_classes, **kwargs)
    elif model_name == 'lstm':
        return BehaviorLSTM(n_features, n_classes, **kwargs)
    elif model_name == 'cnn':
        return BehaviorCNN(n_features, n_classes, **kwargs)
    elif model_name == 'transformer':
        return BehaviorTransformer(n_features, n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_sklearn_model(model_name, n_classes=None, random_state=42, **kwargs):
    """
    获取sklearn模型实例
    
    Args:
        model_name: 模型名称 ('rf', 'svm', 'gmm')
        n_classes: 类别数 (某些模型需要)
        random_state: 随机种子
        **kwargs: 额外参数
        
    Returns:
        model: sklearn模型实例
    """
    model_name = model_name.lower()
    
    if model_name in ['rf', 'randomforest', 'random_forest']:
        return create_random_forest_model(n_classes, random_state)
    elif model_name == 'svm':
        return create_svm_model(random_state)
    elif model_name == 'gmm':
        n_components = kwargs.get('n_components', 10)
        return create_gmm_model(n_components, random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Training Utilities
# =============================================================================

def compute_class_weights(y_train, n_classes):
    """计算类别权重"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.arange(n_classes)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return torch.FloatTensor(weights)


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_checkpoint(path, model, optimizer=None, epoch=None, seed=None, model_type=None, extra: dict | None = None):
    """保存训练检查点。
    包含模型参数、可选的优化器状态以及元信息（epoch/seed/model_type）。

    Args:
        path: 保存路径（.pth 文件）
        model: PyTorch 模型
        optimizer: 可选，优化器实例
        epoch: 可选，当前训练到的 epoch（int）
        seed: 可选，训练使用的随机种子
        model_type: 可选，模型类型标识（'cnn'、'lstm' 等）
        extra: 可选，附加字典信息（例如数据分割配置等）
    """
    state = {
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        state['epoch'] = int(epoch)
    if seed is not None:
        state['seed'] = int(seed)
    if model_type is not None:
        state['model_type'] = str(model_type)
    if extra:
        state['extra'] = extra

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    """加载训练检查点。

    Args:
        path: 检查点路径
        model: PyTorch 模型，用于加载 state_dict
        optimizer: 可选，优化器；若提供且检查点包含优化器状态，则一并恢复
        map_location: 设备映射

    Returns:
        info: dict，包含 'epoch'（若存在）、'seed'、'model_type' 等元信息
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt.get('state_dict', ckpt))

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass  # 若不匹配则忽略优化器恢复

    return {
        'epoch': ckpt.get('epoch'),
        'seed': ckpt.get('seed'),
        'model_type': ckpt.get('model_type'),
        'extra': ckpt.get('extra')
    }


def find_checkpoint(checkpoints_dir, model_type, seed):
    """按约定命名查找检查点文件：{model_type}_seed{seed}.pth"""
    import os
    filename = f"{model_type}_seed{seed}.pth"
    path = os.path.join(checkpoints_dir, filename)
    return path if os.path.isfile(path) else None


# ============== 测试代码 ==============
if __name__ == "__main__":
    print("Model Definitions Module")
    print("=" * 50)
    
    # 测试模型创建
    n_features = 34
    n_classes = 7
    batch_size = 32
    
    print("\nTesting PyTorch models:")
    
    for name in ['mlp', 'lstm', 'cnn', 'transformer']:
        model = get_pytorch_model(name, n_features, n_classes)
        x = torch.randn(batch_size, n_features)
        out = model(x)
        print(f"  {name.upper()}: input={x.shape} -> output={out.shape}")
    
    print("\nTesting sklearn models:")
    for name in ['rf', 'svm', 'gmm']:
        try:
            model = get_sklearn_model(name, n_classes)
            print(f"  {name.upper()}: {type(model).__name__}")
        except ImportError as e:
            print(f"  {name.upper()}: Not available ({e})")
    
    print("\n✓ All models tested successfully!")
