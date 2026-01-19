"""
Mouse Behavior Classification - 8-Model Comparison with Error Bars
===================================================================

This script runs comprehensive model comparison experiments for mouse behavior
classification using 8 different machine learning models.

Supported Models:
    1. GMM (Gaussian Mixture Model) - Probabilistic baseline
    2. LSTM (Long Short-Term Memory) - Deep learning, temporal
    3. CNN (Convolutional Neural Network) - Deep learning, spatial
    4. LightGBM - Gradient boosting, efficient
    5. XGBoost - Gradient boosting, regularized
    6. Random Forest - Ensemble method
    7. SVM (Support Vector Machine) - Kernel methods
    8. HMM (Hidden Markov Model) - Probabilistic, sequential

Experiment Modes:
    - "behavior": Multi-class behavior classification (3 classes)
    - "aggression": 6-class aggression classification (classes 1-6)

Both modes exclude class 0 from training and evaluation.

Author: DeepLabVideo Team
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Fix for Windows loky/joblib subprocess error
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Try to import hmmlearn
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠ hmmlearn not available - HMM experiments will be skipped")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Experiment configuration and hyperparameters"""
    
    # Experiment settings
    EXPERIMENT_MODE = "aggression"  # "behavior" or "aggression"
    N_RUNS = 5  # Number of runs for error bar calculation
    RANDOM_SEED_BASE = 42
    
    # Data paths (relative to code/ directory)
    FEATURE_FILE = "../data/dataset58/feature8_58.xlsx"
    LABEL_FILES = {
        "behavior": "../data/dataset58/merged_labels.xlsx",
        "aggression": "../data/dataset58/merged_labels_aggression.xlsx"
    }
    
    # Data split ratios
    TEST_SIZE = 0.3  # 30% for temp (val + test)
    VAL_TEST_SPLIT = 0.5  # 50% of temp for val, 50% for test
    
    # Deep Learning Hyperparameters
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.5
    
    # LSTM specific
    LSTM_HIDDEN_SIZE = 64
    LSTM_NUM_LAYERS = 1
    LSTM_BIDIRECTIONAL = True
    
    # CNN specific
    CNN_CONV1_CHANNELS = 32
    CNN_CONV2_CHANNELS = 64
    CNN_FC_HIDDEN = 128
    CNN_KERNEL_SIZE = 3
    
    # GMM specific
    GMM_N_COMPONENTS = 10
    GMM_N_INIT = 10
    GMM_COVARIANCE_TYPE = 'full'
    
    # LightGBM specific
    LGBM_NUM_LEAVES = 31
    LGBM_LEARNING_RATE = 0.1
    LGBM_NUM_BOOST_ROUND = 100
    
    # XGBoost specific
    XGB_MAX_DEPTH = 6
    XGB_ETA = 0.1
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE_BYTREE = 0.8
    XGB_NUM_BOOST_ROUND = 200
    XGB_EARLY_STOPPING = 20
    
    # Random Forest specific
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_SPLIT = 5
    
    # SVM specific
    SVM_KERNEL = 'rbf'
    SVM_C = 1.0
    SVM_GAMMA = 'scale'
    
    # HMM specific
    HMM_N_COMPONENTS = 3
    HMM_N_ITER = 100
    HMM_COVARIANCE_TYPE = 'diag'
    
    # Output
    OUTPUT_DIR = "../output"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features(feature_path):
    """Load feature data from Excel file"""
    print("Loading feature data...")
    try:
        X_raw = pd.read_excel(feature_path, index_col=0).values
        print(f"✓ Features loaded: shape={X_raw.shape}")
        return X_raw
    except FileNotFoundError:
        print(f"⚠ Feature file not found: {feature_path}")
        raise


def load_labels(label_path):
    """Load label data from Excel file"""
    print(f"Loading labels from: {label_path}")
    try:
        y_raw = pd.read_excel(label_path, index_col=0).iloc[:, 0].values
        print(f"✓ Labels loaded: shape={y_raw.shape}")
        return y_raw
    except FileNotFoundError:
        print(f"⚠ Label file not found: {label_path}")
        raise


def filter_data(X_raw, y_raw, experiment_mode):
    """
    Filter data based on experiment mode.
    Removes class 0 and selects appropriate classes.
    """
    print(f"\nFiltering data for {experiment_mode} experiment...")
    
    # Analyze class distribution
    unique_classes = np.unique(y_raw.astype(int))
    class_counts = np.bincount(y_raw.astype(int))
    
    # Build list of available classes with sample counts
    available_classes = []
    for cls in unique_classes:
        if cls < len(class_counts) and class_counts[cls] > 0:
            available_classes.append((cls, class_counts[cls]))
    available_classes.sort(key=lambda x: x[1], reverse=True)
    
    print("Available classes (sorted by sample count):")
    for cls, count in available_classes:
        print(f"  Class {cls}: {count:,} samples")
    
    # Determine number of classes based on experiment mode
    if experiment_mode == "behavior":
        n_classes = 3
        print(f"\n📊 BEHAVIOR EXPERIMENT: 3-class classification")
    else:  # aggression
        n_classes = 6
        print(f"\n📊 AGGRESSION EXPERIMENT: 6-class classification (classes 1-6)")
    
    # Auto-select classes (excluding class 0)
    selected_classes = []
    for cls, count in available_classes:
        if cls != 0 and len(selected_classes) < n_classes:
            selected_classes.append(cls)
    
    print(f"Selected classes: {selected_classes}")
    
    # Filter data
    mask = np.zeros(len(y_raw), dtype=bool)
    for cls in selected_classes:
        mask |= (y_raw == cls)
    
    X_filtered = X_raw[mask]
    y_filtered = y_raw[mask]
    
    # Remap to contiguous range [0, n_classes-1]
    class_mapping = {old_cls: new_cls for new_cls, old_cls in enumerate(selected_classes)}
    y_mapped = np.array([class_mapping[label] for label in y_filtered], dtype=np.int32)
    
    print(f"\nFiltered data:")
    print(f"  X shape: {X_filtered.shape}")
    print(f"  y shape: {y_mapped.shape}")
    print(f"  Class mapping: {class_mapping}")
    print(f"  New distribution: {np.bincount(y_mapped)}")
    
    return X_filtered, y_mapped, n_classes, class_mapping


# =============================================================================
# PYTORCH MODELS
# =============================================================================

class MouseDataset(Dataset):
    """PyTorch Dataset for mouse behavior data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BehaviorLSTM(nn.Module):
    """Bidirectional LSTM for behavior classification"""
    def __init__(self, input_size, num_classes, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add time dimension
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.permute(1, 0, 2).contiguous().view(h_n.size(1), -1)
        x = self.dropout(h_n)
        x = self.fc(x)
        return x


class BehaviorCNN(nn.Module):
    """1D CNN for behavior classification"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(y_true, y_pred, num_classes):
    """Compute comprehensive classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
    
    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'class_f1': f1_per_class,
        'best_class_f1': np.max(f1_per_class) if len(f1_per_class) > 0 else 0,
        'worst_class_f1': np.min(f1_per_class) if len(f1_per_class) > 0 else 0,
        'y_true': y_true,
        'y_pred': y_pred
    }


# =============================================================================
# MODEL EXPERIMENT FUNCTIONS
# =============================================================================

def run_gmm_experiment(X_raw, y_raw, num_classes, split_seed=42, **kwargs):
    """Run Gaussian Mixture Model experiment"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    gmm = GaussianMixture(
        n_components=Config.GMM_N_COMPONENTS,
        random_state=split_seed,
        n_init=Config.GMM_N_INIT
    )
    gmm.fit(X_train)
    y_pred = gmm.predict(X_test) % num_classes
    
    return {'test_metrics': compute_metrics(y_test, y_pred, num_classes)}


def run_lstm_experiment(X_raw, y_raw, num_classes, split_seed=42, device='cpu', **kwargs):
    """Run LSTM experiment"""
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)
    
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    train_loader = DataLoader(MouseDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MouseDataset(X_test, y_test), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = BehaviorLSTM(input_size=X_train.shape[1], num_classes=num_classes).to(device)
    
    # Class weights
    class_weights = np.ones(num_classes)
    classes_present = np.unique(y_train)
    if len(classes_present) == num_classes:
        computed_weights = compute_class_weight('balanced', classes=classes_present, y=y_train)
        class_weights[classes_present] = computed_weights
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Training
    for epoch in range(Config.EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Test
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
    
    return {'test_metrics': compute_metrics(y_test, np.array(y_pred), num_classes)}


def run_cnn_experiment(X_raw, y_raw, num_classes, split_seed=42, device='cpu', **kwargs):
    """Run CNN experiment"""
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)
    
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    train_loader = DataLoader(MouseDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(MouseDataset(X_test, y_test), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = BehaviorCNN(input_size=X_train.shape[1], num_classes=num_classes).to(device)
    
    # Class weights
    class_weights = np.ones(num_classes)
    classes_present = np.unique(y_train)
    if len(classes_present) == num_classes:
        computed_weights = compute_class_weight('balanced', classes=classes_present, y=y_train)
        class_weights[classes_present] = computed_weights
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Training
    for epoch in range(Config.EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Test
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
    
    return {'test_metrics': compute_metrics(y_test, np.array(y_pred), num_classes)}


def run_lightgbm_experiment(X_raw, y_raw, num_classes, split_seed=42, use_gpu=False, **kwargs):
    """Run LightGBM experiment"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'num_leaves': Config.LGBM_NUM_LEAVES,
        'learning_rate': Config.LGBM_LEARNING_RATE,
        'verbose': -1
    }
    
    if use_gpu:
        params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    try:
        model = lgb.train(params, train_data, num_boost_round=Config.LGBM_NUM_BOOST_ROUND)
    except Exception as e:
        if use_gpu:
            params['device'] = 'cpu'
            model = lgb.train(params, train_data, num_boost_round=Config.LGBM_NUM_BOOST_ROUND)
        else:
            raise
    
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return {'test_metrics': compute_metrics(y_test, y_pred, num_classes)}


def run_xgboost_experiment(X_raw, y_raw, num_classes, split_seed=42, use_gpu=False, **kwargs):
    """Run XGBoost experiment"""
    y_raw_copy = y_raw.copy()
    unique_classes = np.unique(y_raw_copy)
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    y_raw_copy = np.array([class_mapping[label] for label in y_raw_copy], dtype=np.int32)
    actual_num_classes = len(unique_classes)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw_copy, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw_copy
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    params = {
        'objective': 'multi:softmax',
        'num_class': actual_num_classes,
        'max_depth': Config.XGB_MAX_DEPTH,
        'eta': Config.XGB_ETA,
        'subsample': Config.XGB_SUBSAMPLE,
        'colsample_bytree': Config.XGB_COLSAMPLE_BYTREE,
        'eval_metric': 'mlogloss',
        'seed': split_seed,
        'verbosity': 0
    }
    
    if use_gpu:
        params['tree_method'] = 'gpu_hist'
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=Config.XGB_NUM_BOOST_ROUND,
                      evals=evals, early_stopping_rounds=Config.XGB_EARLY_STOPPING, verbose_eval=False)
    
    y_pred = model.predict(dtest).astype(np.int32)
    
    return {'test_metrics': compute_metrics(y_test, y_pred, actual_num_classes)}


def run_random_forest_experiment(X_raw, y_raw, num_classes, split_seed=42, **kwargs):
    """Run Random Forest experiment"""
    y_raw_copy = y_raw.copy()
    unique_classes = np.unique(y_raw_copy)
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    y_raw_copy = np.array([class_mapping[label] for label in y_raw_copy], dtype=np.int32)
    actual_num_classes = len(unique_classes)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw_copy, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw_copy
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
        class_weight='balanced',
        random_state=split_seed,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {'test_metrics': compute_metrics(y_test, y_pred, actual_num_classes)}


def run_svm_experiment(X_raw, y_raw, num_classes, split_seed=42, **kwargs):
    """Run SVM experiment"""
    y_raw_copy = y_raw.copy()
    unique_classes = np.unique(y_raw_copy)
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    y_raw_copy = np.array([class_mapping[label] for label in y_raw_copy], dtype=np.int32)
    actual_num_classes = len(unique_classes)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw_copy, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw_copy
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = SVC(
        kernel=Config.SVM_KERNEL,
        C=Config.SVM_C,
        gamma=Config.SVM_GAMMA,
        class_weight='balanced',
        random_state=split_seed,
        probability=True
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {'test_metrics': compute_metrics(y_test, y_pred, actual_num_classes)}


def run_hmm_experiment(X_raw, y_raw, num_classes, split_seed=42, **kwargs):
    """Run HMM experiment - one Gaussian HMM per class"""
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn not available")
    
    y_raw_copy = y_raw.copy()
    unique_classes = np.unique(y_raw_copy)
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    y_raw_copy = np.array([class_mapping[label] for label in y_raw_copy], dtype=np.int32)
    actual_num_classes = len(unique_classes)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_raw_copy, test_size=Config.TEST_SIZE, random_state=split_seed, stratify=y_raw_copy
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_TEST_SPLIT, random_state=split_seed, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train one HMM per class
    models = {}
    for cls in range(actual_num_classes):
        cls_data = X_train[y_train == cls]
        if len(cls_data) == 0:
            continue
        
        n_components = min(Config.HMM_N_COMPONENTS, max(1, len(cls_data) // 2))
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=Config.HMM_COVARIANCE_TYPE,
            n_iter=Config.HMM_N_ITER,
            random_state=split_seed,
            verbose=False
        )
        try:
            model.fit(cls_data)
            models[cls] = model
        except:
            continue
    
    # Predict by log-likelihood
    y_pred = []
    for x in X_test:
        x_reshaped = x.reshape(1, -1)
        log_likelihoods = []
        for cls in range(actual_num_classes):
            if cls in models:
                try:
                    ll = models[cls].score(x_reshaped)
                    log_likelihoods.append(ll)
                except:
                    log_likelihoods.append(float('-inf'))
            else:
                log_likelihoods.append(float('-inf'))
        y_pred.append(np.argmax(log_likelihoods))
    
    return {'test_metrics': compute_metrics(y_test, np.array(y_pred), actual_num_classes)}


# =============================================================================
# MULTIPLE RUNS FOR ERROR BARS
# =============================================================================

def run_multiple_experiments(model_name, experiment_func, X_raw, y_raw, num_classes,
                            n_runs=5, device='cpu', use_gpu=False):
    """Run experiment multiple times with different random seeds for error bars"""
    print(f"\n{'='*60}")
    print(f"Running {model_name} ({n_runs} runs)")
    print(f"{'='*60}")
    
    all_metrics = {
        'accuracy': [], 'weighted_f1': [], 'macro_f1': [],
        'best_class_f1': [], 'worst_class_f1': [], 'class_f1': []
    }
    
    successful_runs = 0
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}", end="")
        
        np.random.seed(Config.RANDOM_SEED_BASE + run)
        torch.manual_seed(Config.RANDOM_SEED_BASE + run)
        split_seed = Config.RANDOM_SEED_BASE + run
        
        try:
            result = experiment_func(
                X_raw, y_raw, num_classes=num_classes,
                split_seed=split_seed, device=device, use_gpu=use_gpu
            )
            
            metrics = result['test_metrics']
            for key in all_metrics:
                if key == 'class_f1':
                    all_metrics[key].append(metrics[key])
                else:
                    all_metrics[key].append(metrics[key])
            
            successful_runs += 1
            print(f" ✓ (acc={metrics['accuracy']:.3f}, f1={metrics['weighted_f1']:.3f})")
        except Exception as e:
            print(f" ✗ Error: {str(e)[:40]}")
            continue
    
    print(f"\nCompleted {successful_runs}/{n_runs} runs for {model_name}")
    
    # Calculate statistics
    stats = {}
    for key in all_metrics:
        if all_metrics[key] and len(all_metrics[key]) > 1:
            values = np.array(all_metrics[key])
            if key == 'class_f1':
                stats[key] = {'mean': np.mean(values, axis=0), 'std': np.std(values, axis=0, ddof=1)}
            else:
                stats[key] = {'mean': np.mean(values), 'std': np.std(values, ddof=1)}
        elif all_metrics[key]:
            values = np.array(all_metrics[key])
            if key == 'class_f1':
                stats[key] = {'mean': np.mean(values, axis=0), 'std': np.zeros_like(np.mean(values, axis=0))}
            else:
                stats[key] = {'mean': np.mean(values), 'std': 0.0}
        else:
            if key == 'class_f1':
                stats[key] = {'mean': np.zeros(num_classes), 'std': np.zeros(num_classes)}
            else:
                stats[key] = {'mean': 0, 'std': 0}
    
    return stats


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_graphs(all_results, output_dir, num_classes):
    """Create comparison graphs with error bars"""
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(all_results.keys())
    metrics = ['accuracy', 'weighted_f1', 'macro_f1']
    
    plt.style.use('seaborn-v0_8')
    
    # Overall Performance
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25
    for i, metric in enumerate(metrics):
        means = [all_results[m][metric]['mean'] for m in models]
        stds = [all_results[m][metric]['std'] for m in models]
        plt.bar(x + i*width, means, width, label=metric.replace('_', ' ').title(),
                yerr=stds, capsize=5, alpha=0.85)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'{num_classes}-Class Performance Comparison ({Config.N_RUNS} runs, mean ± std)', fontsize=14)
    plt.xticks(x + width, models, rotation=15, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_overall.png'), dpi=300)
    plt.close()
    
    # Per-Class F1
    plt.figure(figsize=(14, 6))
    classes = [f'Class {i}' for i in range(num_classes)]
    for i, model in enumerate(models):
        means = all_results[model]['class_f1']['mean']
        stds = all_results[model]['class_f1']['std']
        x_pos = np.arange(len(classes)) + i * (0.8 / len(models))
        plt.bar(x_pos, means, 0.8 / len(models), label=model, yerr=stds, capsize=3, alpha=0.85)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'Per-Class F1 Scores ({Config.N_RUNS} runs, mean ± std)', fontsize=14)
    plt.xticks(np.arange(len(classes)) + 0.35, classes)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_per_class.png'), dpi=300)
    plt.close()
    
    print(f"✓ Graphs saved to {output_dir}")


def print_statistics_table(all_results, num_classes):
    """Print detailed statistics table"""
    print("\n" + "="*100)
    print(f"DETAILED STATISTICS TABLE ({Config.N_RUNS} runs, mean ± std)")
    print("="*100)
    
    models = list(all_results.keys())
    
    header = f"{'Model':<12} {'Accuracy':<16} {'Weighted F1':<16} {'Macro F1':<16} {'Best F1':<16} {'Worst F1':<16}"
    print(header)
    print("-" * len(header))
    
    for model in models:
        row = f"{model:<12}"
        for metric in ['accuracy', 'weighted_f1', 'macro_f1', 'best_class_f1', 'worst_class_f1']:
            mean = all_results[model][metric]['mean']
            std = all_results[model][metric]['std']
            row += f"{mean:.4f}±{std:.4f}".ljust(16)
        print(row)
    
    print("="*100)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("MOUSE BEHAVIOR CLASSIFICATION - 8-MODEL COMPARISON WITH ERROR BARS")
    print("="*80)
    
    # Load data
    X_raw = load_features(Config.FEATURE_FILE)
    y_raw = load_labels(Config.LABEL_FILES[Config.EXPERIMENT_MODE])
    
    # Ensure same length
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]
    
    # Filter data
    X_filtered, y_filtered, num_classes, class_mapping = filter_data(
        X_raw, y_raw, Config.EXPERIMENT_MODE
    )
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Number of runs: {Config.N_RUNS}")
    
    # Define experiments
    experiments = {
        'GMM': run_gmm_experiment,
        'LSTM': run_lstm_experiment,
        'CNN': run_cnn_experiment,
        'LightGBM': run_lightgbm_experiment,
        'XGBoost': run_xgboost_experiment,
        'RandomForest': run_random_forest_experiment,
        'SVM': run_svm_experiment,
    }
    
    if HMM_AVAILABLE:
        experiments['HMM'] = run_hmm_experiment
    
    print(f"\nModels to run ({len(experiments)}): {list(experiments.keys())}")
    
    # Run experiments
    all_results = {}
    for model_name, experiment_func in experiments.items():
        try:
            results = run_multiple_experiments(
                model_name, experiment_func, X_filtered, y_filtered,
                num_classes, Config.N_RUNS, device
            )
            all_results[model_name] = results
            print(f"✓ {model_name} completed")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
    
    # Generate outputs
    if all_results:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        create_comparison_graphs(all_results, Config.OUTPUT_DIR, num_classes)
        print_statistics_table(all_results, num_classes)
        
        print(f"\n{'='*80}")
        print(f"✓ Completed {len(all_results)} model comparisons")
        print(f"✓ Output saved to {Config.OUTPUT_DIR}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
