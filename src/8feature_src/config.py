"""
Hyperparameter Configuration for Mouse Behavior Classification
===============================================================

This file contains all hyperparameter settings for the 8-model comparison experiment.
Modify these values to tune model performance.
"""

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

EXPERIMENT_CONFIG = {
    # Experiment mode: "behavior" (3-class) or "aggression" (6-class)
    "experiment_mode": "aggression",
    
    # Number of runs for error bar calculation
    "n_runs": 5,
    
    # Base random seed (each run uses seed_base + run_id)
    "random_seed_base": 42,
}

# =============================================================================
# DATA SPLIT SETTINGS
# =============================================================================

DATA_SPLIT = {
    "test_size": 0.3,        # 30% for validation + test
    "val_test_split": 0.5,   # Split temp data 50/50 for val and test
    # Final split: 70% train, 15% validation, 15% test
}

# =============================================================================
# DEEP LEARNING COMMON PARAMETERS
# =============================================================================

DEEP_LEARNING = {
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "optimizer": "Adam",
}

# =============================================================================
# LSTM PARAMETERS
# =============================================================================

LSTM_PARAMS = {
    "hidden_size": 64,
    "num_layers": 1,
    "bidirectional": True,
    "gradient_clip": 1.0,
    # Inherits: batch_size, epochs, learning_rate, weight_decay, dropout from DEEP_LEARNING
}

# =============================================================================
# CNN PARAMETERS
# =============================================================================

CNN_PARAMS = {
    "conv1_channels": 32,
    "conv2_channels": 64,
    "kernel_size": 3,
    "padding": 1,
    "fc_hidden": 128,
    # Inherits: batch_size, epochs, learning_rate, weight_decay, dropout from DEEP_LEARNING
}

# =============================================================================
# GMM PARAMETERS
# =============================================================================

GMM_PARAMS = {
    "n_components": 10,
    "n_init": 10,
    "covariance_type": "full",  # Options: 'full', 'tied', 'diag', 'spherical'
}

# =============================================================================
# LIGHTGBM PARAMETERS
# =============================================================================

LIGHTGBM_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "num_boost_round": 100,
    "verbose": -1,
    # GPU settings (optional)
    "use_gpu": False,
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}

# =============================================================================
# XGBOOST PARAMETERS
# =============================================================================

XGBOOST_PARAMS = {
    "objective": "multi:softmax",
    "max_depth": 6,
    "eta": 0.1,  # learning_rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "num_boost_round": 200,
    "early_stopping_rounds": 20,
    "verbosity": 0,
    # GPU settings (optional)
    "use_gpu": False,
    "tree_method": "hist",  # Use "gpu_hist" for GPU
}

# =============================================================================
# RANDOM FOREST PARAMETERS
# =============================================================================

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,  # Use all CPU cores
    "verbose": 0,
}

# =============================================================================
# SVM PARAMETERS
# =============================================================================

SVM_PARAMS = {
    "kernel": "rbf",  # Options: 'linear', 'poly', 'rbf', 'sigmoid'
    "C": 1.0,
    "gamma": "scale",  # Options: 'scale', 'auto', or float
    "class_weight": "balanced",
    "probability": True,
    "max_iter": -1,  # No limit
}

# =============================================================================
# HMM PARAMETERS
# =============================================================================

HMM_PARAMS = {
    "n_components": 3,  # Number of hidden states
    "covariance_type": "diag",  # Options: 'spherical', 'diag', 'full', 'tied'
    "n_iter": 100,  # Maximum EM iterations
    "tol": 1e-2,  # Convergence threshold
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_CONFIG = {
    "output_dir": "../output",
    "save_models": False,
    "save_predictions": False,
    "figure_dpi": 300,
    "figure_format": "png",
}

# =============================================================================
# HELPER FUNCTION TO GET ALL PARAMS
# =============================================================================

def get_all_params():
    """Return all parameters as a dictionary"""
    return {
        "experiment": EXPERIMENT_CONFIG,
        "data_split": DATA_SPLIT,
        "deep_learning": DEEP_LEARNING,
        "lstm": LSTM_PARAMS,
        "cnn": CNN_PARAMS,
        "gmm": GMM_PARAMS,
        "lightgbm": LIGHTGBM_PARAMS,
        "xgboost": XGBOOST_PARAMS,
        "random_forest": RANDOM_FOREST_PARAMS,
        "svm": SVM_PARAMS,
        "hmm": HMM_PARAMS,
        "output": OUTPUT_CONFIG,
    }


def print_all_params():
    """Print all parameters in a formatted way"""
    import json
    params = get_all_params()
    print("=" * 60)
    print("HYPERPARAMETER CONFIGURATION")
    print("=" * 60)
    for category, values in params.items():
        print(f"\n{category.upper()}:")
        print("-" * 40)
        for key, value in values.items():
            print(f"  {key}: {value}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_all_params()
