"""
XGBoost Classifier for Mouse Behavior Classification
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import xgboost as xgb

from data_load import load_mouse_data


def save_model(model, scaler, experiment_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_{experiment_type}_{timestamp}.pkl"
    joblib.dump({'model': model, 'scaler': scaler}, model_name)
    print(f"Model saved as {model_name}")
    return model_name


def compute_metrics(y_true, y_pred, num_classes):
    """Compute comprehensive metrics"""
    acc = 100 * accuracy_score(y_true, y_pred)
    weighted_f1 = 100 * f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = 100 * f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate per-class F1 scores
    class_f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0) * 100
    best_class_f1 = np.max(class_f1) if len(class_f1) > 0 else 0
    worst_class_f1 = np.min(class_f1) if len(class_f1) > 0 else 0
    
    return {
        'accuracy': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'best_class_f1': best_class_f1,
        'worst_class_f1': worst_class_f1,
        'class_f1': class_f1
    }


def run_experiment(X_raw, y_raw, num_classes=4, include_base=True, 
                   experiment_name="xgboost", split_seed=None, use_gpu=False):
    """
    Run XGBoost experiment with stratified splitting
    
    Args:
        X_raw: Raw feature matrix
        y_raw: Raw labels
        num_classes: Number of classes
        include_base: Whether to include base class
        experiment_name: Name for saved model
        split_seed: Random seed for data splitting (different seeds = different splits)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dictionary containing test metrics and model path
    """
    # Ensure X_raw and y_raw have the same length
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        print(f"Warning: X_raw and y_raw have different lengths ({len(X_raw)} vs {len(y_raw)}). Trimming to {min_len}.")
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]
    
    # Get unique classes in the data
    unique_classes = np.unique(y_raw)
    print(f"Unique classes in data: {unique_classes}")
    
    X_raw_exp, y_raw_exp = X_raw, y_raw.copy()
    
    # Map non-contiguous classes to contiguous range [0, num_classes-1]
    # This is CRITICAL for XGBoost which requires labels in [0, num_class)
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    reverse_mapping = {idx: cls for cls, idx in class_mapping.items()}
    print(f"Class mapping: {class_mapping}")
    
    # Apply mapping to labels and ensure np.int32 type
    y_raw_exp = np.array([class_mapping[label] for label in y_raw_exp], dtype=np.int32)
    
    # Update num_classes to match actual number of unique classes
    actual_num_classes = len(unique_classes)
    print(f"Updated num_classes from {num_classes} to {actual_num_classes}")
    num_classes = actual_num_classes
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw_exp)
    
    # Split data using stratified splitting (70% train, 15% val, 15% test)
    if split_seed is None:
        split_seed = 42
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_raw_exp, test_size=0.3, random_state=split_seed, stratify=y_raw_exp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=split_seed, stratify=y_temp)
    
    # Ensure labels are proper type for XGBoost
    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    # Configure XGBoost parameters
    params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': split_seed,  # Use split_seed for model randomness
        'verbosity': 0
    }
    
    # Try GPU if available and requested
    if use_gpu:
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Training with early stopping
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    try:
        model = xgb.train(
            params, dtrain, 
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False
        )
    except Exception as e:
        if use_gpu:
            # If GPU training fails, fallback to CPU
            print(f"    ⚠ GPU training failed, using CPU: {str(e)[:50]}")
            params.pop('tree_method', None)
            params.pop('gpu_id', None)
            model = xgb.train(
                params, dtrain, 
                num_boost_round=200,
                evals=evals,
                early_stopping_rounds=20,
                verbose_eval=False
            )
        else:
            raise
    
    # Save model
    model_path = save_model(model, scaler, f"{experiment_name}_{num_classes}class")
    
    # Validation evaluation
    val_preds = model.predict(dval).astype(np.int32)
    val_accuracy = 100 * accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Test evaluation
    test_preds = model.predict(dtest).astype(np.int32)
    
    # Calculate comprehensive metrics
    test_metrics = compute_metrics(
        y_true=y_test, 
        y_pred=test_preds, 
        num_classes=num_classes
    )
    
    # Print benchmark results
    print("\n=== FINAL BENCHMARK RESULTS ===")
    print(f"Accuracy (%): {test_metrics['accuracy']:.2f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.2f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.2f}")
    print(f"Best Class F1: {test_metrics['best_class_f1']:.2f}")
    print(f"Worst Class F1: {test_metrics['worst_class_f1']:.2f}")
    print("Per-class F1 Scores:")
    for i, f1 in enumerate(test_metrics['class_f1']):
        print(f"  Class {i}: {f1:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report (test):")
    print(classification_report(y_test, test_preds, digits=4))
    
    return {
        'test_metrics': test_metrics,
        'model_path': model_path
    }


def main():
    # Load data
    X_raw, y_raw = load_mouse_data(
        "../data/dataset58/feature8_58.xlsx",
        "../data/dataset58/merged_labels.xlsx"
    )
    
    # Run experiments
    results = {}
    for include_base, num_classes in [(True, 4), (False, 3)]:
        exp_type = f"{num_classes}class"
        print(f"\n=== STARTING XGBoost {exp_type.upper()} EXPERIMENT ===")
        results[exp_type] = run_experiment(
            X_raw, y_raw, 
            experiment_name="xgboost",
            num_classes=num_classes,
            include_base=include_base
        )


if __name__ == "__main__":
    main()
