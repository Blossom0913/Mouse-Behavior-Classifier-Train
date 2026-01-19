"""
Support Vector Machine (SVM) Classifier for Mouse Behavior Classification
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

from data_load import load_mouse_data


def save_model(model, scaler, experiment_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"svm_{experiment_type}_{timestamp}.pkl"
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
                   experiment_name="svm", split_seed=None,
                   kernel='rbf', C=1.0, gamma='scale'):
    """
    Run SVM experiment with stratified splitting
    
    Args:
        X_raw: Raw feature matrix
        y_raw: Raw labels
        num_classes: Number of classes
        include_base: Whether to include base class
        experiment_name: Name for saved model
        split_seed: Random seed for data splitting (different seeds = different splits)
        kernel: SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        C: Regularization parameter
        gamma: Kernel coefficient
    
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
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    reverse_mapping = {idx: cls for cls, idx in class_mapping.items()}
    print(f"Class mapping: {class_mapping}")
    
    # Apply mapping to labels
    y_raw_exp = np.array([class_mapping[label] for label in y_raw_exp], dtype=np.int32)
    
    # Update num_classes to match actual number of unique classes
    actual_num_classes = len(unique_classes)
    print(f"Updated num_classes from {num_classes} to {actual_num_classes}")
    num_classes = actual_num_classes
    
    # Standardize features (crucial for SVM)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw_exp)
    
    # Split data using stratified splitting (70% train, 15% val, 15% test)
    if split_seed is None:
        split_seed = 42
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_raw_exp, test_size=0.3, random_state=split_seed, stratify=y_raw_exp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=split_seed, stratify=y_temp)
    
    # Create and train SVM model
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight='balanced',
        random_state=split_seed,  # Use split_seed for model randomness
        probability=True,  # Enable probability estimates
        decision_function_shape='ovr'
    )
    model.fit(X_train, y_train)
    
    # Save model
    model_path = save_model(model, scaler, f"{experiment_name}_{num_classes}class")
    
    # Validation evaluation
    val_preds = model.predict(X_val)
    val_accuracy = 100 * accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Test evaluation
    test_preds = model.predict(X_test)
    
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
    
    # Print support vectors info
    print(f"\nNumber of support vectors: {len(model.support_)}")
    print(f"Support vectors per class: {model.n_support_}")
    
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
        print(f"\n=== STARTING SVM {exp_type.upper()} EXPERIMENT ===")
        results[exp_type] = run_experiment(
            X_raw, y_raw, 
            experiment_name="svm",
            num_classes=num_classes,
            include_base=include_base
        )


if __name__ == "__main__":
    main()
