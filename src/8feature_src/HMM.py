"""
Hidden Markov Model (HMM) Classifier for Mouse Behavior Classification
Uses one Gaussian HMM per class and classifies based on log-likelihood
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. Run 'pip install hmmlearn'")

from data_load import load_mouse_data


def save_model(models, scaler, experiment_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"hmm_{experiment_type}_{timestamp}.pkl"
    joblib.dump({'models': models, 'scaler': scaler}, model_name)
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
                   experiment_name="hmm", split_seed=None,
                   n_components=3, n_iter=100, covariance_type='diag'):
    """
    Run HMM experiment with stratified splitting
    
    Uses one Gaussian HMM per class. Each sample is classified based on 
    which class's HMM gives the highest log-likelihood.
    
    Args:
        X_raw: Raw feature matrix
        y_raw: Raw labels
        num_classes: Number of classes
        include_base: Whether to include base class
        experiment_name: Name for saved model
        split_seed: Random seed for data splitting (different seeds = different splits)
        n_components: Number of hidden states in each HMM
        n_iter: Maximum number of EM iterations
        covariance_type: Type of covariance parameters ('diag', 'full', 'spherical', 'tied')
    
    Returns:
        Dictionary containing test metrics and model path
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn is not installed. Run 'pip install hmmlearn'")
    
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
    
    # Train one HMM per class
    models = {}
    for cls in range(num_classes):
        # Get training data for this class
        cls_data = X_train[y_train == cls]
        
        if len(cls_data) == 0:
            print(f"Warning: No training data for class {cls}")
            continue
        
        # Adjust n_components if necessary
        n_comp = min(n_components, max(1, len(cls_data) // 2))
        
        # Create and train HMM for this class
        model = hmm.GaussianHMM(
            n_components=n_comp,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=split_seed,  # Use split_seed for model randomness
            verbose=False
        )
        
        try:
            # For HMM, we need to treat each sample as a sequence of length 1
            # or reshape data appropriately
            # Here we treat each feature vector as a single observation
            model.fit(cls_data)
            models[cls] = model
            print(f"  Trained HMM for class {cls} with {n_comp} components on {len(cls_data)} samples")
        except Exception as e:
            print(f"  Warning: Failed to train HMM for class {cls}: {e}")
            continue
    
    if len(models) == 0:
        raise ValueError("No HMM models were successfully trained")
    
    # Save model
    model_path = save_model(models, scaler, f"{experiment_name}_{num_classes}class")
    
    def predict_class(X_samples, models, num_classes):
        """Predict class based on highest log-likelihood"""
        predictions = []
        for x in X_samples:
            x_reshaped = x.reshape(1, -1)  # Single observation
            log_likelihoods = []
            
            for cls in range(num_classes):
                if cls in models:
                    try:
                        ll = models[cls].score(x_reshaped)
                        log_likelihoods.append(ll)
                    except:
                        log_likelihoods.append(float('-inf'))
                else:
                    log_likelihoods.append(float('-inf'))
            
            predictions.append(np.argmax(log_likelihoods))
        
        return np.array(predictions)
    
    # Validation evaluation
    val_preds = predict_class(X_val, models, num_classes)
    val_accuracy = 100 * accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # Test evaluation
    test_preds = predict_class(X_test, models, num_classes)
    
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
    
    # Print HMM info
    print(f"\nNumber of HMM models trained: {len(models)}")
    for cls, model in models.items():
        print(f"  Class {cls}: {model.n_components} hidden states")
    
    # Print detailed classification report
    print("\nClassification Report (test):")
    print(classification_report(y_test, test_preds, digits=4))
    
    return {
        'test_metrics': test_metrics,
        'model_path': model_path
    }


def main():
    if not HMM_AVAILABLE:
        print("Error: hmmlearn is not installed. Run 'pip install hmmlearn'")
        return
    
    # Load data
    X_raw, y_raw = load_mouse_data(
        "../data/dataset58/feature8_58.xlsx",
        "../data/dataset58/merged_labels.xlsx"
    )
    
    # Run experiments
    results = {}
    for include_base, num_classes in [(True, 4), (False, 3)]:
        exp_type = f"{num_classes}class"
        print(f"\n=== STARTING HMM {exp_type.upper()} EXPERIMENT ===")
        results[exp_type] = run_experiment(
            X_raw, y_raw, 
            experiment_name="hmm",
            num_classes=num_classes,
            include_base=include_base
        )


if __name__ == "__main__":
    main()
