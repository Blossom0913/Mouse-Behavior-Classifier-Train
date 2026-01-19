import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import the models
from GMM import run_experiment as run_gmm_experiment
from LSTM import run_experiment as run_lstm_experiment
from CNN import run_experiment as run_cnn_experiment
from LightGBM import run_experiment as run_lightgbm_experiment
from XGBoost import run_experiment as run_xgboost_experiment
from RandomForest import run_experiment as run_random_forest_experiment
from SVM import run_experiment as run_svm_experiment
from HMM import run_experiment as run_hmm_experiment
from data_load import load_mouse_data

def run_multiple_experiments(model_name, experiment_func, X_raw, y_raw, num_classes, 
                           include_base, n_runs=5, **kwargs):
    """
    Run the same experiment multiple times to get statistics for error bars
    """
    print(f"\n=== Running {model_name} {n_runs} times ===")
    
    all_metrics = {
        'accuracy': [],
        'weighted_f1': [],
        'macro_f1': [],
        'best_class_f1': [],
        'worst_class_f1': [],
        'class_f1': []
    }
    
    successful_runs = 0
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        
        # Set different random seeds for each run to create variation
        np.random.seed(42 + run)
        # Also set torch random seed if available
        try:
            import torch
            torch.manual_seed(42 + run)
            torch.cuda.manual_seed_all(42 + run)
        except:
            pass
        
        # Generate different split seed for each run to ensure different train/val/test splits
        split_seed = 42 + run
        
        try:
            if model_name == "GMM":
                result = experiment_func(X_raw, y_raw, n_components=10, 
                                      num_classes=num_classes, include_base=include_base, 
                                      experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            elif model_name == "LSTM":
                result = experiment_func(X_raw, y_raw, batch_size=256, epochs=50, 
                                      device='cpu', num_classes=num_classes, 
                                      include_base=include_base, experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            elif model_name == "CNN":
                result = experiment_func(X_raw, y_raw, batch_size=256, epochs=50,
                                      device='cpu', num_classes=num_classes,
                                      include_base=include_base, experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            elif model_name == "LightGBM":
                result = experiment_func(X_raw, y_raw, num_classes=num_classes, 
                                      include_base=include_base, experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            elif model_name == "XGBoost":
                result = experiment_func(X_raw, y_raw, num_classes=num_classes, 
                                      include_base=include_base, experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            elif model_name == "RandomForest":
                result = experiment_func(X_raw, y_raw, num_classes=num_classes, 
                                      include_base=include_base, experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            elif model_name == "SVM":
                result = experiment_func(X_raw, y_raw, num_classes=num_classes, 
                                      include_base=include_base, experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            elif model_name == "HMM":
                result = experiment_func(X_raw, y_raw, num_classes=num_classes, 
                                      include_base=include_base, experiment_name=f"{model_name.lower()}_{run}",
                                      split_seed=split_seed)
            else:
                print(f"Unknown model type: {model_name}")
                continue
            
            # Extract metrics
            metrics = result['test_metrics']
            # Print a compact summary only (avoid dumping y_true/y_pred arrays)
            compact_keys = ['accuracy', 'weighted_f1', 'macro_f1', 'best_class_f1', 'worst_class_f1']
            compact = {k: float(metrics.get(k, 0)) for k in compact_keys}
            print(f"  Run {run + 1} summary: {compact}")
            for key in all_metrics:
                if key == 'class_f1':
                    all_metrics[key].append(metrics[key])
                else:
                    all_metrics[key].append(metrics[key])
            
            successful_runs += 1
            print(f"✓ Run {run + 1} completed successfully")
                    
        except Exception as e:
            print(f"✗ Error in run {run + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nCompleted {successful_runs}/{n_runs} successful runs for {model_name}")
    
    # Debug: Print collected metrics
    print(f"Collected metrics for {model_name}:")
    for key, values in all_metrics.items():
        print(f"  {key}: {len(values)} values - {values}")
    
    # Calculate statistics
    stats = {}
    for key in all_metrics:
        if all_metrics[key] and len(all_metrics[key]) > 1:  # Need at least 2 values for std
            values = np.array(all_metrics[key])
            if key == 'class_f1':
                # For class F1, calculate mean and std for each class
                mean_val = np.mean(values, axis=0)
                std_val = np.std(values, axis=0, ddof=1)  # Use ddof=1 for sample std
                stats[key] = {'mean': mean_val, 'std': std_val}
                print(f"  {key}: mean={mean_val}, std={std_val}")
            else:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)  # Use ddof=1 for sample std
                stats[key] = {'mean': mean_val, 'std': std_val}
                print(f"  {key}: mean={mean_val}, std={std_val}")
        elif all_metrics[key] and len(all_metrics[key]) == 1:
            # Only one successful run - can't calculate std
            values = np.array(all_metrics[key])
            if key == 'class_f1':
                mean_val = np.mean(values, axis=0)
                stats[key] = {
                    'mean': mean_val,
                    'std': np.zeros_like(mean_val)  # Zero std for single run
                }
                print(f"  {key}: mean={mean_val}, std=0 (single run)")
            else:
                mean_val = np.mean(values)
                stats[key] = {'mean': mean_val, 'std': 0.0}  # Zero std for single run
                print(f"  {key}: mean={mean_val}, std=0 (single run)")
        else:
            # No successful runs
            if key == 'class_f1':
                stats[key] = {'mean': np.zeros(num_classes), 'std': np.zeros(num_classes)}
                print(f"  {key}: no data")
            else:
                stats[key] = {'mean': 0, 'std': 0}
                print(f"  {key}: no data")
    
    return stats

def create_comparison_graphs(all_results, base_name='model_comparison'):
    """
    Create separate comparison graphs with error bars and save each to its own PNG.
    Files:
      - {base_name}_overall.png
      - {base_name}_per_class.png
      - {base_name}_best_worst.png
      - {base_name}_stability.png
    """
    # Style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    models = list(all_results.keys())
    metrics = ['accuracy', 'weighted_f1', 'macro_f1']

    # 1) Overall Performance
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    for i, metric in enumerate(metrics):
        means = [all_results[m][metric]['mean'] for m in models]
        stds = [all_results[m][metric]['std'] for m in models]
        plt.bar(x + i*width, means, width, label=metric.replace('_', ' ').title(), yerr=stds, capsize=5, alpha=0.85)
    plt.xlabel('Models')
    plt.ylabel('Score (%)')
    plt.title('Overall Performance Comparison')
    plt.xticks(x + width, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_name}_overall.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Per-Class F1
    plt.figure(figsize=(12, 6))
    classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    for i, model in enumerate(models):
        means = all_results[model]['class_f1']['mean']
        stds = all_results[model]['class_f1']['std']
        x_pos = np.arange(len(classes)) + i * 0.2
        plt.bar(x_pos, means, 0.2, label=model, yerr=stds, capsize=3, alpha=0.85)
    plt.xlabel('Classes')
    plt.ylabel('F1 Score (%)')
    plt.title('Per-Class F1 Scores')
    plt.xticks(np.arange(len(classes)) + 0.3, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_name}_per_class.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Best vs Worst Class F1
    plt.figure(figsize=(10, 6))
    best_means = [all_results[m]['best_class_f1']['mean'] for m in models]
    best_stds = [all_results[m]['best_class_f1']['std'] for m in models]
    worst_means = [all_results[m]['worst_class_f1']['mean'] for m in models]
    worst_stds = [all_results[m]['worst_class_f1']['std'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, best_means, width, label='Best Class F1', yerr=best_stds, capsize=5, alpha=0.85, color='green')
    plt.bar(x + width/2, worst_means, width, label='Worst Class F1', yerr=worst_stds, capsize=5, alpha=0.85, color='red')
    plt.xlabel('Models')
    plt.ylabel('F1 Score (%)')
    plt.title('Best vs Worst Class Performance')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_name}_best_worst.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4) Stability (Coefficient of Variation)
    plt.figure(figsize=(10, 6))
    cv_accuracy = []
    for m in models:
        mean = all_results[m]['accuracy']['mean']
        std = all_results[m]['accuracy']['std']
        cv = (std / mean) * 100 if mean > 0 else 0
        cv_accuracy.append(cv)
    bars = plt.bar(models, cv_accuracy, alpha=0.85, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title('Performance Stability (Lower is Better)')
    plt.grid(True, alpha=0.3)
    for bar, cv in zip(bars, cv_accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{cv:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"{base_name}_stability.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {base_name}_overall.png, {base_name}_per_class.png, {base_name}_best_worst.png, {base_name}_stability.png")

def create_detailed_statistics_table(all_results):
    """
    Create a detailed statistics table
    """
    print("\n" + "="*80)
    print("DETAILED STATISTICS TABLE")
    print("="*80)
    
    models = list(all_results.keys())
    metrics = ['accuracy', 'weighted_f1', 'macro_f1', 'best_class_f1', 'worst_class_f1']
    
    # Header
    header = f"{'Model':<12} {'Accuracy':<12} {'Weighted F1':<12} {'Macro F1':<12} {'Best F1':<12} {'Worst F1':<12}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for model in models:
        row = f"{model:<12}"
        for metric in metrics:
            mean = all_results[model][metric]['mean']
            std = all_results[model][metric]['std']
            row += f"{mean:.1f}±{std:.1f}".ljust(12)
        print(row)
    
    print("\n" + "="*80)
    print("PER-CLASS F1 SCORES")
    print("="*80)
    
    # Per-class F1 scores
    classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    header = f"{'Model':<12} " + " ".join([f"{cls:<12}" for cls in classes])
    print(header)
    print("-" * len(header))
    
    for model in models:
        row = f"{model:<12}"
        means = all_results[model]['class_f1']['mean']
        stds = all_results[model]['class_f1']['std']
        
        for i in range(len(classes)):
            mean = means[i] if i < len(means) else 0
            std = stds[i] if i < len(stds) else 0
            row += f"{mean:.1f}±{std:.1f}".ljust(12)
        print(row)

def main():
    """
    Main function to run all experiments and generate comparison graphs
    """
    print("=== Model Comparison with Error Bars ===")
    print("Running multiple experiments for each model to generate error bars...")
    
    # Load data
    print("\nLoading data...")
    X_raw, y_raw = load_mouse_data(
        "../data/dataset58/feature8_58.xlsx",
        "../data/dataset58/merged_labels.xlsx"
    )
    
    # Configuration
    n_runs = 5  # Number of runs for each model
    num_classes = 4  # Focus on 4-class classification
    include_base = True
    
    # Define experiments
    experiments = {
        'CNN': run_cnn_experiment,
        'GMM': run_gmm_experiment,
        'LSTM': run_lstm_experiment,
        'LightGBM': run_lightgbm_experiment,
        'XGBoost': run_xgboost_experiment,
        'RandomForest': run_random_forest_experiment,
        'SVM': run_svm_experiment,
        'HMM': run_hmm_experiment
    }
    
    # Run all experiments
    all_results = {}
    
    for model_name, experiment_func in experiments.items():
        print(f"\n{'='*50}")
        print(f"Running {model_name} experiments...")
        print(f"{'='*50}")
        
        try:
            results = run_multiple_experiments(
                model_name, experiment_func, X_raw, y_raw, 
                num_classes, include_base, n_runs
            )
            all_results[model_name] = results
            print(f"✓ {model_name} completed successfully")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate graphs and statistics
    if all_results:
        print(f"\n{'='*50}")
        print("Generating comparison graphs with error bars...")
        print(f"{'='*50}")
        
        create_comparison_graphs(all_results)
        create_detailed_statistics_table(all_results)
        
        print(f"\n✓ All experiments completed successfully!")
        print(f"✓ Generated {len(all_results)} model comparisons with error bars")
    else:
        print("✗ No experiments completed successfully")

if __name__ == "__main__":
    main()
