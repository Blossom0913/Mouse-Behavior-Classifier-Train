import torch
import numpy as np
import pandas as pd
import os
import sys

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_solver import load_mouse_data
import CNN
import LSTM

# Configuration
SEEDS = [42, 100, 2024, 7, 999]
EPOCHS = 50 
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_multi_seed_experiments():
    # Define paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_path = os.path.join(base_dir, "../data/dataset58/feature8_58.xlsx")
    
    # OPTION: Use aggression labels and exclude class 0
    # Set this to True to use merged_labels_aggression.xlsx and filter out class 0
    USE_AGGRESSION_LABELS = False 

    if USE_AGGRESSION_LABELS:
        label_path = os.path.join(base_dir, "../data/dataset58/merged_labels_aggression.xlsx")
    else:
        label_path = os.path.join(base_dir, "../data/dataset58/merged_labels.xlsx")
    
    print(f"Loading data from {feature_path} and {label_path}...")
    
    try:
        X_raw, y_raw = load_mouse_data(feature_path, label_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure same length as features (Fix for IndexError)
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        print(f"Mismatch in length: X={len(X_raw)}, y={len(y_raw)}. Truncating to {min_len}.")
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]

    # Filter out class 0 regardless of which label file is used (User Request)
    print("Filtering out class 0 (Based on user instruction)...")
    mask = y_raw != 0
    original_count = len(y_raw)
    X_raw = X_raw[mask]
    y_raw = y_raw[mask]
    print(f"Filtered {original_count - len(y_raw)} samples. Remaining: {len(y_raw)}")
    
    unique_classes = np.unique(y_raw)
    print(f"Remaining classes: {unique_classes}")
    
    # Dynamically determine number of classes
    # This handles both the 3-class case (merged_labels) and 6-class case (aggression)
    num_classes = len(unique_classes)
    include_base = False
    
    print(f"Configuration: num_classes={num_classes}, include_base={include_base}")

    results = {
        'CNN': {'accuracy': [], 'macro_f1': []},
        'LSTM': {'accuracy': [], 'macro_f1': []}
    }
    
    for seed in SEEDS:
        print(f"\n{'='*20} RUNNING WITH SEED {seed} {'='*20}")
        
        # --- CNN ---
        print(f"\nRunning CNN (Seed {seed})...")
        try:
            cnn_res = CNN.run_experiment(
                X_raw, y_raw, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                device=DEVICE, 
                num_classes=num_classes, 
                include_base=include_base, 
                experiment_name=f"cnn_seed{seed}", 
                split_seed=seed
            )
            results['CNN']['accuracy'].append(cnn_res['test_metrics']['accuracy'])
            results['CNN']['macro_f1'].append(cnn_res['test_metrics']['macro_f1'])
        except Exception as e:
            print(f"CNN failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
        
        # --- LSTM ---
        print(f"\nRunning LSTM (Seed {seed})...")
        try:
            lstm_res = LSTM.run_experiment(
                X_raw, y_raw, 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS, 
                device=DEVICE, 
                num_classes=num_classes, 
                include_base=include_base, 
                experiment_name=f"lstm_seed{seed}", 
                split_seed=seed
            )
            results['LSTM']['accuracy'].append(lstm_res['test_metrics']['accuracy'])
            results['LSTM']['macro_f1'].append(lstm_res['test_metrics']['macro_f1'])
        except Exception as e:
            print(f"LSTM failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("FINAL AGGREGATE RESULTS (5 Runs)")
    print("="*50)
    
    for model_name in ['CNN', 'LSTM']:
        accs = results[model_name]['accuracy']
        f1s = results[model_name]['macro_f1']
        
        if len(accs) > 0:
            print(f"\nModel: {model_name}")
            print(f"  Accuracy: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
            print(f"  Macro F1: {np.mean(f1s):.2f}% +/- {np.std(f1s):.2f}%")
            print(f"  Raw Accuracies: {accs}")
        else:
            print(f"\nModel: {model_name} - No successful runs.")

if __name__ == "__main__":
    run_multi_seed_experiments()
