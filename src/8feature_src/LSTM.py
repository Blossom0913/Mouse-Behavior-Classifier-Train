import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from data_load import *
# from data_windowing import create_windowed_data, WindowedMouseDataset # Removed for reverting

# Modified LSTM model for single time-step classification
class BehaviorLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=4, dropout=0.5):
        super(BehaviorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Since we're not using windowing, we'll treat each sample as a single time step
        # We can still use LSTM to learn complex feature interactions
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch_size, 1, input_size) - treat each sample as single time step
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add time dimension if not present
        
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the LSTM
        # lstm_out shape: (Batch, Seq, Hidden*2)
        # We want to flatten/squeeze if Seq=1
        
        # For single step, we can just take the last step or squeeze
        # Since we unsqueezed dim 1, we squeeze it back from the output?
        # Typically LSTM output is (Batch, Seq, Features)
        
        # If we take the last time step:
        lstm_output = lstm_out[:, -1, :] 
        
        # Apply dropout and classification
        out = self.dropout(lstm_output)
        out = self.fc(out)
        return out

def save_model(model, experiment_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "lstm_{}_{}.pth".format(experiment_type, timestamp)
    torch.save(model.state_dict(), model_name)
    print("Model saved as {}".format(model_name))
    return model_name

# Update the compute_metrics function to match 1DCNN.py
def compute_metrics(y_true, y_pred, num_classes):
    acc = 100 * accuracy_score(y_true, y_pred)
    weighted_f1 = 100 * f1_score(y_true, y_pred, average='weighted')
    macro_f1 = 100 * f1_score(y_true, y_pred, average='macro')
    
    # Calculate per-class F1 scores
    class_f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes)) * 100
    best_class_f1 = np.max(class_f1)
    worst_class_f1 = np.min(class_f1)
    
    return {
        'accuracy': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'best_class_f1': best_class_f1,
        'worst_class_f1': worst_class_f1,
        'class_f1': class_f1
    }

def run_experiment(X_raw, y_raw, batch_size, epochs, device, 
                   num_classes, include_base=True, experiment_name="lstm", split_seed=None):
    # Ensure X_raw and y_raw have the same length
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        print("Warning: X_raw and y_raw have different lengths ({} vs {}). Trimming to {}.".format(len(X_raw), len(y_raw), min_len))
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]
    
    # Data preparation
    # For aggression dataset, we need to handle non-contiguous class labels
    # First, get unique classes in the data
    unique_classes = np.unique(y_raw)
    print("Unique classes in data: {}".format(unique_classes))
    
    # Use all available classes in the data
    X_raw_exp, y_raw_exp = X_raw, y_raw
    
    # Map non-contiguous classes to contiguous range [0, num_classes-1]
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    reverse_mapping = {idx: cls for cls, idx in class_mapping.items()}
    print("Class mapping: {}".format(class_mapping))
    
    # Apply mapping to labels
    y_raw_exp = np.array([class_mapping[label] for label in y_raw_exp])
    
    # Update num_classes to match actual number of unique classes
    actual_num_classes = len(unique_classes)
    print("Updated num_classes from {} to {}".format(num_classes, actual_num_classes))
    num_classes = actual_num_classes
    
    # Strict Temporal Split to prevent data leakage (Fix for time-series)
    # Train: 0-70%, Val: 70-85%, Test: 85-100%
    # Note: split_seed is used for model initialization, not data splitting
    if split_seed is not None:
        torch.manual_seed(split_seed)
        np.random.seed(split_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(split_seed)

    total_samples = len(X_raw_exp)
    train_end = int(total_samples * 0.70)
    val_end = int(total_samples * 0.85)

    X_train_raw = X_raw_exp[:train_end]
    y_train = y_raw_exp[:train_end]

    X_val_raw = X_raw_exp[train_end:val_end]
    y_val = y_raw_exp[train_end:val_end]

    X_test_raw = X_raw_exp[val_end:]
    y_test = y_raw_exp[val_end:]

    print("Data Split (Temporal):")
    print("  Train: {} samples".format(len(X_train_raw)))
    print("  Val:   {} samples".format(len(X_val_raw)))
    print("  Test:  {} samples".format(len(X_test_raw)))

    # Check for empty classes in splits
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique_in_split = np.unique(y_split)
        print("  Classes in {}: {}".format(name, unique_in_split))
        if len(unique_in_split) < num_classes:
            print("  WARNING: {} set is missing classes! Missing: {}".format(name, set(range(num_classes)) - set(unique_in_split)))

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Create datasets
    train_dataset = MouseDataset(X_train, y_train)
    val_dataset = MouseDataset(X_val, y_val)
    test_dataset = MouseDataset(X_test, y_test)
    
    # Create dataloaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    # Model setup
    model = BehaviorLSTM(input_size=X_train.shape[1], num_classes=num_classes).to(device)
    classes = np.unique(y_raw_exp)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_raw_exp)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4) # keep the updated slow learning rate
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Ensure we always have a checkpoint to load even if val_acc never improves
    model_path = save_model(model, "{}_{}class_initial".format(experiment_name, num_classes))
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = 15
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in loaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        history['train_loss'].append(train_loss / len(loaders['train']))
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in loaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss / len(loaders['val']))
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = save_model(model, "{}_{}class".format(experiment_name, num_classes))
            patience_counter = 0
            print("  -> New best model saved! (Acc: {:.2f}%)".format(best_val_acc))
        else:
            patience_counter += 1
            print("  -> No improvement. Patience: {}/{}".format(patience_counter, early_stopping_patience))
        
        print("Epoch {}/{} | Train Acc: {:.2f}% | Val Acc: {:.2f}%".format(epoch+1, epochs, train_acc, val_acc))
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print("Early stopping at epoch {}".format(epoch+1))
            break

    
    # Load best model for testing
    model.load_state_dict(torch.load(model_path))
    
    # Test evaluation
    test_loss, test_acc, test_f1, class_acc, test_metrics = evaluate_model(
        model, loaders['test'], criterion, device, num_classes
    )
    
    # Update the benchmark printing
    print("\n=== FINAL BENCHMARK RESULTS ===")
    print("Accuracy (%): {:.2f}".format(test_metrics['accuracy']))
    print("Weighted F1: {:.2f}".format(test_metrics['weighted_f1']))
    print("Macro F1: {:.2f}".format(test_metrics['macro_f1']))
    print("Best Class F1: {:.2f}".format(test_metrics['best_class_f1']))
    print("Worst Class F1: {:.2f}".format(test_metrics['worst_class_f1']))
    print("Per-class F1 Scores:")
    for i, f1 in enumerate(test_metrics['class_f1']):
        print("  Class {}: {:.2f}%".format(i, f1))
    
    # Print detailed classification report
    print("\nClassification Report (test):")
    print(classification_report(test_metrics['y_true'], test_metrics['y_pred'], digits=4))
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'model_path': model_path
    }

def evaluate_model(model, loader, criterion, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update class-wise counts
            for i in range(num_classes):
                idx = (labels == i)
                class_correct[i] += (preds[idx] == labels[idx]).sum().item()
                class_total[i] += idx.sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = 100 * accuracy_score(all_labels, all_preds)
    f1 = 100 * f1_score(all_labels, all_preds, average='weighted')
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                      for i in range(num_classes)]
    
    test_metrics = compute_metrics(
        y_true=np.array(all_labels), y_pred=np.array(all_preds), num_classes=num_classes
    )
    
    # Add y_true and y_pred for classification report
    test_metrics['y_true'] = all_labels
    test_metrics['y_pred'] = all_preds
    
    return avg_loss, accuracy, f1, class_accuracy, test_metrics

def main():
    # Hyperparameters
    CONFIG = {
        'batch_size': 256,  # Larger batch size since no windowing
        'epochs': 100,      # More epochs for LSTM convergence
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # Load data
    X_raw, y_raw = load_mouse_data(
        "../data/dataset58/feature8_58.xlsx",
        "../data/dataset58/merged_labels.xlsx"
    )
    
    # Run experiments
    results = {}
    for include_base, num_classes in [(True, 4), (False, 3)]:
        exp_type = "{}class".format(num_classes)
        print("\n=== STARTING {} EXPERIMENT ===".format(exp_type.upper()))
        history, model_path = run_experiment(
            X_raw, y_raw, 
            experiment_name="lstm",
            num_classes=num_classes,
            include_base=include_base,
            **CONFIG
        )
        results[exp_type] = {
            'history': history,
            'model_path': model_path
        }

if __name__ == "__main__":
    main()
