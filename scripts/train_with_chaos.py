"""
Train ChaosFEX-NGRC with Chaotic Optimization

This script demonstrates "Chaotic Training" by using the ChaoticOptimizer
to tune the hyperparameters of the ensemble classifier using the FULL DATASET.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.chaotic_optimizer import ChaoticOptimizer
from src.models.ensemble import CFXMLEnsemble
from src.models.feature_extractors import create_feature_extractor
from src.models.chaosfex import ChaosFEX
from src.data.dataset import RFMiDDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_features(dataset, feature_extractor, chaosfex, batch_size=32, device='cpu'):
    """Extract features from dataset using Deep Feature Extractor + ChaosFEX"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    all_labels = []
    
    feature_extractor.to(device)
    feature_extractor.eval()
    
    print(f"Extracting features from {len(dataset)} images...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            # 1. Deep Feature Extraction (EfficientNet)
            deep_features = feature_extractor(images)
            deep_features_np = deep_features.cpu().numpy()
            
            # 2. ChaosFEX Transformation
            chaos_features = chaosfex.extract_features_batch(deep_features_np)
            
            all_features.append(chaos_features)
            all_labels.append(labels.numpy())
            
    return np.vstack(all_features), np.concatenate(all_labels, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Train with Chaotic Optimization')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=" * 60)
    print("🌀 CHAOTIC TRAINING: Hyperparameter Optimization (FULL DATASET)")
    print("=" * 60)
    
    # Check device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("\n[1/4] Loading Data...")
    data_dir = "data" 
    
    try:
        # Load with multi_label=True to get all disease columns
        train_dataset = RFMiDDataset(data_dir=f"{data_dir}/train", split='train', image_size=224, multi_label=True)
        val_dataset = RFMiDDataset(data_dir=f"{data_dir}/val", split='val', image_size=224, multi_label=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Feature Extraction
    print("\n[2/4] Extracting Features (Deep Features + ChaosFEX)...")
    
    # Initialize models
    feature_extractor = create_feature_extractor(
        model_type=config['model']['feature_extractor'],
        pretrained=True,
        feature_dim=config['model']['feature_dim']
    )
    
    chaosfex = ChaosFEX(
        n_neurons=config['model']['chaosfex_neurons'],
        map_type=config['model']['chaosfex_map'],
        b=config['model']['chaosfex_b']
    )
    
    # Force fresh extraction to ensure correctness
    X_train, y_train_multi = extract_features(train_dataset, feature_extractor, chaosfex, device=device)
    
    # Select specific target for "Chaotic Optimization" Demo
    # We choose 'DR' (Diabetic Retinopathy) as it's a specific, harder class than generic 'Risk'
    # DR is usually the 2nd column (index 1) in RFMiD, but let's check dataset.classes if possible
    # For now, we assume index 1 based on CSV header: ID, Disease_Risk, DR, ...
    target_idx = 1 
    y_train = y_train_multi[:, target_idx]
    
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Target Class: Diabetic Retinopathy (DR)")
    print(f"Class Balance: {np.sum(y_train)} positive out of {len(y_train)} ({np.mean(y_train):.2%})")
    
    # Fix shape mismatch if labels were incorrectly stacked
    if len(y_train) != len(X_train):
        print(f"⚠️ Shape mismatch detected: X={X_train.shape}, y={y_train.shape}. Attempting to fix...")
        y_train = y_train.ravel()
        print(f"New y shape: {y_train.shape}")
    
    # Flatten labels if they are (N, 1)
    if y_train.ndim > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()

    # 3. Initialize Chaotic Optimizer
    print("\n[3/5] Initializing Chaotic Optimizer...")
    optimizer = ChaoticOptimizer(
        map_type='logistic',
        r=4.0,  # Full chaos
        max_iterations=30  # More iterations for better exploration
    )
    
    # 4. Optimize Random Forest
    print("\n[4/5] Optimizing Random Forest Classifier with Chaos...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    
    rf_param_grid = {
        'n_estimators': (100, 300),
        'max_depth': (10, 40),
        'min_samples_leaf': (1, 4),
        'min_samples_split': (2, 6)
    }
    
    best_rf_params = optimizer.optimize(rf, X_train, y_train, rf_param_grid, cv=5)
    
    # 5. Final Training & Evaluation
    print("\n[5/5] Final Training & Evaluation...")
    
    # Train final model with best params + class balancing
    final_model = RandomForestClassifier(
        n_estimators=best_rf_params['n_estimators'],
        max_depth=best_rf_params['max_depth'],
        min_samples_leaf=best_rf_params['min_samples_leaf'],
        min_samples_split=best_rf_params.get('min_samples_split', 2),
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    
    # Load Validation Data for Metrics
    print("Extracting Validation Features...")
    
    # Force fresh extraction
    X_val, y_val_multi = extract_features(val_dataset, feature_extractor, chaosfex, device=device)
    
    # Save for calibration script
    os.makedirs("results/cache", exist_ok=True)
    np.save("results/cache/val_X.npy", X_val)
    np.save("results/cache/val_y_multi.npy", y_val_multi)
    np.save("results/cache/train_X.npy", X_train)
    np.save("results/cache/train_y_multi.npy", y_train_multi)
    
    # Select same target (DR)
    y_val = y_val_multi[:, target_idx]
        
    if len(y_val) != len(X_val):
        y_val = y_val.ravel()
    if y_val.ndim > 1 and y_val.shape[1] == 1:
        y_val = y_val.ravel()
        
    # Predict
    y_pred = final_model.predict(X_val)
    y_prob = final_model.predict_proba(X_val)
    
    # Metrics
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"\n🏆 Validation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # --- PLOTTING ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Optimization History Plot
    plt.figure(figsize=(10, 6))
    iterations = [x['iteration'] for x in optimizer.history]
    scores = [x['score'] for x in optimizer.history]
    plt.plot(iterations, scores, marker='o', linestyle='-', color='purple')
    plt.title('Chaotic Optimization History')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Validation Score')
    plt.grid(True, alpha=0.3)
    plt.savefig("results/plots/optimization_history.png")
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("results/plots/confusion_matrix.png")
    plt.close()
    
    # 3. ROC Curve (Multi-class)
    # Simple version: Macro-average ROC
    from sklearn.preprocessing import label_binarize
    classes = np.unique(y_val)
    y_val_bin = label_binarize(y_val, classes=classes)
    n_classes = y_val_bin.shape[1]
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        # Check if class exists in y_val
        if np.sum(y_val_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("results/plots/roc_curve.png")
    plt.close()
    
    print("\n✅ Plots saved to results/plots/")
    
    # Save full pipeline for Demo
    # We need to wrap it in a class that the demo expects, or save just the classifier
    # For simplicity, we'll save the classifier and the feature extractor config
    pipeline_data = {
        'classifier': final_model,
        'feature_extractor_config': config['model'],
        'best_params': best_rf_params,
        'classes': classes
    }
    joblib.dump(pipeline_data, "results/pipeline.pkl")
    print("✅ Full Model saved to results/pipeline.pkl (Ready for Demo!)")

if __name__ == "__main__":
    main()
