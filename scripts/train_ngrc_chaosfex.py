"""
ChaosFEX-NGRC: Training Script

Chaos-Based Feature Extraction with Next Generation Reservoir Computing
for Rare Retinal Disease Classification

Training Script for ChaosFEX-NGRC Pipeline on RFMiD 2.0

Usage:
    python scripts/train_ngrc_chaosfex.py --config config/config.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import NGRCChaosFEXPipeline
from src.data.dataset import RFMiDDataset, create_data_splits


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(config: dict):
    """Load RFMiD dataset"""
    data_dir = config['data']['data_dir']
    image_size = config['data']['image_size']
    multi_label = config['data']['multi_label']
    
    # Create splits if they don't exist
    splits_dir = Path(data_dir) / 'splits'
    if not splits_dir.exists():
        print("Creating train/val/test splits...")
        create_data_splits(
            data_dir=data_dir,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            random_seed=config['training']['random_seed']
        )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = RFMiDDataset(
        data_dir=str(splits_dir),
        split='train',
        image_size=image_size,
        multi_label=multi_label
    )
    
    val_dataset = RFMiDDataset(
        data_dir=str(splits_dir),
        split='val',
        image_size=image_size,
        multi_label=multi_label
    )
    
    test_dataset = RFMiDDataset(
        data_dir=str(splits_dir),
        split='test',
        image_size=image_size,
        multi_label=multi_label
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


def dataset_to_numpy(dataset):
    """Convert dataset to numpy arrays"""
    images = []
    labels = []
    
    print(f"Loading {len(dataset)} samples...")
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        
        # Convert tensor to numpy if needed
        if hasattr(image, 'numpy'):
            image = image.numpy()
        
        # Transpose from (C, H, W) to (H, W, C)
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = (image * 255).astype(np.uint8)
        
        images.append(image)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


def evaluate_model(pipeline, images, labels, multi_label=True):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    # Predict
    predictions = pipeline.predict(images)
    probabilities = pipeline.predict_proba(images)
    
    metrics = {}
    
    if multi_label:
        # Multi-label metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        
        # AUC-ROC per class
        try:
            auc_scores = []
            for i in range(labels.shape[1]):
                if len(np.unique(labels[:, i])) > 1:  # Only if both classes present
                    auc = roc_auc_score(labels[:, i], probabilities[:, i])
                    auc_scores.append(auc)
            auc_macro = np.mean(auc_scores) if auc_scores else 0.0
        except:
            auc_macro = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'auc_macro': auc_macro
        }
    else:
        # Single-label metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # AUC-ROC
        try:
            n_classes = probabilities.shape[1]
            labels_bin = label_binarize(labels, classes=range(n_classes))
            auc_macro = roc_auc_score(labels_bin, probabilities, average='macro', multi_class='ovr')
        except:
            auc_macro = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc_macro': auc_macro
        }
    
    return metrics, predictions, probabilities


def main():
    parser = argparse.ArgumentParser(description='Train NG-RC + ChaosFEX Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = output_dir / f"experiment_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_data(config)
    
    # Convert to numpy
    print("\nConverting datasets to numpy arrays...")
    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val = dataset_to_numpy(val_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    
    # Create pipeline
    print("\nCreating NG-RC + ChaosFEX Pipeline...")
    pipeline = NGRCChaosFEXPipeline(
        feature_extractor_type=config['model']['feature_extractor'],
        feature_dim=config['model']['feature_dim'],
        chaosfex_neurons=config['model']['chaosfex_neurons'],
        chaosfex_map=config['model']['chaosfex_map'],
        chaosfex_b=config['model']['chaosfex_b'],
        ngrc_reservoir_size=config['model']['ngrc_reservoir_size'],
        ngrc_spectral_radius=config['model']['ngrc_spectral_radius'],
        classifier_type=config['model']['classifier_type'],
        multi_label=config['data']['multi_label'],
        use_multiscale_chaosfex=config['model']['use_multiscale_chaosfex'],
        random_seed=config['training']['random_seed']
    )
    
    # Train pipeline
    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train, verbose=True)
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    val_metrics, val_preds, val_probs = evaluate_model(
        pipeline, X_val, y_val, multi_label=config['data']['multi_label']
    )
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    test_metrics, test_preds, test_probs = evaluate_model(
        pipeline, X_test, y_test, multi_label=config['data']['multi_label']
    )
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        'config': config,
        'val_metrics': {k: float(v) for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'timestamp': timestamp
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.save(exp_dir / 'val_predictions.npy', val_preds)
    np.save(exp_dir / 'val_probabilities.npy', val_probs)
    np.save(exp_dir / 'test_predictions.npy', test_preds)
    np.save(exp_dir / 'test_probabilities.npy', test_probs)
    
    # Save pipeline
    print(f"\nSaving pipeline to {exp_dir / 'pipeline.pkl'}...")
    pipeline.save(exp_dir / 'pipeline.pkl')
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
