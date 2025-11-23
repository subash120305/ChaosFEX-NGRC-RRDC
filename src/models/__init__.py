"""
Complete NG-RC + ChaosFEX Pipeline

Integrates all components:
1. Deep Feature Extraction
2. ChaosFEX Transformation
3. NG-RC Processing
4. Classification (ChaosNet or CFX+ML Ensemble)
"""

import numpy as np
import torch
from typing import Literal, Optional, Dict, Any
from pathlib import Path
import pickle

from .feature_extractors import create_feature_extractor
from .chaosfex import ChaosFEX, MultiScaleChaosFEX
from .ngrc import SimpleNGRC, NextGenReservoirComputing
from .chaosnet import ChaosNet, MultiLabelChaosNet
from .ensemble import CFXMLEnsemble, MultiLabelCFXMLEnsemble


class NGRCChaosFEXPipeline:
    """
    Complete NG-RC + ChaosFEX Pipeline
    
    Args:
        feature_extractor_type: Type of deep feature extractor
        feature_dim: Dimension of deep features
        chaosfex_neurons: Number of ChaosFEX neurons
        chaosfex_map: Type of chaotic map ('GLS', 'Logistic', 'Hybrid')
        ngrc_reservoir_size: Size of NG-RC reservoir
        classifier_type: Type of classifier ('chaosnet', 'ensemble')
        multi_label: Whether to use multi-label classification
        use_multiscale_chaosfex: Whether to use multi-scale ChaosFEX
    """
    
    def __init__(
        self,
        feature_extractor_type: str = 'efficientnet_b3',
        feature_dim: int = 1024,
        chaosfex_neurons: int = 200,
        chaosfex_map: Literal['GLS', 'Logistic', 'Hybrid'] = 'GLS',
        chaosfex_b: float = 0.1,
        ngrc_reservoir_size: int = 300,
        ngrc_spectral_radius: float = 0.9,
        classifier_type: Literal['chaosnet', 'ensemble'] = 'ensemble',
        multi_label: bool = True,
        use_multiscale_chaosfex: bool = False,
        random_seed: int = 42
    ):
        self.config = {
            'feature_extractor_type': feature_extractor_type,
            'feature_dim': feature_dim,
            'chaosfex_neurons': chaosfex_neurons,
            'chaosfex_map': chaosfex_map,
            'chaosfex_b': chaosfex_b,
            'ngrc_reservoir_size': ngrc_reservoir_size,
            'ngrc_spectral_radius': ngrc_spectral_radius,
            'classifier_type': classifier_type,
            'multi_label': multi_label,
            'use_multiscale_chaosfex': use_multiscale_chaosfex,
            'random_seed': random_seed
        }
        
        # Initialize components
        print("Initializing NG-RC + ChaosFEX Pipeline...")
        
        # Stage 1: Deep Feature Extractor
        print(f"  [1/4] Loading {feature_extractor_type}...")
        self.feature_extractor = create_feature_extractor(
            model_type=feature_extractor_type,
            pretrained=True,
            feature_dim=feature_dim
        )
        
        # Stage 2: ChaosFEX
        print(f"  [2/4] Initializing ChaosFEX ({chaosfex_neurons} neurons, {chaosfex_map} map)...")
        if use_multiscale_chaosfex:
            self.chaosfex = MultiScaleChaosFEX(
                n_neurons_per_scale=chaosfex_neurons // 3,
                n_scales=3,
                map_type=chaosfex_map
            )
        else:
            self.chaosfex = ChaosFEX(
                n_neurons=chaosfex_neurons,
                map_type=chaosfex_map,
                b=chaosfex_b,
                random_seed=random_seed
            )
        
        # Stage 3: NG-RC
        print(f"  [3/4] Initializing NG-RC ({ngrc_reservoir_size} neurons)...")
        chaos_feature_dim = chaosfex_neurons * 4
        if use_multiscale_chaosfex:
            chaos_feature_dim *= 3
        
        self.ngrc = SimpleNGRC(
            input_dim=chaos_feature_dim,
            reservoir_size=ngrc_reservoir_size,
            spectral_radius=ngrc_spectral_radius,
            random_seed=random_seed
        )
        
        # Stage 4: Classifier
        print(f"  [4/4] Initializing {classifier_type} classifier...")
        self.classifier_type = classifier_type
        self.multi_label = multi_label
        self.classifier = None  # Will be initialized during training
        
        print("Pipeline initialized successfully!\n")
    
    def extract_deep_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract deep features from images
        
        Args:
            images: Input images (N x H x W x 3)
            
        Returns:
            Deep features (N x feature_dim)
        """
        return self.feature_extractor.extract_features_numpy(images)
    
    def extract_chaos_features(self, deep_features: np.ndarray) -> np.ndarray:
        """
        Extract ChaosFEX features from deep features
        
        Args:
            deep_features: Deep features (N x feature_dim)
            
        Returns:
            Chaos features (N x 4*chaosfex_neurons)
        """
        return self.chaosfex.extract_features_batch(deep_features)
    
    def extract_ngrc_features(self, chaos_features: np.ndarray) -> np.ndarray:
        """
        Extract NG-RC features from chaos features
        
        Args:
            chaos_features: Chaos features (N x chaos_dim)
            
        Returns:
            NG-RC reservoir states (N x reservoir_size)
        """
        return self.ngrc.get_states(chaos_features)
    
    def fit(self, images: np.ndarray, labels: np.ndarray, verbose: bool = True):
        """
        Train the complete pipeline
        
        Args:
            images: Training images (N x H x W x 3)
            labels: Training labels (N,) or (N x n_classes) for multi-label
            verbose: Whether to print progress
        """
        if verbose:
            print("Training NG-RC + ChaosFEX Pipeline")
            print("=" * 60)
        
        # Stage 1: Extract deep features
        if verbose:
            print("[1/4] Extracting deep features...")
        deep_features = self.extract_deep_features(images)
        if verbose:
            print(f"      Deep features shape: {deep_features.shape}")
        
        # Stage 2: Extract chaos features
        if verbose:
            print("[2/4] Extracting ChaosFEX features...")
        chaos_features = self.extract_chaos_features(deep_features)
        if verbose:
            print(f"      Chaos features shape: {chaos_features.shape}")
        
        # Stage 3: Extract NG-RC features
        if verbose:
            print("[3/4] Extracting NG-RC features...")
        ngrc_features = self.extract_ngrc_features(chaos_features)
        if verbose:
            print(f"      NG-RC features shape: {ngrc_features.shape}")
        
        # Stage 4: Train classifier
        if verbose:
            print(f"[4/4] Training {self.classifier_type} classifier...")
        
        if self.classifier_type == 'chaosnet':
            if self.multi_label:
                self.classifier = MultiLabelChaosNet(adaptive_threshold=True)
            else:
                self.classifier = ChaosNet(distance_metric='cosine')
        else:  # ensemble
            if self.multi_label:
                self.classifier = MultiLabelCFXMLEnsemble(
                    classifiers=['rf', 'svm', 'knn'],
                    random_seed=self.config['random_seed']
                )
            else:
                self.classifier = CFXMLEnsemble(
                    classifiers=['rf', 'svm', 'adaboost', 'knn', 'gnb'],
                    voting='soft',
                    handle_imbalance='smote',
                    random_seed=self.config['random_seed']
                )
        
        self.classifier.fit(ngrc_features, labels)
        
        if verbose:
            print("\nTraining completed successfully!")
            print("=" * 60)
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Predict labels for images
        
        Args:
            images: Input images (N x H x W x 3)
            
        Returns:
            Predicted labels (N,) or (N x n_classes)
        """
        # Extract features through pipeline
        deep_features = self.extract_deep_features(images)
        chaos_features = self.extract_chaos_features(deep_features)
        ngrc_features = self.extract_ngrc_features(chaos_features)
        
        # Predict
        return self.classifier.predict(ngrc_features)
    
    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for images
        
        Args:
            images: Input images (N x H x W x 3)
            
        Returns:
            Class probabilities (N x n_classes)
        """
        # Extract features through pipeline
        deep_features = self.extract_deep_features(images)
        chaos_features = self.extract_chaos_features(deep_features)
        ngrc_features = self.extract_ngrc_features(chaos_features)
        
        # Predict probabilities
        return self.classifier.predict_proba(ngrc_features)
    
    def save(self, save_path: str):
        """
        Save pipeline to disk
        
        Args:
            save_path: Path to save pipeline
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save components
        pipeline_data = {
            'config': self.config,
            'chaosfex': self.chaosfex,
            'ngrc': self.ngrc,
            'classifier': self.classifier
        }
        
        # Save feature extractor separately (PyTorch model)
        torch.save(
            self.feature_extractor.state_dict(),
            save_path.parent / f"{save_path.stem}_feature_extractor.pth"
        )
        
        # Save other components
        with open(save_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline saved to {save_path}")
    
    def load(self, load_path: str):
        """
        Load pipeline from disk
        
        Args:
            load_path: Path to load pipeline from
        """
        load_path = Path(load_path)
        
        # Load components
        with open(load_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        self.config = pipeline_data['config']
        self.chaosfex = pipeline_data['chaosfex']
        self.ngrc = pipeline_data['ngrc']
        self.classifier = pipeline_data['classifier']
        
        # Load feature extractor
        self.feature_extractor = create_feature_extractor(
            model_type=self.config['feature_extractor_type'],
            pretrained=False,
            feature_dim=self.config['feature_dim']
        )
        self.feature_extractor.load_state_dict(
            torch.load(load_path.parent / f"{load_path.stem}_feature_extractor.pth")
        )
        
        print(f"Pipeline loaded from {load_path}")


if __name__ == "__main__":
    # Example usage
    print("NG-RC + ChaosFEX Pipeline Example")
    print("=" * 60)
    
    # Create pipeline
    pipeline = NGRCChaosFEXPipeline(
        feature_extractor_type='efficientnet_b3',
        feature_dim=1024,
        chaosfex_neurons=100,
        chaosfex_map='GLS',
        ngrc_reservoir_size=200,
        classifier_type='ensemble',
        multi_label=True
    )
    
    # Generate synthetic data
    np.random.seed(42)
    N_train = 50
    N_test = 10
    
    # Simulate fundus images (224x224x3)
    train_images = np.random.randint(0, 255, (N_train, 224, 224, 3), dtype=np.uint8)
    train_labels = np.random.randint(0, 2, (N_train, 49))  # 49 diseases, multi-label
    
    test_images = np.random.randint(0, 255, (N_test, 224, 224, 3), dtype=np.uint8)
    
    # Train pipeline
    print("\nTraining pipeline...")
    pipeline.fit(train_images, train_labels, verbose=True)
    
    # Predict
    print("\nMaking predictions...")
    predictions = pipeline.predict(test_images)
    probabilities = pipeline.predict_proba(test_images)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample prediction: {predictions[0]}")
    print(f"Sample probabilities: {probabilities[0][:5]}")
