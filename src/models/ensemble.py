"""
CFX+ML Ensemble Classifier

Combines ChaosFEX features with traditional ML classifiers:
- Random Forest
- Support Vector Machine (SVM)
- AdaBoost
- k-Nearest Neighbors (k-NN)
- Gaussian Naive Bayes (GNB)
- Decision Tree

Uses soft voting for ensemble predictions.
"""

import numpy as np
from typing import List, Optional, Literal
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


class CFXMLEnsemble:
    """
    CFX+ML Ensemble Classifier
    
    Args:
        classifiers: List of classifier names to include
        voting: Voting method ('hard' or 'soft')
        use_scaling: Whether to scale features
        handle_imbalance: Method to handle class imbalance ('smote', 'undersample', None)
    """
    
    def __init__(
        self,
        classifiers: Optional[List[str]] = None,
        voting: Literal['hard', 'soft'] = 'soft',
        use_scaling: bool = True,
        handle_imbalance: Optional[Literal['smote', 'undersample']] = 'smote',
        random_seed: int = 42
    ):
        if classifiers is None:
            classifiers = ['rf', 'svm', 'adaboost', 'knn', 'gnb']
        
        self.classifier_names = classifiers
        self.voting = voting
        self.use_scaling = use_scaling
        self.handle_imbalance = handle_imbalance
        self.random_seed = random_seed
        
        # Initialize components
        self.scaler = StandardScaler() if use_scaling else None
        self.ensemble = None
        self.individual_classifiers = {}
        
        # Build ensemble
        self._build_ensemble()
    
    def _create_classifier(self, name: str):
        """Create individual classifier"""
        if name == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=self.random_seed,
                n_jobs=-1
            )
        elif name == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=self.random_seed
            )
        elif name == 'adaboost':
            return AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=self.random_seed
            )
        elif name == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            )
        elif name == 'gnb':
            return GaussianNB()
        elif name == 'dt':
            return DecisionTreeClassifier(
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=self.random_seed
            )
        else:
            raise ValueError(f"Unknown classifier: {name}")
    
    def _build_ensemble(self):
        """Build voting ensemble"""
        estimators = []
        
        for name in self.classifier_names:
            clf = self._create_classifier(name)
            self.individual_classifiers[name] = clf
            estimators.append((name, clf))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train ensemble
        
        Args:
            X: Training features (N x D)
            y: Training labels (N,) or (N, n_classes) for multi-label
        """
        # Handle imbalance
        if self.handle_imbalance is not None and y.ndim == 1:
            X, y = self._handle_imbalance(X, y)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Train ensemble
        self.ensemble.fit(X, y)
    
    def _handle_imbalance(self, X: np.ndarray, y: np.ndarray):
        """Handle class imbalance"""
        if self.handle_imbalance == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=self.random_seed)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif self.handle_imbalance == 'undersample':
            # Random undersampling
            rus = RandomUnderSampler(random_state=self.random_seed)
            X_resampled, y_resampled = rus.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        return X_resampled, y_resampled
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Predicted labels (N,)
        """
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Class probabilities (N x n_classes)
        """
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        if self.voting == 'soft':
            return self.ensemble.predict_proba(X)
        else:
            # For hard voting, use individual classifier probabilities
            probas = []
            for name, clf in self.individual_classifiers.items():
                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(X)
                    probas.append(proba)
            
            # Average probabilities
            return np.mean(probas, axis=0)
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance from tree-based models
        
        Returns:
            Dictionary of feature importances
        """
        importances = {}
        
        for name, clf in self.individual_classifiers.items():
            if hasattr(clf, 'feature_importances_'):
                importances[name] = clf.feature_importances_
        
        return importances


class MultiLabelCFXMLEnsemble:
    """
    Multi-label CFX+ML Ensemble for concurrent diseases
    """
    
    def __init__(
        self,
        classifiers: Optional[List[str]] = None,
        use_scaling: bool = True,
        random_seed: int = 42
    ):
        if classifiers is None:
            classifiers = ['rf', 'svm', 'knn']
        
        self.classifier_names = classifiers
        self.use_scaling = use_scaling
        self.random_seed = random_seed
        
        self.scaler = StandardScaler() if use_scaling else None
        self.multi_output_classifiers = {}
        
        # Build multi-output classifiers
        self._build_classifiers()
    
    def _build_classifiers(self):
        """Build multi-output classifiers"""
        for name in self.classifier_names:
            base_clf = self._create_base_classifier(name)
            self.multi_output_classifiers[name] = MultiOutputClassifier(base_clf)
    
    def _create_base_classifier(self, name: str):
        """Create base classifier"""
        if name == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.random_seed,
                n_jobs=-1
            )
        elif name == 'svm':
            return SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=self.random_seed
            )
        elif name == 'knn':
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classifier: {name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train multi-label ensemble
        
        Args:
            X: Training features (N x D)
            y: Training labels (N x n_classes) - binary multi-label
        """
        # Scale features
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Train each classifier
        for name, clf in self.multi_output_classifiers.items():
            clf.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict multi-label outputs (ensemble voting)
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Predicted labels (N x n_classes)
        """
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Collect predictions from all classifiers
        all_predictions = []
        for name, clf in self.multi_output_classifiers.items():
            pred = clf.predict(X)
            all_predictions.append(pred)
        
        # Majority voting
        all_predictions = np.array(all_predictions)
        ensemble_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
        
        return ensemble_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (averaged across ensemble)
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Class probabilities (N x n_classes)
        """
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Collect probabilities from classifiers that support it
        all_probas = []
        for name, clf in self.multi_output_classifiers.items():
            if hasattr(clf.estimators_[0], 'predict_proba'):
                # Get probabilities for each output
                probas = []
                for estimator in clf.estimators_:
                    proba = estimator.predict_proba(X)[:, 1]  # Probability of class 1
                    probas.append(proba)
                probas = np.array(probas).T  # (N x n_classes)
                all_probas.append(probas)
        
        # Average probabilities
        if all_probas:
            return np.mean(all_probas, axis=0)
        else:
            # Fallback to predictions
            return self.predict(X).astype(float)


if __name__ == "__main__":
    # Example usage
    print("CFX+ML Ensemble Example")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    N_train = 200
    N_test = 50
    feature_dim = 400  # ChaosFEX features
    n_classes = 5
    
    X_train = np.random.randn(N_train, feature_dim)
    y_train = np.random.randint(0, n_classes, N_train)
    
    X_test = np.random.randn(N_test, feature_dim)
    
    # Create and train ensemble
    ensemble = CFXMLEnsemble(
        classifiers=['rf', 'svm', 'knn'],
        voting='soft',
        handle_imbalance='smote'
    )
    
    print("Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Predict
    predictions = ensemble.predict(X_test)
    probabilities = ensemble.predict_proba(X_test)
    
    print(f"Training samples: {N_train}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {n_classes}")
    print(f"Test predictions shape: {predictions.shape}")
    print(f"Test probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions[:10]}")
    
    # Feature importance
    importances = ensemble.get_feature_importance()
    print(f"\nFeature importance available for: {list(importances.keys())}")
    
    # Multi-label example
    print("\nMulti-Label Ensemble Example")
    print("=" * 50)
    
    y_train_multilabel = np.random.randint(0, 2, (N_train, n_classes))
    
    ml_ensemble = MultiLabelCFXMLEnsemble(
        classifiers=['rf', 'knn']
    )
    
    print("Training multi-label ensemble...")
    ml_ensemble.fit(X_train, y_train_multilabel)
    
    ml_predictions = ml_ensemble.predict(X_test)
    ml_probabilities = ml_ensemble.predict_proba(X_test)
    
    print(f"Multi-label predictions shape: {ml_predictions.shape}")
    print(f"Sample prediction: {ml_predictions[0]}")
    print(f"Sample probabilities: {ml_probabilities[0]}")
