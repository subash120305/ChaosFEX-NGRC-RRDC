"""
ChaosNet Classifier

Implements the ChaosNet classification algorithm based on cosine similarity
with class mean representation vectors.

Reference:
Harikrishnan NB, et al. "ChaosNet: A chaos based artificial neural network 
architecture for classification." Chaos, 2020.
"""

import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity


class ChaosNet:
    """
    ChaosNet Classifier using cosine similarity
    
    The classifier computes mean representation vectors for each class
    and assigns labels based on highest cosine similarity.
    
    Args:
        distance_metric: Distance/similarity metric ('cosine', 'euclidean', 'manhattan')
    """
    
    def __init__(self, distance_metric: str = 'cosine'):
        self.distance_metric = distance_metric
        self.mean_vectors = {}
        self.classes = None
        self.n_classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train ChaosNet by computing mean representation vectors
        
        Args:
            X: Training features (N x D) - ChaosFEX features
            y: Training labels (N,)
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Compute mean representation vector for each class
        for class_label in self.classes:
            class_samples = X[y == class_label]
            self.mean_vectors[class_label] = np.mean(class_samples, axis=0)
    
    def _compute_similarity(self, x: np.ndarray, mean_vector: np.ndarray) -> float:
        """
        Compute similarity between sample and mean vector
        
        Args:
            x: Sample feature vector
            mean_vector: Class mean representation vector
            
        Returns:
            Similarity score
        """
        if self.distance_metric == 'cosine':
            # Cosine similarity
            similarity = cosine_similarity(
                x.reshape(1, -1),
                mean_vector.reshape(1, -1)
            )[0, 0]
        elif self.distance_metric == 'euclidean':
            # Negative Euclidean distance (higher is more similar)
            similarity = -np.linalg.norm(x - mean_vector)
        elif self.distance_metric == 'manhattan':
            # Negative Manhattan distance
            similarity = -np.sum(np.abs(x - mean_vector))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return similarity
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Predicted labels (N,)
        """
        predictions = []
        
        for x in X:
            # Compute similarity with each class mean vector
            similarities = {}
            for class_label, mean_vector in self.mean_vectors.items():
                similarities[class_label] = self._compute_similarity(x, mean_vector)
            
            # Assign label with highest similarity
            predicted_label = max(similarities, key=similarities.get)
            predictions.append(predicted_label)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using softmax of similarities
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Class probabilities (N x n_classes)
        """
        probabilities = []
        
        for x in X:
            # Compute similarity with each class mean vector
            similarities = []
            for class_label in self.classes:
                mean_vector = self.mean_vectors[class_label]
                sim = self._compute_similarity(x, mean_vector)
                similarities.append(sim)
            
            # Convert to probabilities using softmax
            similarities = np.array(similarities)
            exp_sim = np.exp(similarities - np.max(similarities))  # Numerical stability
            probs = exp_sim / np.sum(exp_sim)
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (similarity scores)
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Similarity scores (N x n_classes)
        """
        scores = []
        
        for x in X:
            class_scores = []
            for class_label in self.classes:
                mean_vector = self.mean_vectors[class_label]
                score = self._compute_similarity(x, mean_vector)
                class_scores.append(score)
            scores.append(class_scores)
        
        return np.array(scores)


class MultiLabelChaosNet:
    """
    Multi-label ChaosNet for diseases with multiple concurrent conditions
    Uses threshold-based classification
    """
    
    def __init__(
        self,
        distance_metric: str = 'cosine',
        threshold: float = 0.5,
        adaptive_threshold: bool = True
    ):
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.mean_vectors = {}
        self.classes = None
        self.n_classes = None
        self.class_thresholds = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train multi-label ChaosNet
        
        Args:
            X: Training features (N x D)
            y: Training labels (N x n_classes) - binary multi-label
        """
        self.n_classes = y.shape[1]
        self.classes = np.arange(self.n_classes)
        
        # Compute mean representation vector for each class
        for class_idx in range(self.n_classes):
            # Samples with this label
            positive_samples = X[y[:, class_idx] == 1]
            if len(positive_samples) > 0:
                self.mean_vectors[class_idx] = np.mean(positive_samples, axis=0)
            else:
                # No positive samples for this class
                self.mean_vectors[class_idx] = np.zeros(X.shape[1])
        
        # Compute adaptive thresholds if enabled
        if self.adaptive_threshold:
            self._compute_adaptive_thresholds(X, y)
    
    def _compute_adaptive_thresholds(self, X: np.ndarray, y: np.ndarray):
        """
        Compute class-specific thresholds based on training data
        
        Args:
            X: Training features
            y: Training labels
        """
        for class_idx in range(self.n_classes):
            # Compute similarities for this class
            similarities = []
            for x in X:
                mean_vector = self.mean_vectors[class_idx]
                if self.distance_metric == 'cosine':
                    sim = cosine_similarity(
                        x.reshape(1, -1),
                        mean_vector.reshape(1, -1)
                    )[0, 0]
                else:
                    sim = -np.linalg.norm(x - mean_vector)
                similarities.append(sim)
            
            similarities = np.array(similarities)
            
            # Set threshold as mean of positive and negative similarities
            positive_sims = similarities[y[:, class_idx] == 1]
            negative_sims = similarities[y[:, class_idx] == 0]
            
            if len(positive_sims) > 0 and len(negative_sims) > 0:
                self.class_thresholds[class_idx] = (
                    np.mean(positive_sims) + np.mean(negative_sims)
                ) / 2
            else:
                self.class_thresholds[class_idx] = self.threshold
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict multi-label outputs
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Predicted labels (N x n_classes)
        """
        predictions = []
        
        for x in X:
            sample_predictions = []
            for class_idx in range(self.n_classes):
                mean_vector = self.mean_vectors[class_idx]
                
                # Compute similarity
                if self.distance_metric == 'cosine':
                    sim = cosine_similarity(
                        x.reshape(1, -1),
                        mean_vector.reshape(1, -1)
                    )[0, 0]
                else:
                    sim = -np.linalg.norm(x - mean_vector)
                
                # Apply threshold
                threshold = self.class_thresholds.get(class_idx, self.threshold)
                prediction = 1 if sim >= threshold else 0
                sample_predictions.append(prediction)
            
            predictions.append(sample_predictions)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (sigmoid of similarities)
        
        Args:
            X: Test features (N x D)
            
        Returns:
            Class probabilities (N x n_classes)
        """
        probabilities = []
        
        for x in X:
            sample_probs = []
            for class_idx in range(self.n_classes):
                mean_vector = self.mean_vectors[class_idx]
                
                # Compute similarity
                if self.distance_metric == 'cosine':
                    sim = cosine_similarity(
                        x.reshape(1, -1),
                        mean_vector.reshape(1, -1)
                    )[0, 0]
                else:
                    sim = -np.linalg.norm(x - mean_vector)
                
                # Convert to probability using sigmoid
                prob = 1 / (1 + np.exp(-sim))
                sample_probs.append(prob)
            
            probabilities.append(sample_probs)
        
        return np.array(probabilities)


if __name__ == "__main__":
    # Example usage
    print("ChaosNet Example")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    N_train = 100
    N_test = 20
    feature_dim = 400  # ChaosFEX features
    n_classes = 5
    
    X_train = np.random.randn(N_train, feature_dim)
    y_train = np.random.randint(0, n_classes, N_train)
    
    X_test = np.random.randn(N_test, feature_dim)
    
    # Train ChaosNet
    chaosnet = ChaosNet(distance_metric='cosine')
    chaosnet.fit(X_train, y_train)
    
    # Predict
    predictions = chaosnet.predict(X_test)
    probabilities = chaosnet.predict_proba(X_test)
    
    print(f"Training samples: {N_train}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {n_classes}")
    print(f"Test predictions: {predictions}")
    print(f"Test probabilities shape: {probabilities.shape}")
    
    # Multi-label example
    print("\nMulti-Label ChaosNet Example")
    print("=" * 50)
    
    y_train_multilabel = np.random.randint(0, 2, (N_train, n_classes))
    
    ml_chaosnet = MultiLabelChaosNet(adaptive_threshold=True)
    ml_chaosnet.fit(X_train, y_train_multilabel)
    
    ml_predictions = ml_chaosnet.predict(X_test)
    ml_probabilities = ml_chaosnet.predict_proba(X_test)
    
    print(f"Multi-label predictions shape: {ml_predictions.shape}")
    print(f"Sample prediction: {ml_predictions[0]}")
    print(f"Sample probabilities: {ml_probabilities[0]}")
