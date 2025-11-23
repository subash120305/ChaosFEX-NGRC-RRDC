"""
ChaosFEX: Chaos-based Feature Extraction Module

Implements chaos-based feature extraction using:
1. Generalized Luroth Series (GLS) map
2. Logistic map

Extracts four key features per neuron:
- Mean Firing Time (MFT)
- Mean Firing Rate (MFR)
- Mean Energy (ME)
- Mean Entropy (MEnt)
"""

import numpy as np
from typing import Literal, Tuple, Optional
from scipy.stats import entropy


class ChaosFEX:
    """
    Chaos-based Feature Extractor
    
    Args:
        n_neurons: Number of chaotic neurons
        map_type: Type of chaotic map ('GLS' or 'Logistic')
        b: Parameter for GLS map (default: 0.1)
        r: Parameter for Logistic map (default: 3.8)
        threshold: Firing threshold (default: 0.5)
        max_iterations: Maximum iterations for chaotic dynamics (default: 1000)
    """
    
    def __init__(
        self,
        n_neurons: int = 100,
        map_type: Literal['GLS', 'Logistic', 'Hybrid'] = 'GLS',
        b: float = 0.1,
        r: float = 3.8,
        threshold: float = 0.5,
        max_iterations: int = 1000,
        random_seed: Optional[int] = None
    ):
        self.n_neurons = n_neurons
        self.map_type = map_type
        self.b = b
        self.r = r
        self.threshold = threshold
        self.max_iterations = max_iterations
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def gls_map(self, x: float, b: float) -> float:
        """
        Generalized Luroth Series (GLS) map
        
        x_{n+1} = x_n + b * x_n^2 (mod 1)
        
        Args:
            x: Current state (0 < x < 1)
            b: GLS parameter
            
        Returns:
            Next state
        """
        return (x + b * x**2) % 1.0
    
    def logistic_map(self, x: float, r: float) -> float:
        """
        Logistic map
        
        x_{n+1} = r * x_n * (1 - x_n)
        
        Args:
            x: Current state (0 < x < 1)
            r: Logistic parameter (3.6 < r < 4.0 for chaos)
            
        Returns:
            Next state
        """
        return r * x * (1 - x)
    
    def compute_firing_times(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute firing times when trajectory crosses threshold
        
        Args:
            trajectory: Chaotic trajectory
            
        Returns:
            Array of firing times
        """
        crossings = np.where(np.diff((trajectory > self.threshold).astype(int)) == 1)[0]
        return crossings
    
    def compute_features(self, trajectory: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute ChaosFEX features from trajectory
        
        Args:
            trajectory: Chaotic trajectory
            
        Returns:
            Tuple of (MFT, MFR, ME, MEnt)
        """
        # Mean Firing Time (MFT)
        firing_times = self.compute_firing_times(trajectory)
        if len(firing_times) > 1:
            mft = np.mean(np.diff(firing_times))
        else:
            mft = self.max_iterations  # No firing
        
        # Mean Firing Rate (MFR)
        mfr = len(firing_times) / self.max_iterations
        
        # Mean Energy (ME) - Average squared amplitude
        me = np.mean(trajectory**2)
        
        # Mean Entropy (MEnt) - Shannon entropy of binned trajectory
        hist, _ = np.histogram(trajectory, bins=20, range=(0, 1), density=True)
        hist = hist / np.sum(hist)  # Normalize
        ment = entropy(hist + 1e-10)  # Add small constant to avoid log(0)
        
        return mft, mfr, me, ment
    
    def generate_trajectory(self, initial_condition: float) -> np.ndarray:
        """
        Generate chaotic trajectory
        
        Args:
            initial_condition: Initial state (0 < x < 1)
            
        Returns:
            Chaotic trajectory
        """
        trajectory = np.zeros(self.max_iterations)
        trajectory[0] = initial_condition
        
        for i in range(1, self.max_iterations):
            if self.map_type == 'GLS':
                trajectory[i] = self.gls_map(trajectory[i-1], self.b)
            elif self.map_type == 'Logistic':
                trajectory[i] = self.logistic_map(trajectory[i-1], self.r)
            elif self.map_type == 'Hybrid':
                # Alternate between GLS and Logistic
                if i % 2 == 0:
                    trajectory[i] = self.gls_map(trajectory[i-1], self.b)
                else:
                    trajectory[i] = self.logistic_map(trajectory[i-1], self.r)
        
        return trajectory
    
    def map_input_to_initial_conditions(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Map input feature vector to initial conditions for chaotic neurons
        
        Args:
            input_vector: Input feature vector (D-dimensional)
            
        Returns:
            Initial conditions for n_neurons (values in (0, 1))
        """
        # Normalize input to [0, 1]
        input_norm = (input_vector - input_vector.min()) / (input_vector.max() - input_vector.min() + 1e-10)
        
        # If input dimension < n_neurons, repeat and add noise
        if len(input_norm) < self.n_neurons:
            repeats = int(np.ceil(self.n_neurons / len(input_norm)))
            input_norm = np.tile(input_norm, repeats)[:self.n_neurons]
            # Add small random noise to create diversity
            input_norm += np.random.uniform(0, 0.01, self.n_neurons)
        else:
            # If input dimension > n_neurons, sample uniformly
            indices = np.linspace(0, len(input_norm)-1, self.n_neurons, dtype=int)
            input_norm = input_norm[indices]
        
        # Ensure values are in (0, 1)
        initial_conditions = np.clip(input_norm, 0.01, 0.99)
        
        return initial_conditions
    
    def extract_features(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Extract ChaosFEX features from input vector
        
        Args:
            input_vector: Input feature vector (D-dimensional)
            
        Returns:
            ChaosFEX features (4*n_neurons dimensional)
        """
        # Map input to initial conditions
        initial_conditions = self.map_input_to_initial_conditions(input_vector)
        
        # Extract features from each neuron
        features = []
        for ic in initial_conditions:
            trajectory = self.generate_trajectory(ic)
            mft, mfr, me, ment = self.compute_features(trajectory)
            features.extend([mft, mfr, me, ment])
        
        return np.array(features)
    
    def extract_features_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Extract ChaosFEX features from batch of input vectors
        
        Args:
            input_batch: Batch of input vectors (N x D)
            
        Returns:
            ChaosFEX features (N x 4*n_neurons)
        """
        batch_features = []
        for input_vector in input_batch:
            features = self.extract_features(input_vector)
            batch_features.append(features)
        
        return np.array(batch_features)
    
    def get_feature_names(self) -> list:
        """
        Get feature names for interpretability
        
        Returns:
            List of feature names
        """
        feature_names = []
        for i in range(self.n_neurons):
            feature_names.extend([
                f'Neuron{i}_MFT',
                f'Neuron{i}_MFR',
                f'Neuron{i}_ME',
                f'Neuron{i}_MEnt'
            ])
        return feature_names


class MultiScaleChaosFEX:
    """
    Multi-scale ChaosFEX with different parameter settings
    Captures dynamics at multiple temporal scales
    """
    
    def __init__(
        self,
        n_neurons_per_scale: int = 50,
        n_scales: int = 3,
        map_type: Literal['GLS', 'Logistic'] = 'GLS'
    ):
        self.n_scales = n_scales
        self.extractors = []
        
        if map_type == 'GLS':
            # Different b values for different scales
            b_values = np.linspace(0.1, 0.3, n_scales)
            for b in b_values:
                self.extractors.append(
                    ChaosFEX(n_neurons=n_neurons_per_scale, map_type='GLS', b=b)
                )
        else:
            # Different r values for different scales
            r_values = np.linspace(3.6, 3.95, n_scales)
            for r in r_values:
                self.extractors.append(
                    ChaosFEX(n_neurons=n_neurons_per_scale, map_type='Logistic', r=r)
                )
    
    def extract_features(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale ChaosFEX features
        
        Args:
            input_vector: Input feature vector
            
        Returns:
            Concatenated multi-scale features
        """
        all_features = []
        for extractor in self.extractors:
            features = extractor.extract_features(input_vector)
            all_features.append(features)
        
        return np.concatenate(all_features)
    
    def extract_features_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale ChaosFEX features from batch
        
        Args:
            input_batch: Batch of input vectors
            
        Returns:
            Batch of multi-scale features
        """
        batch_features = []
        for input_vector in input_batch:
            features = self.extract_features(input_vector)
            batch_features.append(features)
        
        return np.array(batch_features)


if __name__ == "__main__":
    # Example usage
    print("ChaosFEX Example")
    print("=" * 50)
    
    # Create ChaosFEX extractor
    chaosfex = ChaosFEX(n_neurons=10, map_type='GLS', b=0.1)
    
    # Generate random input (simulating deep features)
    input_vector = np.random.randn(1024)
    
    # Extract features
    chaos_features = chaosfex.extract_features(input_vector)
    
    print(f"Input dimension: {len(input_vector)}")
    print(f"ChaosFEX dimension: {len(chaos_features)}")
    print(f"Feature names: {chaosfex.get_feature_names()[:8]}...")
    print(f"Sample features: {chaos_features[:8]}")
    
    # Multi-scale example
    print("\nMulti-Scale ChaosFEX Example")
    print("=" * 50)
    
    multiscale = MultiScaleChaosFEX(n_neurons_per_scale=5, n_scales=3)
    multiscale_features = multiscale.extract_features(input_vector)
    
    print(f"Multi-scale dimension: {len(multiscale_features)}")
    print(f"Sample features: {multiscale_features[:8]}")
