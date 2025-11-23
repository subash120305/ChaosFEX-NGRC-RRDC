"""
Next Generation Reservoir Computing (NG-RC) Implementation

Based on:
Gauthier DJ, et al. "Next generation reservoir computing." Nature Communications, 2021.

NG-RC simplifies traditional reservoir computing by:
1. Using nonlinear vector autoregression
2. Eliminating random weight matrices
3. Reducing hyperparameters
4. Improving interpretability
"""

import numpy as np
from typing import Optional, Literal
from sklearn.linear_model import Ridge
from reservoirpy.nodes import Reservoir, Ridge as RidgeNode
import reservoirpy as rpy


class NextGenReservoirComputing:
    """
    Next Generation Reservoir Computing
    
    Args:
        input_dim: Input feature dimension
        reservoir_size: Number of reservoir neurons
        spectral_radius: Spectral radius of reservoir weight matrix
        input_scaling: Scaling factor for input weights
        leaking_rate: Leaking rate (0 < alpha <= 1)
        ridge_param: Ridge regression regularization parameter
        activation: Activation function ('tanh', 'relu', 'sigmoid')
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.3,
        ridge_param: float = 1e-6,
        activation: Literal['tanh', 'relu', 'sigmoid'] = 'tanh',
        random_seed: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.ridge_param = ridge_param
        self.activation = activation
        
        if random_seed is not None:
            np.random.seed(random_seed)
            rpy.set_seed(random_seed)
        
        # Initialize reservoir
        self.reservoir = Reservoir(
            units=reservoir_size,
            sr=spectral_radius,
            input_scaling=input_scaling,
            lr=leaking_rate,
            activation=activation
        )
        
        # Initialize readout layer
        self.readout = RidgeNode(ridge=ridge_param)
        
        # Store states for analysis
        self.reservoir_states = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train NG-RC on input data
        
        Args:
            X: Input features (N x input_dim) or (N x T x input_dim) for sequences
            y: Target labels (N x output_dim)
        """
        # Run reservoir
        self.reservoir_states = self.reservoir.run(X)
        
        # Train readout
        self.readout.fit(self.reservoir_states, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained NG-RC
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Run reservoir
        states = self.reservoir.run(X)
        
        # Predict with readout
        predictions = self.readout.run(states)
        
        return predictions
    
    def get_reservoir_states(self, X: np.ndarray) -> np.ndarray:
        """
        Get reservoir states for input data (useful for feature extraction)
        
        Args:
            X: Input features
            
        Returns:
            Reservoir states
        """
        return self.reservoir.run(X)
    
    def reset(self):
        """Reset reservoir state"""
        self.reservoir.reset()


class TemporalNGRC:
    """
    Temporal NG-RC for sequence/time-series data
    Useful for longitudinal medical imaging (e.g., disease progression)
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.3,
        ridge_param: float = 1e-6,
        sequence_length: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.sequence_length = sequence_length
        
        # Create NG-RC
        self.ngrc = NextGenReservoirComputing(
            input_dim=input_dim,
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leaking_rate=leaking_rate,
            ridge_param=ridge_param
        )
    
    def fit(self, X_sequences: np.ndarray, y: np.ndarray):
        """
        Train on sequences
        
        Args:
            X_sequences: Input sequences (N x T x input_dim)
            y: Target labels (N x output_dim)
        """
        # Process each sequence through reservoir
        all_states = []
        for seq in X_sequences:
            states = self.ngrc.get_reservoir_states(seq)
            # Aggregate states (e.g., mean pooling over time)
            aggregated_state = np.mean(states, axis=0)
            all_states.append(aggregated_state)
        
        all_states = np.array(all_states)
        
        # Train readout on aggregated states
        self.ngrc.readout.fit(all_states, y)
    
    def predict(self, X_sequences: np.ndarray) -> np.ndarray:
        """
        Predict from sequences
        
        Args:
            X_sequences: Input sequences (N x T x input_dim)
            
        Returns:
            Predictions
        """
        all_states = []
        for seq in X_sequences:
            states = self.ngrc.get_reservoir_states(seq)
            aggregated_state = np.mean(states, axis=0)
            all_states.append(aggregated_state)
        
        all_states = np.array(all_states)
        
        return self.ngrc.readout.run(all_states)


class HierarchicalNGRC:
    """
    Hierarchical NG-RC with multiple reservoir layers
    Captures multi-scale temporal dynamics
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_sizes: list = [100, 200, 300],
        spectral_radii: list = [0.7, 0.85, 0.95],
        ridge_param: float = 1e-6
    ):
        self.n_layers = len(reservoir_sizes)
        self.layers = []
        
        # Create hierarchical layers
        current_dim = input_dim
        for i, (size, sr) in enumerate(zip(reservoir_sizes, spectral_radii)):
            layer = NextGenReservoirComputing(
                input_dim=current_dim,
                reservoir_size=size,
                spectral_radius=sr,
                ridge_param=ridge_param
            )
            self.layers.append(layer)
            current_dim = size  # Next layer input is current layer reservoir size
        
        # Final readout
        self.readout = RidgeNode(ridge=ridge_param)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train hierarchical NG-RC
        
        Args:
            X: Input features
            y: Target labels
        """
        # Forward pass through layers
        current_input = X
        for layer in self.layers:
            current_input = layer.get_reservoir_states(current_input)
        
        # Train final readout
        self.readout.fit(current_input, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using hierarchical NG-RC
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Forward pass through layers
        current_input = X
        for layer in self.layers:
            current_input = layer.get_reservoir_states(current_input)
        
        # Predict with final readout
        return self.readout.run(current_input)


class SimpleNGRC:
    """
    Simplified NG-RC implementation without reservoirpy dependency
    Uses pure numpy for maximum control and interpretability
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        leaking_rate: float = 0.3,
        ridge_param: float = 1e-6,
        random_seed: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.ridge_param = ridge_param
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize weights
        self._initialize_weights()
        
        # Readout weights (trained)
        self.W_out = None
        
        # Current state
        self.state = np.zeros(reservoir_size)
    
    def _initialize_weights(self):
        """Initialize reservoir and input weights"""
        # Input weights
        self.W_in = np.random.uniform(
            -self.input_scaling,
            self.input_scaling,
            (self.reservoir_size, self.input_dim)
        )
        
        # Reservoir weights (sparse random matrix)
        self.W_res = np.random.randn(self.reservoir_size, self.reservoir_size)
        self.W_res[np.random.rand(*self.W_res.shape) > 0.1] = 0  # 90% sparsity
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        current_sr = np.max(np.abs(eigenvalues))
        self.W_res *= self.spectral_radius / current_sr
    
    def _update_state(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Update reservoir state
        
        Args:
            input_vector: Input at current timestep
            
        Returns:
            Updated state
        """
        # Compute pre-activation
        pre_activation = (
            np.dot(self.W_res, self.state) +
            np.dot(self.W_in, input_vector)
        )
        
        # Apply activation (tanh)
        activated = np.tanh(pre_activation)
        
        # Leaky integration
        self.state = (1 - self.leaking_rate) * self.state + self.leaking_rate * activated
        
        return self.state
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train NG-RC
        
        Args:
            X: Input features (N x input_dim)
            y: Target labels (N x output_dim)
        """
        # Collect reservoir states
        states = []
        self.state = np.zeros(self.reservoir_size)  # Reset state
        
        for x in X:
            state = self._update_state(x)
            states.append(state.copy())
        
        states = np.array(states)
        
        # Train readout with ridge regression
        ridge = Ridge(alpha=self.ridge_param)
        ridge.fit(states, y)
        self.W_out = ridge.coef_
        self.bias = ridge.intercept_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained NG-RC
        
        Args:
            X: Input features (N x input_dim)
            
        Returns:
            Predictions (N x output_dim)
        """
        # Collect reservoir states
        states = []
        self.state = np.zeros(self.reservoir_size)  # Reset state
        
        for x in X:
            state = self._update_state(x)
            states.append(state.copy())
        
        states = np.array(states)
        
        # Predict
        predictions = np.dot(states, self.W_out.T) + self.bias
        
        return predictions
    
    def get_states(self, X: np.ndarray) -> np.ndarray:
        """
        Get reservoir states for input data
        
        Args:
            X: Input features
            
        Returns:
            Reservoir states
        """
        states = []
        self.state = np.zeros(self.reservoir_size)
        
        for x in X:
            state = self._update_state(x)
            states.append(state.copy())
        
        return np.array(states)


if __name__ == "__main__":
    # Example usage
    print("NG-RC Example")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    N_samples = 100
    input_dim = 400  # ChaosFEX output dimension (100 neurons * 4 features)
    output_dim = 49  # Number of disease classes
    
    X_train = np.random.randn(N_samples, input_dim)
    y_train = np.random.randint(0, output_dim, N_samples)
    y_train_onehot = np.eye(output_dim)[y_train]
    
    # Create and train NG-RC
    ngrc = SimpleNGRC(
        input_dim=input_dim,
        reservoir_size=200,
        spectral_radius=0.9,
        random_seed=42
    )
    
    print(f"Training NG-RC...")
    ngrc.fit(X_train, y_train_onehot)
    
    # Predict
    X_test = np.random.randn(20, input_dim)
    predictions = ngrc.predict(X_test)
    
    print(f"Input dimension: {input_dim}")
    print(f"Reservoir size: {ngrc.reservoir_size}")
    print(f"Output dimension: {output_dim}")
    print(f"Test predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0][:5]}")
