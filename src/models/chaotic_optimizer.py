"""
Chaotic Hyperparameter Optimizer

Implements chaotic optimization algorithms (Logistic Map, Tent Map) 
to optimize hyperparameters of the ChaosFEX-NGRC pipeline.

This fulfills the "Chaotic Training" requirement by using chaos theory
to guide the learning process (hyperparameter search).
"""

import numpy as np
from typing import Dict, Any, Callable, List, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import copy

class ChaoticOptimizer:
    """
    Chaotic Optimizer for Hyperparameter Tuning
    
    Uses chaotic maps to explore the hyperparameter space more efficiently
    than random search, avoiding local optima due to the ergodic property of chaos.
    
    Args:
        map_type: Type of chaotic map ('logistic', 'tent', 'sine')
        r: Control parameter for the map (default: 4.0 for full chaos)
        max_iterations: Number of optimization steps
    """
    
    def __init__(
        self,
        map_type: str = 'logistic',
        r: float = 4.0,
        max_iterations: int = 50,
        random_seed: int = 42
    ):
        self.map_type = map_type
        self.r = r
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.history = []
        
    def _logistic_map(self, x: float) -> float:
        """Logistic Map: x_{n+1} = r * x_n * (1 - x_n)"""
        return self.r * x * (1 - x)
    
    def _tent_map(self, x: float) -> float:
        """Tent Map"""
        if x < 0.5:
            return self.r * x
        else:
            return self.r * (1 - x)
            
    def _sine_map(self, x: float) -> float:
        """Sine Map"""
        return self.r / 4.0 * np.sin(np.pi * x)
        
    def _next_state(self, x: float) -> float:
        """Get next chaotic state"""
        if self.map_type == 'logistic':
            return self._logistic_map(x)
        elif self.map_type == 'tent':
            return self._tent_map(x)
        elif self.map_type == 'sine':
            return self._sine_map(x)
        else:
            raise ValueError(f"Unknown map type: {self.map_type}")

    def _map_to_range(self, x: float, param_range: Tuple[Any, Any], param_type: type) -> Any:
        """Map chaotic variable x (0,1) to parameter range"""
        min_val, max_val = param_range
        
        # Linear mapping
        scaled_val = min_val + x * (max_val - min_val)
        
        if param_type == int:
            return int(round(scaled_val))
        elif param_type == float:
            return float(scaled_val)
        else:
            return scaled_val

    def optimize(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, Tuple[Any, Any]],
        scoring: str = 'accuracy',
        cv: int = 3
    ) -> Dict[str, Any]:
        """
        Run chaotic optimization
        
        Args:
            estimator: Sklearn-compatible estimator
            X: Training features
            y: Training labels
            param_grid: Dictionary of parameter ranges {name: (min, max)}
            scoring: Scoring metric
            cv: Cross-validation folds
            
        Returns:
            Best parameters found
        """
        np.random.seed(self.random_seed)
        
        # Initialize chaotic variables for each parameter
        # We use slightly different initial conditions for each parameter to ensure diversity
        chaotic_vars = {
            param: np.random.uniform(0.1, 0.9) 
            for param in param_grid.keys()
        }
        
        best_score = -np.inf
        best_params = {}
        
        print(f"Starting Chaotic Optimization ({self.map_type} map)...")
        print(f"Searching space: {param_grid}")
        
        for i in range(self.max_iterations):
            current_params = {}
            
            # 1. Update chaotic variables and map to parameters
            for param, (min_val, max_val) in param_grid.items():
                # Get parameter type from the range values
                param_type = type(min_val)
                
                # Update chaotic state
                chaotic_vars[param] = self._next_state(chaotic_vars[param])
                
                # Map to physical parameter value
                current_params[param] = self._map_to_range(
                    chaotic_vars[param], 
                    (min_val, max_val), 
                    param_type
                )
            
            # 2. Evaluate candidate
            try:
                # Clone estimator and set parameters
                current_estimator = clone(estimator)
                current_estimator.set_params(**current_params)
                
                # Cross-validation
                scores = cross_val_score(current_estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = np.mean(scores)
                
                # Update best
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = current_params.copy()
                    print(f"Iter {i+1}: New Best Score = {best_score:.4f} | Params: {best_params}")
                
                # Store history
                self.history.append({
                    'iteration': i,
                    'params': current_params,
                    'score': mean_score
                })
                
            except Exception as e:
                print(f"Iter {i+1}: Failed with params {current_params}. Error: {e}")
                
        print(f"\nOptimization Complete!")
        print(f"Best Score: {best_score:.4f}")
        print(f"Best Params: {best_params}")
        
        return best_params

if __name__ == "__main__":
    # Test with a simple classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("Testing Chaotic Optimizer...")
    X, y = make_classification(n_samples=200, n_features=20, random_state=42)
    
    optimizer = ChaoticOptimizer(max_iterations=10)
    
    # Define search space
    param_grid = {
        'n_estimators': (10, 200),
        'max_depth': (2, 20),
        'min_samples_split': (2, 10)
    }
    
    rf = RandomForestClassifier(random_state=42)
    best_params = optimizer.optimize(rf, X, y, param_grid)
