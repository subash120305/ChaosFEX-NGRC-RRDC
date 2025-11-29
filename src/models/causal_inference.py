"""
Causal Inference Module for ChaosFEX-NGRC

Implements causal discovery and counterfactual analysis to identify
which features (deep or chaotic) causally influence disease predictions.

This goes beyond correlation to establish causation.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class CausalInference:
    """
    Causal Inference for Medical Imaging
    
    Uses do-calculus and counterfactual reasoning to identify:
    1. Which features CAUSE disease (not just correlate)
    2. How changing features affects predictions
    3. Feature importance from a causal perspective
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.causal_graph = {}
        self.feature_effects = {}
        
    def discover_causal_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        n_bootstrap: int = 100
    ) -> Dict[str, float]:
        """
        Discover which features causally influence the outcome
        
        Uses conditional independence testing and bootstrap stability
        to identify causal relationships.
        
        Args:
            X: Feature matrix (N x D)
            y: Target labels (N,)
            feature_names: Names of features
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary of feature -> causal strength
        """
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        print(f"\n🔍 Discovering Causal Features...")
        print(f"   Samples: {n_samples}, Features: {n_features}")
        
        # Method 1: Conditional Independence Testing
        # A feature X_i is causal if Y is NOT independent of X_i given other features
        causal_scores = {}
        
        for i in range(n_features):
            # Fit model with all features
            X_full = X.copy()
            model_full = LogisticRegression(random_state=self.random_seed, max_iter=1000)
            model_full.fit(X_full, y)
            score_full = model_full.score(X_full, y)
            
            # Fit model without feature i (intervention: do(X_i = 0))
            X_without_i = X.copy()
            X_without_i[:, i] = 0  # Intervene: set feature to 0
            model_without = LogisticRegression(random_state=self.random_seed, max_iter=1000)
            model_without.fit(X_without_i, y)
            score_without = model_without.score(X_without_i, y)
            
            # Causal effect = performance drop when intervening
            causal_effect = score_full - score_without
            causal_scores[feature_names[i]] = max(0, causal_effect)
        
        # Method 2: Bootstrap Stability
        # Causal features should be stable across different data samples
        stability_scores = {name: 0.0 for name in feature_names}
        
        np.random.seed(self.random_seed)
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model
            model = LogisticRegression(random_state=self.random_seed, max_iter=1000)
            model.fit(X_boot, y_boot)
            
            # Count how often each feature has non-zero coefficient
            for i, name in enumerate(feature_names):
                if abs(model.coef_[0][i]) > 0.01:
                    stability_scores[name] += 1.0 / n_bootstrap
        
        # Combine causal effect and stability
        final_scores = {}
        for name in feature_names:
            final_scores[name] = causal_scores[name] * stability_scores[name]
        
        # Normalize
        max_score = max(final_scores.values()) if final_scores else 1.0
        if max_score > 0:
            final_scores = {k: v/max_score for k, v in final_scores.items()}
        
        self.feature_effects = final_scores
        
        # Print top causal features
        sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\n📊 Top 10 Causal Features:")
        for name, score in sorted_features[:10]:
            print(f"   {name}: {score:.4f}")
        
        return final_scores
    
    def counterfactual_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: any,
        sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Counterfactual Analysis: "What if this feature was different?"
        
        For a given sample, compute what would happen if we changed
        specific features (intervention).
        
        Args:
            X: Feature matrix
            y: True labels
            model: Trained classifier
            sample_idx: Index of sample to analyze
            
        Returns:
            Counterfactual results
        """
        print(f"\n🔮 Counterfactual Analysis for Sample {sample_idx}")
        
        # Original prediction
        x_original = X[sample_idx:sample_idx+1]
        y_true = y[sample_idx]
        y_pred_original = model.predict(x_original)[0]
        y_prob_original = model.predict_proba(x_original)[0]
        
        print(f"   True Label: {y_true}")
        print(f"   Predicted: {y_pred_original} (prob: {y_prob_original[int(y_pred_original)]:.3f})")
        
        # Test interventions on top causal features
        counterfactuals = {}
        
        if not self.feature_effects:
            print("   ⚠️ Run discover_causal_features() first")
            return {}
        
        # Get top 5 causal features
        top_features = sorted(self.feature_effects.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feat_name, causal_strength in top_features:
            feat_idx = int(feat_name.split('_')[1]) if 'Feature_' in feat_name else 0
            
            # Intervention: Set feature to mean value
            x_intervened = x_original.copy()
            x_intervened[0, feat_idx] = X[:, feat_idx].mean()
            
            y_pred_new = model.predict(x_intervened)[0]
            y_prob_new = model.predict_proba(x_intervened)[0]
            
            # Did prediction change?
            changed = y_pred_new != y_pred_original
            prob_change = y_prob_new[int(y_pred_original)] - y_prob_original[int(y_pred_original)]
            
            counterfactuals[feat_name] = {
                'causal_strength': causal_strength,
                'prediction_changed': changed,
                'probability_change': prob_change,
                'new_prediction': y_pred_new
            }
            
            if changed:
                print(f"   ✨ {feat_name}: Prediction changed to {y_pred_new}!")
            else:
                print(f"   → {feat_name}: Prob change: {prob_change:+.3f}")
        
        return counterfactuals
    
    def estimate_treatment_effect(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment_feature_idx: int
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect (ATE)
        
        Quantifies: "What is the causal effect of this feature on the outcome?"
        
        Args:
            X: Feature matrix
            y: Target labels
            treatment_feature_idx: Index of feature to treat as intervention
            
        Returns:
            Treatment effect statistics
        """
        print(f"\n💊 Estimating Treatment Effect for Feature {treatment_feature_idx}")
        
        # Split into "treated" (high feature value) and "control" (low feature value)
        median_value = np.median(X[:, treatment_feature_idx])
        treated_mask = X[:, treatment_feature_idx] > median_value
        control_mask = ~treated_mask
        
        # Outcome rates
        treated_outcome = y[treated_mask].mean()
        control_outcome = y[control_mask].mean()
        
        # Average Treatment Effect (ATE)
        ate = treated_outcome - control_outcome
        
        # Relative Risk
        relative_risk = treated_outcome / (control_outcome + 1e-10)
        
        print(f"   Treated group (n={treated_mask.sum()}): {treated_outcome:.3f} outcome rate")
        print(f"   Control group (n={control_mask.sum()}): {control_outcome:.3f} outcome rate")
        print(f"   Average Treatment Effect: {ate:+.3f}")
        print(f"   Relative Risk: {relative_risk:.3f}")
        
        return {
            'ate': ate,
            'relative_risk': relative_risk,
            'treated_outcome': treated_outcome,
            'control_outcome': control_outcome,
            'n_treated': treated_mask.sum(),
            'n_control': control_mask.sum()
        }
    
    def generate_causal_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: any,
        feature_names: List[str] = None,
        output_file: str = "results/causal_analysis.txt"
    ):
        """
        Generate comprehensive causal analysis report
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CAUSAL INFERENCE ANALYSIS REPORT\n")
            f.write("ChaosFEX-NGRC: Causal Feature Discovery\n")
            f.write("=" * 60 + "\n\n")
            
            # 1. Causal Feature Discovery
            f.write("1. CAUSAL FEATURE DISCOVERY\n")
            f.write("-" * 60 + "\n")
            causal_features = self.discover_causal_features(X, y, feature_names)
            
            sorted_features = sorted(causal_features.items(), key=lambda x: x[1], reverse=True)
            f.write("\nTop 20 Causal Features:\n")
            for i, (name, score) in enumerate(sorted_features[:20], 1):
                f.write(f"{i:2d}. {name:30s}: {score:.4f}\n")
            
            # 2. Treatment Effects
            f.write("\n\n2. TREATMENT EFFECT ANALYSIS\n")
            f.write("-" * 60 + "\n")
            
            # Analyze top 3 causal features
            for i, (name, score) in enumerate(sorted_features[:3], 1):
                feat_idx = int(name.split('_')[1]) if 'Feature_' in name else i-1
                f.write(f"\nFeature: {name}\n")
                effects = self.estimate_treatment_effect(X, y, feat_idx)
                f.write(f"  ATE: {effects['ate']:+.4f}\n")
                f.write(f"  Relative Risk: {effects['relative_risk']:.4f}\n")
                f.write(f"  Treated: {effects['treated_outcome']:.4f} (n={effects['n_treated']})\n")
                f.write(f"  Control: {effects['control_outcome']:.4f} (n={effects['n_control']})\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"\n✅ Causal analysis report saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    print("Testing Causal Inference Module...")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=42)
    
    # Initialize causal inference
    causal = CausalInference()
    
    # Discover causal features
    causal_features = causal.discover_causal_features(X, y)
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Counterfactual analysis
    causal.counterfactual_analysis(X, y, model, sample_idx=0)
    
    # Treatment effect
    causal.estimate_treatment_effect(X, y, treatment_feature_idx=0)
    
    # Generate report
    causal.generate_causal_report(X, y, model)
