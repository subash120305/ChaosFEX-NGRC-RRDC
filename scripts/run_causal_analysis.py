"""
Train with Causal Inference Analysis
Extends the chaotic training with causal feature discovery
"""

import numpy as np
import joblib
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.causal_inference import CausalInference

print("=" * 60)
print("🔍 CAUSAL INFERENCE ANALYSIS")
print("=" * 60)

# Load cached features from previous training
print("\nLoading features...")
X_train = np.load("results/cache/train_X.npy")
y_train_multi = np.load("results/cache/train_y_multi.npy")
X_val = np.load("results/cache/val_X.npy")
y_val_multi = np.load("results/cache/val_y_multi.npy")

# Extract DR target (index 1)
y_train = y_train_multi[:, 1]
y_val = y_val_multi[:, 1]

print(f"Train: {len(y_train)} samples, {X_train.shape[1]} features")
print(f"Val: {len(y_val)} samples")

# Load trained model
pipeline_data = joblib.load("results/pipeline.pkl")
model = pipeline_data['classifier']

print(f"\nModel type: {type(model).__name__}")

# Initialize causal inference
causal = CausalInference(random_seed=42)

# Feature names (800 ChaosFEX features)
feature_names = []
# First 200 features from each of 4 ChaosFEX metrics
for metric in ['MFT', 'MFR', 'ME', 'MEnt']:
    for i in range(200):
        feature_names.append(f"ChaosFEX_{metric}_{i}")

print(f"\nAnalyzing {len(feature_names)} chaotic features...")

# 1. Discover Causal Features
print("\n" + "=" * 60)
print("STEP 1: CAUSAL FEATURE DISCOVERY")
print("=" * 60)

causal_features = causal.discover_causal_features(
    X_train, 
    y_train, 
    feature_names=feature_names,
    n_bootstrap=50
)

# 2. Counterfactual Analysis on validation samples
print("\n" + "=" * 60)
print("STEP 2: COUNTERFACTUAL ANALYSIS")
print("=" * 60)

# Analyze a few interesting samples
interesting_samples = [0, 10, 50, 100]
for idx in interesting_samples:
    if idx < len(X_val):
        causal.counterfactual_analysis(X_val, y_val, model, sample_idx=idx)

# 3. Treatment Effect Estimation
print("\n" + "=" * 60)
print("STEP 3: TREATMENT EFFECT ESTIMATION")
print("=" * 60)

# Get top 5 causal features
top_causal = sorted(causal_features.items(), key=lambda x: x[1], reverse=True)[:5]

treatment_effects = {}
for feat_name, causal_strength in top_causal:
    feat_idx = feature_names.index(feat_name)
    effects = causal.estimate_treatment_effect(X_train, y_train, feat_idx)
    treatment_effects[feat_name] = effects

# 4. Generate Comprehensive Report
print("\n" + "=" * 60)
print("STEP 4: GENERATING CAUSAL REPORT")
print("=" * 60)

causal.generate_causal_report(
    X_train,
    y_train,
    model,
    feature_names=feature_names,
    output_file="results/causal_analysis.txt"
)

# 5. Save causal analysis results
causal_results = {
    'causal_features': causal_features,
    'treatment_effects': treatment_effects,
    'top_causal_features': [name for name, _ in top_causal]
}

joblib.dump(causal_results, "results/causal_results.pkl")
print("\n✅ Causal results saved to results/causal_results.pkl")

# 6. Create visualization
print("\n" + "=" * 60)
print("STEP 5: VISUALIZING CAUSAL EFFECTS")
print("=" * 60)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot top causal features
top_20 = sorted(causal_features.items(), key=lambda x: x[1], reverse=True)[:20]
names = [name.replace('ChaosFEX_', '') for name, _ in top_20]
scores = [score for _, score in top_20]

plt.figure(figsize=(12, 8))
colors = ['#e74c3c' if 'MFT' in n else '#3498db' if 'MFR' in n else '#2ecc71' if 'ME' in n else '#f39c12' for n in names]
plt.barh(range(len(names)), scores, color=colors, alpha=0.7)
plt.yticks(range(len(names)), names, fontsize=9)
plt.xlabel('Causal Strength', fontsize=12)
plt.title('Top 20 Causal Features for DR Detection', fontsize=14, pad=15)
plt.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.7, label='Mean Firing Time (MFT)'),
    Patch(facecolor='#3498db', alpha=0.7, label='Mean Firing Rate (MFR)'),
    Patch(facecolor='#2ecc71', alpha=0.7, label='Mean Energy (ME)'),
    Patch(facecolor='#f39c12', alpha=0.7, label='Mean Entropy (MEnt)')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig("results/plots/causal_features.png", dpi=150, bbox_inches='tight')
plt.close()

print("✅ Causal feature plot saved to results/plots/causal_features.png")

# Summary
print("\n" + "=" * 60)
print("🎉 CAUSAL ANALYSIS COMPLETE!")
print("=" * 60)
print("\n📊 Key Findings:")
print(f"   • Identified {len([s for s in causal_features.values() if s > 0.5])} strongly causal features")
print(f"   • Top causal feature: {top_causal[0][0]} (strength: {top_causal[0][1]:.4f})")
print(f"   • Most causal metric: {max(set([n.split('_')[1] for n, s in top_20]), key=lambda x: sum(s for n, s in top_20 if x in n))}")
print("\n📄 Reports generated:")
print("   • results/causal_analysis.txt")
print("   • results/causal_results.pkl")
print("   • results/plots/causal_features.png")
print("\n" + "=" * 60)
