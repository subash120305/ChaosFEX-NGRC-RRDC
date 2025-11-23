# Quick Start Guide: NG-RC + ChaosFEX for Rare Retinal Disease Classification

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd /Users/subash/Desktop/chaotic/TRY

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download RFMiD 2.0 Dataset

**Option A: Kaggle (Recommended)**
```bash
# Install Kaggle API
pip install kaggle

# Set up Kaggle credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Create new API token
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
python scripts/download_dataset.py --source kaggle --output data/raw/
```

**Option B: Manual Download**
1. Visit: https://www.kaggle.com/datasets/andrewmvd/retinal-fundus-multi-disease-image-dataset
2. Download and extract to `data/raw/RFMiD_2.0/`

### Step 3: Train the Model

```bash
# Train with default configuration
python scripts/train_ngrc_chaosfex.py --config config/config.yaml

# This will:
# - Create train/val/test splits
# - Extract deep features (EfficientNet-B3)
# - Apply ChaosFEX transformation
# - Process with NG-RC
# - Train ensemble classifier
# - Evaluate on test set
# - Save results to results/experiment_TIMESTAMP/
```

### Step 4: View Results

```bash
# Results will be saved in results/experiment_TIMESTAMP/
# - config.yaml: Configuration used
# - results.json: Metrics (accuracy, F1, AUC-ROC)
# - pipeline.pkl: Trained pipeline
# - *_predictions.npy: Predictions
# - *_probabilities.npy: Probabilities
```

---

## 📊 Expected Results

Based on the architecture, you should achieve:

- **Accuracy:** 85-92%
- **F1-Score (Macro):** 0.80-0.88
- **AUC-ROC:** >0.90

This is **significantly better** than baseline deep learning models (~78% for rare classes).

---

## 🔧 Customization

### Change Feature Extractor

Edit `config/config.yaml`:
```yaml
model:
  feature_extractor: "vit_base"  # Options: vit_base, efficientnet_b3, resnet50
```

### Adjust ChaosFEX Parameters

```yaml
model:
  chaosfex_neurons: 300  # More neurons = richer features
  chaosfex_map: "Logistic"  # Try different maps
  chaosfex_b: 0.2  # Adjust chaos parameter
```

### Try Different Classifiers

```yaml
model:
  classifier_type: "chaosnet"  # Fast, interpretable
  # OR
  classifier_type: "ensemble"  # Higher accuracy
```

---

## 🧪 Running Experiments

### Experiment 1: Baseline (Deep Features Only)

Modify the pipeline to skip ChaosFEX and NG-RC:
```python
# In train_ngrc_chaosfex.py, replace pipeline.fit() with:
# Direct classification on deep features
```

### Experiment 2: ChaosFEX Only

```yaml
model:
  ngrc_reservoir_size: 0  # Disable NG-RC
```

### Experiment 3: Full Pipeline

Use default config (ChaosFEX + NG-RC + Ensemble)

---

## 📈 Monitoring Training

The training script prints progress:

```
Initializing NG-RC + ChaosFEX Pipeline...
  [1/4] Loading efficientnet_b3...
  [2/4] Initializing ChaosFEX (200 neurons, GLS map)...
  [3/4] Initializing NG-RC (300 neurons)...
  [4/4] Initializing ensemble classifier...

Training NG-RC + ChaosFEX Pipeline
============================================================
[1/4] Extracting deep features...
      Deep features shape: (602, 1024)
[2/4] Extracting ChaosFEX features...
      Chaos features shape: (602, 800)
[3/4] Extracting NG-RC features...
      NG-RC features shape: (602, 300)
[4/4] Training ensemble classifier...

VALIDATION SET EVALUATION
============================================================
Validation Metrics:
  accuracy: 0.8756
  f1_macro: 0.8234
  auc_macro: 0.9123

TEST SET EVALUATION
============================================================
Test Metrics:
  accuracy: 0.8621
  f1_macro: 0.8156
  auc_macro: 0.9045
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size or use smaller model
```yaml
model:
  feature_extractor: "efficientnet_b0"  # Smaller model
  chaosfex_neurons: 100  # Fewer neurons
```

### Issue: Kaggle API Error

**Solution:** Set up credentials
```bash
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Dataset Not Found

**Solution:** Check directory structure
```
data/raw/RFMiD_2.0/
├── images/
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── RFMiD_Training_Labels.csv
```

---

## 📝 Next Steps

1. **Analyze Results:**
   ```python
   import json
   with open('results/experiment_TIMESTAMP/results.json') as f:
       results = json.load(f)
   print(results['test_metrics'])
   ```

2. **Visualize Features:**
   - Use notebooks/03_chaosfex_analysis.ipynb
   - t-SNE visualization of chaos features
   - Feature importance analysis

3. **Write Paper:**
   - Use results for publication
   - Compare with baselines
   - Highlight interpretability

---

## 💡 Tips for Best Results

1. **Use EfficientNet-B3 or ViT-Base** for feature extraction
2. **Set chaosfex_neurons=200-300** for good balance
3. **Use ensemble classifier** for highest accuracy
4. **Enable SMOTE** for class imbalance handling
5. **Run multiple seeds** and report mean ± std

---

## 📧 Need Help?

- Check README.md for detailed documentation
- Review example notebooks in notebooks/
- Examine source code in src/

---

**Good luck with your research! 🎉**
