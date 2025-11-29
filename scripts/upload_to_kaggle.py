"""
Kaggle Dataset Setup Script
Uploads dataset to Kaggle and creates download script for portability
"""

import os
import subprocess
import sys

print("=" * 60)
print("📦 KAGGLE DATASET SETUP")
print("=" * 60)

# Check if kaggle is installed
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    print("✅ Kaggle API found")
except ImportError:
    print("❌ Kaggle not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    from kaggle.api.kaggle_api_extended import KaggleApi
    print("✅ Kaggle installed")

# Initialize API
api = KaggleApi()
api.authenticate()
print("✅ Authenticated with Kaggle")

# Dataset info
dataset_slug = "subashsss/rfmid-dataset"
dataset_path = "data"

print(f"\n📤 Uploading dataset to Kaggle...")
print(f"   Dataset: {dataset_slug}")
print(f"   Source: {dataset_path}")

try:
    # Create new dataset
    api.dataset_create_new(
        folder=dataset_path,
        dir_mode='zip',
        convert_to_csv=False,
        public=True
    )
    print("✅ Dataset uploaded successfully!")
    print(f"   View at: https://www.kaggle.com/datasets/{dataset_slug}")
    
except Exception as e:
    if "already exists" in str(e).lower():
        print("⚠️  Dataset already exists. Updating...")
        try:
            api.dataset_create_version(
                folder=dataset_path,
                version_notes="Updated dataset",
                dir_mode='zip',
                convert_to_csv=False
            )
            print("✅ Dataset updated successfully!")
        except Exception as e2:
            print(f"❌ Update failed: {e2}")
            print("\n💡 Manual steps:")
            print(f"   1. Go to https://www.kaggle.com/datasets/{dataset_slug}/settings")
            print(f"   2. Click 'New Version'")
            print(f"   3. Upload the files from {dataset_path}/")
    else:
        print(f"❌ Upload failed: {e}")
        print("\n💡 Manual upload:")
        print(f"   1. Go to https://www.kaggle.com/datasets")
        print(f"   2. Click 'New Dataset'")
        print(f"   3. Upload files from {dataset_path}/")
        print(f"   4. Set slug to: rfmid-dataset")

print("\n" + "=" * 60)
