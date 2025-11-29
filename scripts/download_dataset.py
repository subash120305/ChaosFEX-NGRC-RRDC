"""
Download Dataset from Kaggle
Run this on any system to download the dataset automatically
"""

import os
import sys
import subprocess
from pathlib import Path

print("=" * 60)
print("📥 DOWNLOADING DATASET FROM KAGGLE")
print("=" * 60)

# Check if kaggle is installed
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Installing Kaggle API...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
    from kaggle.api.kaggle_api_extended import KaggleApi

# Dataset configuration
DATASET_SLUG = "subashsss/rfmid-dataset"
DOWNLOAD_PATH = "data"

# Check if data already exists
if Path(DOWNLOAD_PATH).exists() and len(list(Path(DOWNLOAD_PATH).glob("*"))) > 2:
    print(f"✅ Dataset already exists at {DOWNLOAD_PATH}/")
    response = input("Download again? (y/n): ").strip().lower()
    if response != 'y':
        print("Skipping download.")
        sys.exit(0)

# Initialize API
print("\n🔐 Authenticating with Kaggle...")
api = KaggleApi()
try:
    api.authenticate()
    print("✅ Authenticated")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    print("\n💡 Setup Instructions:")
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
    print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
    sys.exit(1)

# Download dataset
print(f"\n📥 Downloading dataset: {DATASET_SLUG}")
print(f"   Destination: {DOWNLOAD_PATH}/")

try:
    # Create download directory
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    
    # Download and unzip
    api.dataset_download_files(
        DATASET_SLUG,
        path=DOWNLOAD_PATH,
        unzip=True,
        quiet=False
    )
    
    print("\n✅ Dataset downloaded successfully!")
    print(f"   Location: {os.path.abspath(DOWNLOAD_PATH)}/")
    
    # Show structure
    print("\n📁 Dataset structure:")
    for item in sorted(Path(DOWNLOAD_PATH).glob("*")):
        if item.is_dir():
            count = len(list(item.rglob("*")))
            print(f"   📂 {item.name}/ ({count} items)")
        else:
            size = item.stat().st_size / (1024*1024)
            print(f"   📄 {item.name} ({size:.1f} MB)")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\n💡 Manual download:")
    print(f"   1. Go to https://www.kaggle.com/datasets/{DATASET_SLUG}")
    print(f"   2. Click 'Download'")
    print(f"   3. Extract to {DOWNLOAD_PATH}/")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 Ready to train! Run: python scripts/train_with_chaos.py")
print("=" * 60)
