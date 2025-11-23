"""
Download RFMiD 2.0 Dataset

Downloads the Retinal Fundus Multi-Disease Image Dataset from Kaggle or Zenodo.

Usage:
    python scripts/download_dataset.py --source kaggle --output data/raw/
"""

import argparse
import os
import sys
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm


def download_from_kaggle(output_dir: str):
    """
    Download RFMiD dataset from Kaggle
    
    Prerequisites:
        1. Install kaggle: pip install kaggle
        2. Set up Kaggle API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Install with: pip install kaggle")
        sys.exit(1)
    
    print("Downloading RFMiD 2.0 from Kaggle...")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    dataset_name = "andrewmvd/retinal-fundus-multi-disease-image-dataset"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading to {output_path}...")
    api.dataset_download_files(
        dataset_name,
        path=str(output_path),
        unzip=True
    )
    
    print("Download completed!")
    print(f"Dataset saved to: {output_path}")


def download_from_zenodo(output_dir: str, record_id: str = "6524199"):
    """
    Download RFMiD dataset from Zenodo
    
    Args:
        output_dir: Output directory
        record_id: Zenodo record ID (default: 6524199 for RFMiD 2.0)
    """
    print(f"Downloading RFMiD 2.0 from Zenodo (Record: {record_id})...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get record metadata
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Could not fetch record {record_id}")
        sys.exit(1)
    
    record = response.json()
    files = record['files']
    
    print(f"Found {len(files)} files")
    
    # Download each file
    for file_info in files:
        filename = file_info['key']
        file_url = file_info['links']['self']
        file_size = file_info['size']
        
        print(f"\nDownloading {filename} ({file_size / 1024 / 1024:.2f} MB)...")
        
        output_file = output_path / filename
        
        # Download with progress bar
        response = requests.get(file_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract if zip file
        if filename.endswith('.zip'):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            
            # Remove zip file
            output_file.unlink()
            print(f"Extracted and removed {filename}")
    
    print("\nDownload completed!")
    print(f"Dataset saved to: {output_path}")


def download_from_url(url: str, output_dir: str):
    """
    Download dataset from direct URL
    
    Args:
        url: Direct download URL
        output_dir: Output directory
    """
    print(f"Downloading from {url}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = url.split('/')[-1]
    output_file = output_path / filename
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_file, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Extract if zip file
    if filename.endswith('.zip'):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        output_file.unlink()
        print(f"Extracted and removed {filename}")
    
    print("Download completed!")
    print(f"Dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download RFMiD 2.0 Dataset')
    parser.add_argument('--source', type=str, default='kaggle',
                        choices=['kaggle', 'zenodo', 'url'],
                        help='Download source')
    parser.add_argument('--output', type=str, default='data/raw/',
                        help='Output directory')
    parser.add_argument('--url', type=str, default=None,
                        help='Direct download URL (if source=url)')
    parser.add_argument('--zenodo_record', type=str, default='6524199',
                        help='Zenodo record ID (if source=zenodo)')
    
    args = parser.parse_args()
    
    if args.source == 'kaggle':
        download_from_kaggle(args.output)
    elif args.source == 'zenodo':
        download_from_zenodo(args.output, args.zenodo_record)
    elif args.source == 'url':
        if args.url is None:
            print("Error: --url required when source=url")
            sys.exit(1)
        download_from_url(args.url, args.output)
    
    print("\n" + "="*60)
    print("DATASET DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify the dataset structure:")
    print(f"   ls {args.output}")
    print("2. Run preprocessing:")
    print("   python scripts/preprocess_data.py")
    print("3. Train the model:")
    print("   python scripts/train_ngrc_chaosfex.py")


if __name__ == "__main__":
    main()
