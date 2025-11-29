"""
RFMiD 2.0 Dataset Loader

Loads and preprocesses the Retinal Fundus Multi-Disease Image Dataset
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RFMiDDataset(Dataset):
    """
    RFMiD 2.0 Dataset
    
    Args:
        data_dir: Path to dataset directory
        split: Dataset split ('train', 'val', 'test')
        image_size: Size to resize images to
        transform: Albumentations transform
        multi_label: Whether to return multi-label targets
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 224,
        transform: Optional[A.Compose] = None,
        multi_label: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.multi_label = multi_label
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Get disease names
        self.disease_names = self._get_disease_names()
        self.n_classes = len(self.disease_names)
        
        # Set transform
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
    
    def _load_annotations(self) -> pd.DataFrame:
        """Load annotation CSV file"""
        csv_path = self.data_dir / f"{self.split}_labels.csv"
        
        if not csv_path.exists():
            # Try alternative naming
            csv_path = self.data_dir / "RFMiD_Training_Labels.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Annotations not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        return df
    
    def _get_disease_names(self) -> List[str]:
        """Get list of disease names from annotations"""
        # Exclude ID and image name columns
        exclude_cols = ['ID', 'Image', 'image_id', 'filename']
        disease_cols = [col for col in self.annotations.columns if col not in exclude_cols]
        return disease_cols
    
    def _get_default_transform(self) -> A.Compose:
        """Get default augmentation transform"""
        if self.split == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get item by index
        
        Returns:
            image: Preprocessed image
            label: Disease labels (multi-label or single-label)
        """
        # Get annotation row
        row = self.annotations.iloc[idx]
        
        # Load image
        # Try different column names for image ID
        if 'Image' in row:
            image_name = row['Image']
        elif 'filename' in row:
            image_name = row['filename']
        elif 'image_id' in row:
            image_name = row['image_id']
        elif 'ID' in row:
            image_name = str(row['ID'])
        else:
            # Fallback: assume first column is ID if it's an integer
            image_name = str(row.iloc[0])
        image_path = self.data_dir / 'images' / f"{image_name}"
        
        if not image_path.exists():
            # Try with .png extension
            image_path = self.data_dir / 'images' / f"{image_name}.png"
        if not image_path.exists():
            # Try with .jpg extension
            image_path = self.data_dir / 'images' / f"{image_name}.jpg"
        
        image = cv2.imread(str(image_path))
        if image is None:
            # Try finding the file with a case-insensitive search if direct load fails
            found = False
            if not image_path.exists():
                parent = image_path.parent
                if parent.exists():
                    for f in parent.iterdir():
                        if f.name.lower() == image_path.name.lower():
                            image_path = f
                            found = True
                            break
            
            if found:
                image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Warning: Could not load image at {image_path}")
                # Return a black image as placeholder to avoid crashing
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get labels
        labels = row[self.disease_names].values.astype(np.float32)
        
        if not self.multi_label:
            # Convert to single-label (take first positive class)
            positive_classes = np.where(labels == 1)[0]
            if len(positive_classes) > 0:
                label = positive_classes[0]
            else:
                label = 0  # Default to first class if no positive
            labels = label
        
        return image, labels
    
    def get_class_weights(self) -> np.ndarray:
        """
        Compute class weights for imbalanced dataset
        
        Returns:
            Class weights (n_classes,)
        """
        if self.multi_label:
            # For multi-label, compute per-class weights
            label_counts = self.annotations[self.disease_names].sum(axis=0).values
            total_samples = len(self.annotations)
            weights = total_samples / (self.n_classes * label_counts + 1e-6)
        else:
            # For single-label
            from sklearn.utils.class_weight import compute_class_weight
            labels = []
            for idx in range(len(self)):
                _, label = self[idx]
                labels.append(label)
            labels = np.array(labels)
            weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        
        return weights
    
    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        stats = {
            'n_samples': len(self),
            'n_classes': self.n_classes,
            'disease_names': self.disease_names,
            'image_size': self.image_size,
            'split': self.split
        }
        
        if self.multi_label:
            # Multi-label statistics
            label_counts = self.annotations[self.disease_names].sum(axis=0)
            stats['label_distribution'] = label_counts.to_dict()
            stats['avg_labels_per_sample'] = self.annotations[self.disease_names].sum(axis=1).mean()
        
        return stats


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Create train/val/test splits from RFMiD dataset
    
    Args:
        data_dir: Path to dataset directory
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
    """
    data_dir = Path(data_dir)
    
    # Load full annotations
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    df = pd.read_csv(csv_files[0])
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_df = df[:n_train]
    val_df = df[n_train:n_train+n_val]
    test_df = df[n_train+n_val:]
    
    # Save splits
    splits_dir = data_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(splits_dir / 'train_labels.csv', index=False)
    val_df.to_csv(splits_dir / 'val_labels.csv', index=False)
    test_df.to_csv(splits_dir / 'test_labels.csv', index=False)
    
    print(f"Created splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"Saved to {splits_dir}")


if __name__ == "__main__":
    # Example usage
    print("RFMiD Dataset Example")
    print("=" * 50)
    
    # Note: Update this path to your actual dataset location
    data_dir = "/path/to/RFMiD_2.0"
    
    # Create splits (run once)
    # create_data_splits(data_dir)
    
    # Load dataset
    # dataset = RFMiDDataset(
    #     data_dir=data_dir + '/splits',
    #     split='train',
    #     image_size=224,
    #     multi_label=True
    # )
    
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Number of classes: {dataset.n_classes}")
    # print(f"Disease names: {dataset.disease_names[:5]}...")
    
    # # Get sample
    # image, label = dataset[0]
    # print(f"Image shape: {image.shape}")
    # print(f"Label shape: {label.shape}")
    # print(f"Sample label: {label}")
    
    # # Get statistics
    # stats = dataset.get_statistics()
    # print(f"\nDataset statistics:")
    # for key, value in stats.items():
    #     if key != 'disease_names' and key != 'label_distribution':
    #         print(f"  {key}: {value}")
    
    print("\nTo use this dataset:")
    print("1. Download RFMiD 2.0 from Kaggle or Zenodo")
    print("2. Update data_dir path above")
    print("3. Run create_data_splits() to create train/val/test splits")
    print("4. Load dataset with RFMiDDataset()")
