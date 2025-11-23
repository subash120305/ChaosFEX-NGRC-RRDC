"""
Deep Feature Extractors for Medical Images

Implements pre-trained models for extracting deep features from fundus images:
- Vision Transformer (ViT)
- EfficientNet
- ResNet
- ConvNeXt
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from typing import Literal, Optional, Tuple
from torchvision import transforms


class FeatureExtractor(nn.Module):
    """
    Base class for deep feature extractors
    
    Args:
        model_name: Name of the pre-trained model
        pretrained: Whether to use pre-trained weights
        feature_dim: Dimension of output features
        freeze_backbone: Whether to freeze backbone weights
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b3',
        pretrained: bool = True,
        feature_dim: int = 1024,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        
        # Load pre-trained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[1]
        
        # Projection head to desired feature dimension
        if backbone_dim != feature_dim:
            self.projection = nn.Linear(backbone_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images
        
        Args:
            x: Input images (B x 3 x H x W)
            
        Returns:
            Features (B x feature_dim)
        """
        features = self.backbone(x)
        features = self.projection(features)
        return features
    
    def extract_features_numpy(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from numpy images
        
        Args:
            images: Input images (B x H x W x 3) or (H x W x 3)
            
        Returns:
            Features (B x feature_dim) or (feature_dim,)
        """
        # Handle single image
        single_image = False
        if images.ndim == 3:
            images = images[np.newaxis, ...]
            single_image = True
        
        # Convert to torch tensor
        images_tensor = torch.from_numpy(images).float()
        
        # Transpose to (B x 3 x H x W)
        images_tensor = images_tensor.permute(0, 3, 1, 2)
        
        # Normalize (ImageNet stats)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        images_tensor = normalize(images_tensor / 255.0)
        
        # Extract features
        self.eval()
        with torch.no_grad():
            features = self.forward(images_tensor)
        
        features_numpy = features.cpu().numpy()
        
        if single_image:
            features_numpy = features_numpy[0]
        
        return features_numpy


class VisionTransformerExtractor(FeatureExtractor):
    """
    Vision Transformer (ViT) feature extractor
    
    Available models:
    - vit_base_patch16_224
    - vit_large_patch16_224
    - vit_base_patch16_384
    """
    
    def __init__(
        self,
        model_size: Literal['base', 'large'] = 'base',
        image_size: int = 224,
        pretrained: bool = True,
        feature_dim: int = 1024
    ):
        if model_size == 'base':
            model_name = f'vit_base_patch16_{image_size}'
        elif model_size == 'large':
            model_name = f'vit_large_patch16_{image_size}'
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            feature_dim=feature_dim
        )


class EfficientNetExtractor(FeatureExtractor):
    """
    EfficientNet feature extractor
    
    Available models:
    - efficientnet_b0 to efficientnet_b7
    """
    
    def __init__(
        self,
        model_size: Literal['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'] = 'b3',
        pretrained: bool = True,
        feature_dim: int = 1024
    ):
        model_name = f'efficientnet_{model_size}'
        
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            feature_dim=feature_dim
        )


class ResNetExtractor(FeatureExtractor):
    """
    ResNet feature extractor
    
    Available models:
    - resnet50, resnet101, resnet152
    """
    
    def __init__(
        self,
        model_size: Literal['50', '101', '152'] = '50',
        pretrained: bool = True,
        feature_dim: int = 1024
    ):
        model_name = f'resnet{model_size}'
        
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            feature_dim=feature_dim
        )


class ConvNeXtExtractor(FeatureExtractor):
    """
    ConvNeXt feature extractor (modern CNN architecture)
    
    Available models:
    - convnext_tiny, convnext_small, convnext_base
    """
    
    def __init__(
        self,
        model_size: Literal['tiny', 'small', 'base'] = 'small',
        pretrained: bool = True,
        feature_dim: int = 1024
    ):
        model_name = f'convnext_{model_size}'
        
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            feature_dim=feature_dim
        )


class EnsembleFeatureExtractor:
    """
    Ensemble of multiple feature extractors
    Concatenates features from different models
    """
    
    def __init__(
        self,
        extractors: list,
        fusion_method: Literal['concat', 'mean', 'max'] = 'concat'
    ):
        self.extractors = extractors
        self.fusion_method = fusion_method
    
    def extract_features_numpy(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features using ensemble
        
        Args:
            images: Input images
            
        Returns:
            Fused features
        """
        all_features = []
        
        for extractor in self.extractors:
            features = extractor.extract_features_numpy(images)
            all_features.append(features)
        
        # Fuse features
        if self.fusion_method == 'concat':
            fused_features = np.concatenate(all_features, axis=-1)
        elif self.fusion_method == 'mean':
            fused_features = np.mean(all_features, axis=0)
        elif self.fusion_method == 'max':
            fused_features = np.max(all_features, axis=0)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features


def create_feature_extractor(
    model_type: str = 'efficientnet_b3',
    pretrained: bool = True,
    feature_dim: int = 1024
) -> FeatureExtractor:
    """
    Factory function to create feature extractors
    
    Args:
        model_type: Type of model ('vit_base', 'efficientnet_b3', 'resnet50', etc.)
        pretrained: Whether to use pre-trained weights
        feature_dim: Output feature dimension
        
    Returns:
        Feature extractor instance
    """
    if model_type.startswith('vit'):
        # Parse ViT model
        if 'base' in model_type:
            size = 'base'
        elif 'large' in model_type:
            size = 'large'
        else:
            size = 'base'
        
        if '384' in model_type:
            image_size = 384
        else:
            image_size = 224
        
        return VisionTransformerExtractor(
            model_size=size,
            image_size=image_size,
            pretrained=pretrained,
            feature_dim=feature_dim
        )
    
    elif model_type.startswith('efficientnet'):
        # Parse EfficientNet model
        size = model_type.split('_')[-1]  # e.g., 'b3'
        return EfficientNetExtractor(
            model_size=size,
            pretrained=pretrained,
            feature_dim=feature_dim
        )
    
    elif model_type.startswith('resnet'):
        # Parse ResNet model
        size = model_type.replace('resnet', '')  # e.g., '50'
        return ResNetExtractor(
            model_size=size,
            pretrained=pretrained,
            feature_dim=feature_dim
        )
    
    elif model_type.startswith('convnext'):
        # Parse ConvNeXt model
        size = model_type.split('_')[-1]  # e.g., 'small'
        return ConvNeXtExtractor(
            model_size=size,
            pretrained=pretrained,
            feature_dim=feature_dim
        )
    
    else:
        # Generic model using timm
        return FeatureExtractor(
            model_name=model_type,
            pretrained=pretrained,
            feature_dim=feature_dim
        )


if __name__ == "__main__":
    # Example usage
    print("Feature Extractor Example")
    print("=" * 50)
    
    # Create feature extractor
    extractor = create_feature_extractor(
        model_type='efficientnet_b3',
        pretrained=True,
        feature_dim=1024
    )
    
    print(f"Model: {extractor.model_name}")
    print(f"Feature dimension: {extractor.feature_dim}")
    
    # Test with random image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = extractor.extract_features_numpy(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Sample features: {features[:5]}")
    
    # Test with batch
    batch_images = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
    batch_features = extractor.extract_features_numpy(batch_images)
    
    print(f"\nBatch input shape: {batch_images.shape}")
    print(f"Batch output shape: {batch_features.shape}")
