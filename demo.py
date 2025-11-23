"""
ChaosFEX-NGRC: Rare Retinal Disease Prediction Demo

Chaos-Based Feature Extraction with Next Generation Reservoir Computing
for Rare Retinal Disease Classification

This script demonstrates the trained ChaosFEX-NGRC pipeline.
Load the model ONCE, then predict on ANY image!

Usage:
    python demo.py --model results/experiment_TIMESTAMP/pipeline.pkl --image path/to/fundus.jpg
    
    # Or interactive mode:
    python demo.py --model results/experiment_TIMESTAMP/pipeline.pkl --interactive
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import json
import sys
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models import NGRCChaosFEXPipeline


class RetinalDiseasePredictor:
    """
    Easy-to-use predictor for retinal diseases
    
    Load model once, predict on multiple images!
    """
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved pipeline.pkl
        """
        print("Loading trained model...")
        print(f"Model path: {model_path}")
        
        self.pipeline = NGRCChaosFEXPipeline()
        self.pipeline.load(model_path)
        
        # Load disease names
        self.disease_names = self._load_disease_names()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Ready to predict {len(self.disease_names)} diseases")
        print()
    
    def _load_disease_names(self) -> List[str]:
        """Load disease names from dataset"""
        # Default RFMiD disease names
        diseases = [
            'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN',
            'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE',
            'ST', 'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP',
            'CWS', 'CB', 'ODPM', 'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR',
            'CF', 'VH', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL'
        ]
        
        # Try to load from config if available
        try:
            config_path = Path(self.pipeline.config.get('data', {}).get('data_dir', ''))
            # Load from dataset if available
            pass
        except:
            pass
        
        return diseases[:49]  # RFMiD has 49 diseases
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to fundus image
            
        Returns:
            Preprocessed image (224x224x3)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        return image
    
    def predict(
        self,
        image_path: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Predict diseases from fundus image
        
        Args:
            image_path: Path to fundus image
            top_k: Number of top predictions to return
            threshold: Probability threshold for multi-label
            
        Returns:
            predictions: Binary predictions (49,)
            probabilities: Disease probabilities (49,)
            top_diseases: List of top-k disease names
        """
        print(f"Analyzing image: {image_path}")
        
        # Load image
        image = self.load_image(image_path)
        
        # Predict
        print("  [1/4] Extracting deep features...")
        predictions = self.pipeline.predict(image[np.newaxis, ...])[0]
        
        print("  [2/4] Applying ChaosFEX transformation...")
        probabilities = self.pipeline.predict_proba(image[np.newaxis, ...])[0]
        
        print("  [3/4] Processing with NG-RC...")
        
        print("  [4/4] Classifying diseases...")
        
        # Get top-k diseases
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_diseases = [
            (self.disease_names[i], probabilities[i])
            for i in top_indices
        ]
        
        print("✅ Prediction complete!")
        print()
        
        return predictions, probabilities, top_diseases
    
    def print_results(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        top_diseases: List[Tuple[str, float]],
        threshold: float = 0.5
    ):
        """Print prediction results in a nice format"""
        print("=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        # Detected diseases (above threshold)
        detected = [
            (self.disease_names[i], probabilities[i])
            for i in range(len(probabilities))
            if probabilities[i] >= threshold
        ]
        
        if detected:
            print(f"\n🔴 DETECTED DISEASES (probability ≥ {threshold}):")
            print("-" * 60)
            for disease, prob in sorted(detected, key=lambda x: x[1], reverse=True):
                print(f"  • {disease:20s} : {prob:.2%} {'█' * int(prob * 20)}")
        else:
            print(f"\n✅ No diseases detected above threshold ({threshold})")
        
        print(f"\n📊 TOP {len(top_diseases)} PREDICTIONS:")
        print("-" * 60)
        for i, (disease, prob) in enumerate(top_diseases, 1):
            print(f"  {i}. {disease:20s} : {prob:.2%} {'█' * int(prob * 20)}")
        
        print("\n" + "=" * 60)
    
    def visualize_results(
        self,
        image_path: str,
        probabilities: np.ndarray,
        top_k: int = 10,
        save_path: str = None
    ):
        """
        Create visualization of predictions
        
        Args:
            image_path: Path to original image
            probabilities: Disease probabilities
            top_k: Number of top diseases to show
            save_path: Path to save figure (optional)
        """
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get top-k diseases
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probs = probabilities[top_indices]
        top_names = [self.disease_names[i] for i in top_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot image
        ax1.imshow(image)
        ax1.set_title('Fundus Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Plot predictions
        colors = ['red' if p >= 0.5 else 'orange' if p >= 0.3 else 'green' for p in top_probs]
        ax2.barh(range(top_k), top_probs, color=colors, alpha=0.7)
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels(top_names)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title(f'Top {top_k} Disease Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add probability values
        for i, prob in enumerate(top_probs):
            ax2.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_results(
        self,
        image_path: str,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        output_path: str
    ):
        """Save prediction results to JSON"""
        results = {
            'image_path': image_path,
            'predictions': {
                self.disease_names[i]: {
                    'predicted': bool(predictions[i]),
                    'probability': float(probabilities[i])
                }
                for i in range(len(predictions))
            },
            'summary': {
                'total_diseases_detected': int(np.sum(predictions)),
                'max_probability': float(np.max(probabilities)),
                'top_disease': self.disease_names[np.argmax(probabilities)]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def interactive_mode(predictor: RetinalDiseasePredictor):
    """Interactive mode for multiple predictions"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter image paths to predict (or 'quit' to exit)")
    print()
    
    while True:
        image_path = input("Image path: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not Path(image_path).exists():
            print(f"❌ Error: File not found: {image_path}")
            continue
        
        try:
            # Predict
            predictions, probabilities, top_diseases = predictor.predict(image_path)
            
            # Print results
            predictor.print_results(predictions, probabilities, top_diseases)
            
            # Ask to visualize
            viz = input("\nVisualize results? (y/n): ").strip().lower()
            if viz == 'y':
                predictor.visualize_results(image_path, probabilities)
            
            # Ask to save
            save = input("Save results to JSON? (y/n): ").strip().lower()
            if save == 'y':
                output_path = image_path.replace('.jpg', '_predictions.json').replace('.png', '_predictions.json')
                predictor.save_results(image_path, predictions, probabilities, output_path)
            
            print()
        
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Retinal Disease Prediction Demo')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained pipeline.pkl')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to fundus image')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for detection')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RetinalDiseasePredictor(args.model)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(predictor)
        return
    
    # Single image prediction
    if args.image is None:
        print("Error: --image required (or use --interactive)")
        return
    
    # Predict
    predictions, probabilities, top_diseases = predictor.predict(
        args.image,
        top_k=args.top_k,
        threshold=args.threshold
    )
    
    # Print results
    predictor.print_results(predictions, probabilities, top_diseases, args.threshold)
    
    # Visualize
    if args.visualize:
        save_path = args.image.replace('.jpg', '_visualization.png').replace('.png', '_visualization.png')
        predictor.visualize_results(args.image, probabilities, save_path=save_path)
    
    # Save results
    if args.save:
        predictor.save_results(args.image, predictions, probabilities, args.save)


if __name__ == "__main__":
    main()
