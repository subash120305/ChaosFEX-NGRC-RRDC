"""
ChaosFEX-NGRC: Web Demo Interface

Chaos-Based Feature Extraction with Next Generation Reservoir Computing
for Rare Retinal Disease Classification

A beautiful web interface for demonstrating the trained model.
Upload fundus images and get instant predictions!

Usage:
    python web_demo.py --model results/experiment_TIMESTAMP/pipeline.pkl
    
Then open: http://localhost:5000
"""

import argparse
from flask import Flask, request, render_template_string, jsonify
import numpy as np
import cv2
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import sys

sys.path.append(str(Path(__file__).parent))
from src.models import NGRCChaosFEXPipeline


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ChaosFEX-NGRC: Retinal Disease Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            text-align: center;
            padding: 40px;
            border: 3px dashed #667eea;
            border-radius: 15px;
            background: #f8f9ff;
            margin-bottom: 30px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-section:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        
        .upload-section input[type="file"] {
            display: none;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.2s;
            margin: 10px;
        }
        
        .btn:hover {
            transform: scale(1.05);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .results {
            display: none;
            margin-top: 30px;
        }
        
        .results.show {
            display: block;
        }
        
        .image-preview {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .image-preview img {
            max-width: 400px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .predictions {
            background: #f8f9ff;
            padding: 30px;
            border-radius: 15px;
        }
        
        .prediction-item {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .prediction-rank {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
            margin-right: 20px;
            min-width: 40px;
        }
        
        .prediction-name {
            flex: 1;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .prediction-prob {
            font-size: 1.2em;
            font-weight: bold;
            color: #764ba2;
            margin-right: 15px;
        }
        
        .prediction-bar {
            width: 200px;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .prediction-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .alert-danger {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 ChaosFEX-NGRC</h1>
            <p>Chaos-Based Feature Extraction with Next Generation Reservoir Computing</p>
            <p style="font-size: 0.9em; margin-top: 5px;">Rare Retinal Disease Classification</p>
        </div>
        
        <div class="content">
            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <h2>Upload Fundus Image</h2>
                <p>Click to select or drag and drop</p>
                <p style="font-size: 0.9em; color: #666; margin-top: 10px;">Supported: JPG, PNG</p>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div style="text-align: center;">
                <button class="btn" id="predictBtn" disabled>🔍 Analyze Image</button>
                <button class="btn" id="clearBtn" style="background: #6c757d;">🔄 Clear</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <h3>Analyzing image...</h3>
                <p>This may take a few seconds</p>
            </div>
            
            <div class="results" id="results">
                <div class="image-preview" id="imagePreview"></div>
                
                <div class="predictions">
                    <h2 style="margin-bottom: 20px;">📊 Prediction Results</h2>
                    <div id="predictionsList"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const fileInput = document.getElementById('fileInput');
        const predictBtn = document.getElementById('predictBtn');
        const clearBtn = document.getElementById('clearBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const imagePreview = document.getElementById('imagePreview');
        const predictionsList = document.getElementById('predictionsList');
        
        let selectedFile = null;
        
        fileInput.addEventListener('change', function(e) {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                predictBtn.disabled = false;
                
                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.innerHTML = '<img src="' + e.target.result + '" alt="Preview">';
                };
                reader.readAsDataURL(selectedFile);
            }
        });
        
        predictBtn.addEventListener('click', async function() {
            if (!selectedFile) return;
            
            // Show loading
            loading.classList.add('show');
            results.classList.remove('show');
            predictBtn.disabled = true;
            
            // Create form data
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                // Send to server
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Hide loading
                loading.classList.remove('show');
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    predictBtn.disabled = false;
                    return;
                }
                
                // Show results
                displayResults(data);
                results.classList.add('show');
                
            } catch (error) {
                loading.classList.remove('show');
                alert('Error: ' + error.message);
                predictBtn.disabled = false;
            }
        });
        
        clearBtn.addEventListener('click', function() {
            selectedFile = null;
            fileInput.value = '';
            predictBtn.disabled = true;
            results.classList.remove('show');
            imagePreview.innerHTML = '';
        });
        
        function displayResults(data) {
            let html = '';
            
            data.top_diseases.forEach((item, index) => {
                const prob = (item.probability * 100).toFixed(1);
                html += `
                    <div class="prediction-item">
                        <div class="prediction-rank">#${index + 1}</div>
                        <div class="prediction-name">${item.disease}</div>
                        <div class="prediction-prob">${prob}%</div>
                        <div class="prediction-bar">
                            <div class="prediction-bar-fill" style="width: ${prob}%"></div>
                        </div>
                    </div>
                `;
            });
            
            predictionsList.innerHTML = html;
            predictBtn.disabled = false;
        }
    </script>
</body>
</html>
"""


app = Flask(__name__)
predictor = None


@app.route('/')
def index():
    """Serve the main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (224, 224))
        
        # Predict
        predictions, probabilities, _ = predictor.predict(image[np.newaxis, ...])
        
        # Get top 10 diseases
        top_indices = np.argsort(probabilities)[-10:][::-1]
        top_diseases = [
            {
                'disease': predictor.disease_names[i],
                'probability': float(probabilities[i]),
                'predicted': bool(predictions[i])
            }
            for i in top_indices
        ]
        
        return jsonify({
            'success': True,
            'top_diseases': top_diseases,
            'total_detected': int(np.sum(predictions))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    global predictor
    
    parser = argparse.ArgumentParser(description='Web Demo for Retinal Disease Prediction')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained pipeline.pkl')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run server on')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    from demo import RetinalDiseasePredictor
    predictor = RetinalDiseasePredictor(args.model)
    
    print("\n" + "=" * 60)
    print("🚀 Web Demo Server Starting...")
    print("=" * 60)
    print(f"\n📍 Open your browser and go to:")
    print(f"   http://{args.host}:{args.port}")
    print(f"\n💡 Upload a fundus image to get predictions!")
    print(f"\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run server
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
