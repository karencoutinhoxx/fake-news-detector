from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app_status = {
    'model_loaded': False,
    'total_predictions': 0,
    'last_prediction_time': None,
    'uptime_start': datetime.now()
}

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    uptime = (datetime.now() - app_status['uptime_start']).total_seconds()
    
    return jsonify({
        'status': 'running',
        'total_predictions': app_status['total_predictions'],
        'uptime_seconds': uptime,
        'last_prediction': app_status['last_prediction_time']
    }), 200

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    uptime = (datetime.now() - app_status['uptime_start']).total_seconds()
    
    return jsonify({
        'total_predictions': app_status['total_predictions'],
        'uptime_seconds': uptime,
        'last_prediction': app_status['last_prediction_time']
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data.get('text', '')
        title = data.get('title', '')
        
        # Placeholder prediction logic
        # In production, this would use the trained ensemble model
        fake_prob = 0.65
        real_prob = 1 - fake_prob
        
        label = 'likely_fake' if fake_prob > 0.5 else 'likely_real'
        confidence = max(fake_prob, real_prob)
        
        app_status['total_predictions'] += 1
        app_status['last_prediction_time'] = datetime.now().isoformat()
        
        response = {
            'prediction': round(fake_prob, 3),
            'label': label,
            'confidence': round(confidence, 3),
            'probabilities': {
                'fake': round(fake_prob, 3),
                'real': round(real_prob, 3)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {label} (confidence: {confidence:.3f})")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        articles = data.get('articles', [])
        
        if not articles:
            return jsonify({'error': 'No articles provided'}), 400
        
        results = []
        
        for article in articles:
            text = article.get('text', '')
            
            # Placeholder prediction
            fake_prob = 0.65
            
            results.append({
                'text': text[:100] + '...',
                'prediction': round(fake_prob, 3),
                'label': 'likely_fake' if fake_prob > 0.5 else 'likely_real',
                'confidence': round(max(fake_prob, 1-fake_prob), 3)
            })
        
        app_status['total_predictions'] += len(articles)
        app_status['last_prediction_time'] = datetime.now().isoformat()
        
        return jsonify({
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
