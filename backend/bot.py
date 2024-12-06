import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict_intent():
    if model is None or label_encoder is None:
        return jsonify({
            'error': 'Model or label encoder not loaded. Check server logs.'
        }), 500

    try:
        # Get input text from request
        data = request.get_json()
        input_text = data.get('text', '')

        if not input_text:
            return jsonify({
                'error': 'No text provided'
            }), 400

        # Predict intent
        prediction = model.predict([input_text])
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba([input_text])

        # Create response with top probabilities
        top_labels = label_encoder.inverse_transform(
            np.argsort(probabilities[0])[::-1]
        )
        top_probs = np.sort(probabilities[0])[::-1]

        top_predictions = [
            {'label': label, 'probability': float(prob)}
            for label, prob in zip(top_labels[:3], top_probs[:3])
        ]

        return jsonify({
            'predicted_intent': predicted_label,
            'top_predictions': top_predictions
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'label_encoder_loaded': label_encoder is not None
    })


if __name__ == '__main__':
    if os.path.exists('best_model.pkl') and os.path.exists('label_encoder.pkl'):
        app.run(debug=True, port=5000)
    else:
        print("Model or label encoder file missing. Cannot start server.")