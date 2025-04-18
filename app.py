from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load  RandomForest model from pickle
with open('RandomForest.pkl', 'rb') as f:
    RF = pickle.load(f)


def recommend_crop(n, p, k, temperature, humidity, ph, rainfall):
    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    recommended_crop = RF.predict(features)
    return recommended_crop[0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend_crop_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400

    n = float(data.get('n'))
    p = float(data.get('p'))
    k = float(data.get('k'))
    temperature = float(data.get('temperature'))
    humidity = float(data.get('humidity'))
    ph = float(data.get('ph'))
    rainfall = float(data.get('rainfall'))

    crop = recommend_crop(n, p, k, temperature, humidity, ph, rainfall)
    return jsonify({'crop': crop})


if __name__ == '__main__':
    app.run(debug=True)
