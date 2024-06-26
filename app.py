from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
from scripts.model import prediction_fp, prediction_mc, prediction_ml

# Initialize
app = Flask(__name__)
CORS(app)  # Apply CORS to the entire app

# Creating Endpoints
@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': 'API is working'})

# Fp
@app.route('/prediction/fp', methods=['GET'])
def fp():
    param = request.args.get('date', default='default_value', type=str)
    result = prediction_fp(param)
    return jsonify({'message': f'{result}'})

# Mc
@app.route('/prediction/mc', methods=['GET'])
def mc():
    param = request.args.get('date', default='default_value', type=str)
    result = prediction_mc(param)
    return jsonify({'message': f'{result}'})

# Ml
@app.route('/prediction/ml', methods=['GET'])
def ml():
    param = request.args.get('date', default='default_value', type=str)
    result = prediction_ml(param)
    return jsonify({'message': f'{result}'})

# Start App
if __name__ == '__main__':
    app.run(port=8080, debug=True)
