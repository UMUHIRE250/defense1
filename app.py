from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = tf.keras.models.load_model('path_to_saved_model/my_model.keras')
scaler = joblib.load('path_to_saved_model/scaler.save')

# Dummy house data
house = {
    "Floor 1": {f"Room {i+1}": {"details": f"Details about Room {i+1}", "color": "white"} for i in range(8)},
    "Floor 2": {f"Room {i+1}": {"details": f"Details about Room {i+1}", "color": "white"} for i in range(8)},
    "Floor 3": {f"Room {i+1}": {"details": f"Details about Room {i+1}", "color": "white"} for i in range(8)}
}

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/house', methods=['GET'])
def get_house():
    return jsonify(house)

@app.route('/api/house/<floor>/<room>', methods=['PUT'])
def update_room_details(floor, room):
    if floor in house and room in house[floor]:
        data = request.get_json()
        house[floor][room]['details'] = data.get('details', house[floor][room]['details'])
        house[floor][room]['color'] = data.get('color', house[floor][room]['color'])
        return jsonify(house[floor][room]), 200
    else:
        return jsonify({'error': 'Room not found'}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    rssi_values = np.array(data['rssi_values']).reshape(1, -1)  # Assuming rssi_values is a list of RSSI values
    rssi_values = scaler.transform(rssi_values)  # Apply the same scaler used during training
    prediction = model.predict(rssi_values)
    predicted_room = np.argmax(prediction, axis=1)[0]
    return jsonify({"predicted_room": predicted_room}), 200

@app.route('/api/ip-address', methods=['GET'])
def get_ip_address():
    return request.remote_addr

if __name__ == '__main__':
    app.run(debug=True)
