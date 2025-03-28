import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../'))

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Sample data storage (in-memory)
data_store = {
    'clients': [], # Each client will have it's address
    'current_round_clients': [], # Client Index in 'clients' list
    'round_parameters': {}, # Parameters for current round
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Edge Server API!"})

# Register Clients
@app.route("/register_client", methods=["POST"])
def register_client():
    data = request.json

    address = data.get('address', None)

    # Add client information to data store
    data_store['clients'].append({
        'address': address,
    })

# Start Training
@app.route("/start_training", methods=["POST"])
def start_training():
    data = request.json

    batch_size = data.get('batch_size', None)
    learning_rate = data.get('learning_rate', None)
    num_epochs = data.get('num_epochs', None)
    parameters = data.get('parameters', None)

    for client_idx in data_store['current_round_clients']:
        client_url = data_store['clients'][client_idx]['address']

        data = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'parameters': parameters
        }

        # Ask the client to start training
        requests.post(f"{client_url}/start_training", json=data)

# Aggregation
@app.route("/aggregate", methods=["POST"])
def aggregate():
    # Add model parameters to data store
    # If all devices have sent the data, aggregate
    # Send parameters up the hierarchy
    pass

# Stop training
@app.route("/stop", methods=["POST"])
def stop():
    # Stop
    pass

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run on all interfaces