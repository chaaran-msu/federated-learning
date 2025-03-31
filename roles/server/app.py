import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests

from tasks.aggregation.aggregate import aggregate

app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Sample data storage (in-memory)
data_store = {
    'clients': [], # Each client will have it's address
    'current_round_clients': [], # Client Index in 'clients' list,
    'current_round_data': {}
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

# Aggregation
@app.route("/aggregate", methods=["POST"])
def aggregate_clients():
    data = request.json

    # Add model parameters to data store
    parameters = data.get('parameters', None)
    accuracy = data.get('accuracy', None)
    num_samples = data.get('num_samples', None)
    partition_id = data.get('partition_id', None)

    data_store['current_round_data'][partition_id] = {
        'parameters': parameters,
        'accuracy': accuracy,
        'num_samples': num_samples
    }

    # If all devices have sent the data, aggregate
    if len(data_store['current_round_parameters']) == len(data_store['current_round_clients']):
        aggregate(data_store['current_round_data'])

    # Start Training for next round
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

# Stop training
@app.route("/stop", methods=["POST"])
def stop():
    # Stop
    pass

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run on all interfaces