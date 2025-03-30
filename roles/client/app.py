import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify
import requests

from tasks.traininig.train_round import train_round

app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Sample data storage (in-memory)
data_store = {
    'partition_id': partition_id,
    'num_clients': num_clients,
    'server_address': server_address
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Client API!"})

# Start Training
@app.route("/start_training", methods=["POST"])
def start_training():
    data = request.json

    batch_size = data.get('batch_size', None)
    learning_rate = data.get('learning_rate', None)
    num_epochs = data.get('num_epochs', None)
    parameters = data.get('parameters', None)

    # Training
    round_data = train_round(
        num_clients=data_store['num_clients'],
        partition_id=data_store['partition_id'],
        parameters=parameters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    round_data['partitoin_id'] = data_store['partition_id']

    server_address = data_store['server_address']

    # Send data back to server
    requests.post(
        url=f'{server_address}/aggregate',
        json=round_data
    )

# Stop App
@app.route("/stop", methods=["POST"])
def stop():
    # Kill the process
    pass

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run on all interfaces