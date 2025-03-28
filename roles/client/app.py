import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify

from tasks.traininig.train_round import train_round

app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Sample data storage (in-memory)
data_store = {
    'partition_id': partition_id,
    'num_clients': num_clients
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

    # Deserialize parameters
    parameters = ''

    # Training
    accuracy = train_round(
        num_clients=data_store['num_clients'],
        partition_id=data_store['partition_id'],
        parameters=parameters,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    # Serialize parameters and send data back
    pass

# Stop App
@app.route("/stop", methods=["POST"])
def stop():
    # Kill the process
    pass

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run on all interfaces