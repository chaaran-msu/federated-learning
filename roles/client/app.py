import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../../'))

from flask import Flask, request, jsonify

from tasks.traininig.models.baseline import Baseline

app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing

# Sample data storage (in-memory)
data_store = {}

# Initalize model
model = Baseline()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Client API!"})

# Start Training
@app.route("/start_training", methods=["POST"])
def start_training():
    # Deserialize parameters
    # Set model parameters
    # Start training
    # Serialize parameters and send data back
    pass

# Stop App
@app.route("/stop", methods=["POST"])
def stop():
    # Kill the process
    pass

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Run on all interfaces