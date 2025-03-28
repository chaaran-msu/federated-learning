import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(dirname, '../'))

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
    return jsonify({"message": "Edge Server API!"})

# Register Clients
@app.route("/register_client", methods=["POST"])
def register_client():
    # Add client information to data store
    pass

# Start Training
@app.route("/start_training", methods=["POST"])
def start_training():
    # send it to clients
    pass

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