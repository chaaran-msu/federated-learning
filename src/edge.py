import sys
import requests
from utils import *
from flask import Flask, request, jsonify

port = port()
local = local(port)

edge_index = sys.argv[1]
client_index = sys.argv[2]
server = sys.argv[3]
clients = set()

def create_app():
    app = Flask(__name__)

    with app.app_context():
        url = f'http://{server}/register?address={local}&role=edge&edge_index={edge_index}'
        requests.get(url)

    @app.route('/')
    def home():
        return 'OK'

    @app.route('/bind')
    def bind():
        address = request.args.get('address', None)
        clients.add(address)

        return 'OK'

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=port, debug=False)
