import sys
import requests
from utils import *
from flask import Flask, request

app = Flask(__name__)

port = port()
local = local(port)

edge_index = sys.argv[1]
# client_index = sys.argv[2]
server = sys.argv[3]

clients = set()

# register edge
url = f'http://{server}/register?address={local}&role=edge&edge_index={edge_index}'
res = requests.get(url)

@app.route('/')
def home():
    return 'OK'

@app.route('/bind')
def bind():
    address = request.args.get('address', None)
    clients.add(address)

    url = f'http://{address}/bind?address={local}'
    requests.get(url)

    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
