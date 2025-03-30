import sys
import requests
from utils import *
from flask import Flask, request

app = Flask(__name__)

port = port()
local = local(port)
edge = ''

edge_index = sys.argv[1]
# client_index = sys.argv[2]
server = sys.argv[3]

# register edge
url = f'http://{server}/register?address={local}&role=client&edge_index={edge_index}'
requests.get(url)

@app.route('/')
def home():
    return 'OK'

@app.route('/bind')
def bind():
    address = request.args.get('address', None)
    edge = address

    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
