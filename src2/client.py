import sys
import socket
import requests
from config import Config
from flask import Flask, request
from utils import get_port

edge_index = int(sys.argv[1])
client_index = int(sys.argv[2])
parent_address = str(sys.argv[3])

app = Flask(__name__)
config = Config("_config.yml")

port = get_port(config.get_global_index(edge_index, client_index))
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
local_address = ip + ":" + str(port)

res = requests.get('http://' + parent_address + '/?address=' + local_address)
print(res)
