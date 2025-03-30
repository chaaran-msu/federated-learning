import sys
import requests
from utils import *

port = port()
local = local(port)
edge_index = sys.argv[1]
client_index = sys.argv[2]
server = sys.argv[3]

url = f'http://{server}/register?address={local}&role=client&edge_index={edge_index}'
res = requests.get(url)
print(res)
print("client")
