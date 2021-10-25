import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random
import math
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
from pathlib import Path
import requests
import pickle
import gzip
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import asyncio
import websockets
from enums import ClientStatus
import json
from utils import *
from network import *
pd.options.display.float_format = "{:,.4f}".format

PORT = 8000

class Client:
    def __init__(self, websocket, status, num):
        self.websocket = websocket
        self.status = status
        self.client_number = num

class Server:
    def __init__(self):
        self.clients = {}

        self.train_amount = 4500
        self.valid_amount = 900
        self.test_amount = 900
        self.number_of_samples = 10

        self.centralized_model = Net2nn()
        self.centralized_optimizer = torch.optim.SGD(self.centralized_model.parameters(), lr=0.01, momentum=0.9)

    async def handler(self, websocket, path):
        while True:
            try:    
                data = await websocket.recv()
                if data == 'connect':
                    await self.handle_connection(websocket)
                    if len(self.clients) == 3:
                        await self.distribute_model()
                    
            except Exception as e:
                print(e)
                self.handle_error(websocket)
                return

    async def handle_connection(self, websocket):
        self.clients[str(websocket.id)] = Client(websocket, ClientStatus.CONNECTED, len(self.clients))
        await websocket.send(str(websocket.id))
        print('Client ' + str(websocket.id) + ' connected')
        await websocket.send(str(len(self.clients)))

    async def distribute_model(self):
        print('Distributing model')
        torch.save(self.centralized_model.state_dict(), 'model.pt')
        for client in self.clients:
            await self.clients[client].websocket.send('model')
            await self.clients[client].websocket.send(open('model.pt', 'rb').read())
                            
    def handle_error(self, websocket):
        print('Client ' + str(websocket.id) + ' disconnected')
        self.clients.pop(str(websocket.id))
    

server = Server()

start_server = websockets.serve(server.handler, "localhost", PORT, ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
print('Server started on port ' + str(PORT))
asyncio.get_event_loop().run_forever()