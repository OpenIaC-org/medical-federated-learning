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
        self.model_updates = []

        self.train_amount = 4500
        self.valid_amount = 900
        self.test_amount = 900
        self.number_of_samples = 10

        self.centralized_model = Net2nn()
        self.centralized_optimizer = torch.optim.SGD(self.centralized_model.parameters(), lr=0.01, momentum=0.9)
        self.centralized_criterion = nn.CrossEntropyLoss()

        self.data_is_loaded = False

    async def handler(self, websocket, path):
        while True:
            try:    
                data = await websocket.recv()
                if data == 'connect':
                    await self.handle_connection(websocket)
                    if len(self.clients) == 3:
                        await self.distribute_model()
                if data == 'model_update':
                    await self.save_client_model(websocket)
                    if len(self.model_updates) == len(self.clients):
                        await self.update_centralized_model()
                        self.validate_model()
                        
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
    
    async def save_client_model(self, websocket):
        print(f'Saving model from {websocket.id}')
        self.model_updates.append(websocket.id)
        model = await websocket.recv()
        with open(f'./client_models/{websocket.id}.pt', 'wb') as f:
            f.write(model)
    
    async def update_centralized_model(self):
        print('Updating centralized model')

        l1_weights = []
        l1_biases = []

        l2_weights = []
        l2_biases = []
        
        l3_weights = []
        l3_biases = []

        with torch.no_grad():
            for index, client_id in enumerate(self.model_updates):
                model = torch.load(f'./client_models/{client_id}.pt')
                current_model = Net2nn()
                current_model.load_state_dict(model)
                if index == 0:
                    l1_weights = current_model.fc1.weight.data.clone()
                    l1_biases = current_model.fc1.bias.data.clone()

                    l2_weights = current_model.fc2.weight.data.clone()
                    l2_biases = current_model.fc2.bias.data.clone()

                    l3_weights = current_model.fc3.weight.data.clone()
                    l3_biases = current_model.fc3.bias.data.clone()
                else:
                    l1_weights += current_model.fc1.weight.data.clone()
                    l1_biases += current_model.fc1.bias.data.clone()

                    l2_weights += current_model.fc2.weight.data.clone()
                    l2_biases += current_model.fc2.bias.data.clone()

                    l3_weights += current_model.fc3.weight.data.clone()
                    l3_biases += current_model.fc3.bias.data.clone()

            self.centralized_model.fc1.weight.data = l1_weights / len(self.model_updates)
            self.centralized_model.fc1.bias.data = l1_biases / len(self.model_updates)

            self.centralized_model.fc2.weight.data = l2_weights / len(self.model_updates)
            self.centralized_model.fc2.bias.data = l2_biases / len(self.model_updates)

            self.centralized_model.fc3.weight.data = l3_weights / len(self.model_updates)
            self.centralized_model.fc3.bias.data = l3_biases / len(self.model_updates)

            torch.save(self.centralized_model.state_dict(), 'model.pt')
        print('Model was updated')

    def validate_model(self):
        if not self.data_is_loaded:
            self.load_data()
        print('Validating centralized model')
        
        test_loss, test_accuracy = validation(self.centralized_model, self.test_dl, self.centralized_criterion)
        print(f'Test loss: {test_loss} | Test accuracy: {test_accuracy}')

    def load_data(self):
        print('Loading data')
        with gzip.open('./data/mnist/mnist.pkl.gz', "rb") as f:
            ((self.train_imgs, self.train_labels), (self.valid_imgs, self.valid_labels), (self.test_imgs, self.test_labels)) = pickle.load(f, encoding="latin-1")

        self.test_imgs, self.test_labels = map(torch.tensor, (self.test_imgs, self.test_labels))

        test_ds = TensorDataset(self.test_imgs, self.test_labels)
        self.test_dl = DataLoader(test_ds, batch_size=32 * 2)

        self.data_is_loaded = True


    def handle_error(self, websocket):
        print('Client ' + str(websocket.id) + ' disconnected')
        self.clients.pop(str(websocket.id))
    

server = Server()

start_server = websockets.serve(server.handler, "", PORT, ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
print('Server started on port ' + str(PORT))
asyncio.get_event_loop().run_forever()