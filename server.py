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
pd.options.display.float_format = "{:,.4f}".format

PORT = 8000

class Client:
    def __init__(self, websocket, status):
        self.websocket = websocket
        self.status = status

class Server:
    def __init__(self):
        self.clients = {}
        with gzip.open('./data/mnist/mnist.pkl.gz', "rb") as f:
            ((self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test)) = pickle.load(f, encoding="latin-1")
        
        self.train_amount = 4500
        self.valid_amount = 900
        self.test_amount = 900
        self.number_of_samples = 10

        self.format_data()

    def format_data(self):
        self.label_dict_train=split_and_shuffle_labels(y_data=self.y_train, seed=1, amount=self.train_amount) 
        self.sample_dict_train=get_iid_subsamples_indices(label_dict=self.label_dict_train, number_of_samples=self.number_of_samples, amount=self.train_amount)
        self.x_train_dict, self.y_train_dict = create_iid_subsamples(sample_dict=self.sample_dict_train, x_data=self.x_train, y_data=self.y_train, x_name="x_train", y_name="y_train")

        self.label_dict_valid = split_and_shuffle_labels(y_data=self.y_valid, seed=1, amount=self.train_amount) 
        self.sample_dict_valid = get_iid_subsamples_indices(label_dict=self.label_dict_valid, number_of_samples=self.number_of_samples, amount=self.valid_amount)
        self.x_valid_dict, self.y_valid_dict = create_iid_subsamples(sample_dict=self.sample_dict_valid, x_data=self.x_valid, y_data=self.y_valid, x_name="x_valid", y_name="y_valid")

        self.label_dict_test = split_and_shuffle_labels(y_data=self.y_test, seed=1, amount=self.test_amount) 
        self.sample_dict_test = get_iid_subsamples_indices(label_dict=self.label_dict_test, number_of_samples=self.number_of_samples, amount=self.test_amount)
        self.x_test_dict, self.y_test_dict = create_iid_subsamples(sample_dict=self.sample_dict_test, x_data=self.x_test, y_data=self.y_test, x_name="x_test", y_name="y_test")

        print(self.x_train_dict.keys())

    async def handler(self, websocket, path):
        while True:
            try:    
                data = await websocket.recv()
                if data == 'connect':
                    await self.handle_connection(websocket)
                    if len(self.clients) == 3:
                        await self.distribute_data()
                    
            except Exception as e:
                print(e)
                self.handle_error(websocket)
                return

    async def handle_connection(self, websocket):
        self.clients[str(websocket.id)] = Client(websocket, ClientStatus.CONNECTED)
        await websocket.send(str(websocket.id))
        print('Client ' + str(websocket.id) + ' connected')
    
    async def distribute_data(self):
        print('Distributing data...')
        for c in self.clients:
            current_client = self.clients[c]
            if current_client.status == ClientStatus.CONNECTED:
                await current_client.websocket.send('data')
                # await current_client.websocket.send()
                
    def handle_error(self, websocket):
        print('Client ' + str(websocket.id) + ' disconnected')
        self.clients.pop(str(websocket.id))
    

server = Server()

start_server = websockets.serve(server.handler, "localhost", PORT, ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
print('Server started on port ' + str(PORT))
asyncio.get_event_loop().run_forever()