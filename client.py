import asyncio
import websockets
import sys
import gzip
from network import *
from enums import ClientStatus
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import json


class Client:
    def __init__(self):
        self.my_status = ClientStatus.NOT_CONNECTED
        self.EPOCHS = 10
        self.BATCH_SIZE = 32
        
    
    async def handler(self):
        async with websockets.connect('ws://localhost:8000', ping_interval=None) as websocket:
            self.websocket = websocket
            while True:
                if self.my_status == ClientStatus.NOT_CONNECTED:
                    await self.connect(websocket)
                elif self.my_status == ClientStatus.CONNECTED:
                    msg = await websocket.recv()
                    if msg == 'model':
                        await self.receive_model(websocket)
                    
    async def connect(self, websocket):
        await websocket.send('connect')
        res = await websocket.recv()
        self.id = str(res)
        self.my_status = ClientStatus.CONNECTED
        print('Connected to server. My Id is ' + self.id)
        self.client_number = int(await websocket.recv())
        print(f'I am client number {self.client_number}')
        self.load_data()
    
    async def receive_model(self, websocket):
        model = await websocket.recv()
        with open(f'./client_models/{self.id}.pt', 'wb') as f:
            f.write(model)
        self.model = Net2nn()
        self.model.load_state_dict(torch.load(f'./client_models/{self.id}.pt'))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        print('Received model')
        await self.start_training()

    async def start_training(self):
        train_ds = TensorDataset(self.train_imgs, self.train_labels)
        train_dl = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True)

        valid_ds = TensorDataset(self.valid_imgs, self.valid_labels)
        valid_dl = DataLoader(valid_ds, batch_size=self.BATCH_SIZE * 2)

        test_ds = TensorDataset(self.test_imgs, self.test_labels)
        test_dl = DataLoader(test_ds, batch_size=self.BATCH_SIZE * 2)

        print("------ Training ------")
        for epoch in range(self.EPOCHS):
            train_loss, train_accuracy = train(self.model, train_dl, self.criterion, self.optimizer)
            test_loss, test_accuracy = validation(self.model, test_dl, self.criterion)
            
            print("epoch: {:3.0f}".format(epoch+1) + " | train accuracy: {:7.4f}".format(train_accuracy) + " | test accuracy: {:7.4f}".format(test_accuracy))
        print("------ Training finished ------")

        await self.send_model()
    
    async def send_model(self):
        print('Sending model to server')
        await self.websocket.send('model_update')
        torch.save(self.model.state_dict(), f'./client_models/{self.id}.pt')
        await self.websocket.send(open(f'./client_models/{self.id}.pt', 'rb').read())

    
    def load_data(self):
        with gzip.open('./data/mnist/mnist.pkl.gz', "rb") as f:
            ((self.train_imgs, self.train_labels), (self.valid_imgs, self.valid_labels), (self.test_imgs, self.test_labels)) = pickle.load(f, encoding="latin-1")

        self.format_data()

        self.train_imgs, self.train_labels, self.valid_imgs, self.valid_labels, self.test_imgs, self.test_labels = map(torch.tensor, (self.train_imgs, self.train_labels, self.valid_imgs, self.valid_labels, self.test_imgs, self.test_labels))
        self.train_imgs = self.train_imgs[(self.client_number - 1) * self.train_amount : self.client_number * self.train_amount]
        self.train_labels = self.train_labels[(self.client_number - 1) * self.train_amount : self.client_number * self.train_amount]
        self.valid_imgs = self.valid_imgs[(self.client_number - 1) * self.valid_amount : self.client_number * self.valid_amount]
        self.valid_labels = self.valid_labels[(self.client_number - 1) * self.valid_amount : self.client_number * self.valid_amount]
        self.test_imgs = self.test_imgs[(self.client_number - 1) * self.test_amount : self.client_number * self.test_amount]
        self.test_labels = self.test_labels[(self.client_number - 1) * self.test_amount : self.client_number * self.test_amount]

        print(f'Client {self.client_number} has {self.train_imgs.shape[0]} train images')
        print(f'Client {self.client_number} has {self.valid_imgs.shape[0]} validation images')
        print(f'Client {self.client_number} has {self.test_imgs.shape[0]} test images')

    def format_data(self):
        self.train_amount = 4500
        self.valid_amount = 900
        self.test_amount = 900
        self.number_of_samples = 10

        self.label_dict_train=split_and_shuffle_labels(y_data=self.train_labels, seed=1, amount=self.train_amount) 
        self.sample_dict_train=get_iid_subsamples_indices(label_dict=self.label_dict_train, number_of_samples=self.number_of_samples, amount=self.train_amount)
        self.x_train_dict, self.y_train_dict = create_iid_subsamples(sample_dict=self.sample_dict_train, x_data=self.train_imgs, y_data=self.train_labels, x_name="train_imgs", y_name="train_labels")

        self.label_dict_valid = split_and_shuffle_labels(y_data=self.valid_labels, seed=1, amount=self.train_amount) 
        self.sample_dict_valid = get_iid_subsamples_indices(label_dict=self.label_dict_valid, number_of_samples=self.number_of_samples, amount=self.valid_amount)
        self.x_valid_dict, self.y_valid_dict = create_iid_subsamples(sample_dict=self.sample_dict_valid, x_data=self.valid_imgs, y_data=self.valid_labels, x_name="valid_imgs", y_name="valid_labels")

        self.label_dict_test = split_and_shuffle_labels(y_data=self.test_labels, seed=1, amount=self.test_amount) 
        self.sample_dict_test = get_iid_subsamples_indices(label_dict=self.label_dict_test, number_of_samples=self.number_of_samples, amount=self.test_amount)
        self.x_test_dict, self.y_test_dict = create_iid_subsamples(sample_dict=self.sample_dict_test, x_data=self.test_imgs, y_data=self.test_labels, x_name="test_images", y_name="test_labels")




client = Client()
asyncio.get_event_loop().run_until_complete(client.handler())