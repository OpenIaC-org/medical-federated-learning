import asyncio
import websockets
import sys
from enums import ClientStatus


class Client:
    def __init__(self):
        self.my_status = ClientStatus.NOT_CONNECTED
    
    async def handler(self):
        async with websockets.connect('ws://localhost:8000', ping_interval=None) as websocket:
            while True:
                if self.my_status == ClientStatus.NOT_CONNECTED:
                    await self.connect(websocket)
                elif self.my_status == ClientStatus.CONNECTED:
                    data = await websocket.recv()
                    if data == 'data':
                        await self.receive_data(websocket)

    
    async def connect(self, websocket):
        await websocket.send('connect')
        res = await websocket.recv()
        self.id = str(res)
        self.my_status = ClientStatus.CONNECTED
        print('Connected to server. My Id is ' + self.id)
    
    async def receive_data(self, websocket):
        res = await websocket.recv()
        print(res)


client = Client()
asyncio.get_event_loop().run_until_complete(client.handler())