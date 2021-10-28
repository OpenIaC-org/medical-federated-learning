FROM python:latest

COPY requirements.txt /server/

RUN pip install -r /server/requirements.txt

ADD * /server/

WORKDIR /server/

RUN python server.py
