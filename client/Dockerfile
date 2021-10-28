FROM python:latest

COPY requirements.txt /server/

RUN pip install -r /server/requirements.txt


ADD * /client/

WORKDIR /client/