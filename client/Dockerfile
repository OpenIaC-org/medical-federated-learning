FROM ubuntu:latest

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.9 python3-pip python3.9-dev

COPY requirements.txt /client/

RUN pip install --no-cache-dir -r /client/requirements.txt

ADD . /client/

WORKDIR /client/

RUN mkdir models