FROM ubuntu:latest

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.9 python3-pip python3.9-dev

COPY requirements.txt /server/

RUN pip install --no-cache-dir -r /server/requirements.txt

ADD . /server/

WORKDIR /server/

RUN mkdir client_models