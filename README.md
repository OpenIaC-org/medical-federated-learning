# Federated Learning using Docker
This repository contains an implementation of the federated avarage algorithm using Docker. The MNIST data set is used and split over the three clients. Each client gets commands from a central server to train and return an updated model. Finally, the three models are combined and a test is run on the central server.

## How to run
1. Ensure you have Docker and Docker Compose installed
2. In the root folder run `docker-compose build`
3. In the root folder run `docker-compose up`

## Benchmarks
This table shows beanch marks with different number of training samples on each client. 
|Training Set Size|Time|Test Accuracy|Test Loss|
|--------|---------|---------|--------|
|4500|7.7204|0.936|0.2225|
|7500|10.864|0.9518|0.1625|
|20000|17.415|0.9694|0.1022|
