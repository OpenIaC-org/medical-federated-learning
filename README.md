# Federated Learning using Docker
This repository contains an implementation of the federated avarage algorithm using Docker. The MNIST data set is used and split over the three clients. Each client gets commands from a central server to train and return an updated model. Finally, the three models are combined and a test is run on the central server.

## How to run
1. Ensure you have Docker and Docker Compose installed
2. In the root folder run `docker-compose build`
3. In the root folder run `docker-compose up`

## Benchmarks
This table shows beanch marks with different number of training samples. The values are averaged over several runs for stability.

### Federated
Training set size is per client.
|Training Set Size|Time|Test Accuracy|Test Loss|
|--------|---------|---------|--------|
|4500|7.7204|0.936|0.2225|
|7500|10.864|0.9518|0.1625|
|20000|17.415|0.9694|0.1022|

### Local training with single processor
Training set size is 3x that of federated for comparison when there are three clients.
|Training Set Size|Time|Test Accuracy|Test Loss|
|--------|---------|---------|--------|
|13500|6.3617|0.9515|0.1467|
|22500|10.240|0.9689|0.0826|
|60000|22.375|0.98|0.6937|
