# coding: utf-8
# !/usr/bin/env python
import torch
import torch.optim
import torch.nn as nn
from torch.nn.parallel import DataParallel
from train_utils import data_utils
import math
import time
import os
import sys
import numpy as np
from models.models import standard_model


def define_model(model_name, digits, hl_size, dim, layers):
    # Constructing the Network
    if model_name == 'standard':
        emb_net = standard_model(digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    return emb_net


def create_and_train_triplet_network(experiment_name, model_name, dataset_name, logger, number_of_triplets,
                                     ind_loaders, n, dim, layers,
                                     learning_rate=5e-2, epochs=10,
                                     hl_size=100, batch_size=10000):
    """
    Description: Constructs the OENN network, defines an optimizer and trains the network on the data w.r.t triplet loss.
    :param experiment_name:
    :param model_name:
    :param dataset_name:
    :param ind_loader_selective:
    :param ind_loader_random:
    :param n: # points
    :param dim: # features/ dimensions
    :param layers: # layers
    :param learning_rate: learning rate of optimizer.
    :param epochs: # epochs
    :param hl_size: # width of the hidden layer
    :param batch_size: # batch size for training
    :param logger: # for logging
    :param number_of_triplets: #TODO
    :return:
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    digits = int(math.ceil(math.log2(n)))
    emb_net = define_model(model_name=model_name, digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    emb_net = emb_net.to(device)
    print(emb_net)
    print(digits)
    if torch.cuda.device_count() > 1:
        emb_net = DataParallel(emb_net)
        print('multi-gpu')

    # Optimizer
    optimizer = torch.optim.Adam(emb_net.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    criterion = criterion.to(device)

    logger.info('#### Dataset Selection #### \n')
    logger.info('#### Network and learning parameters #### \n')
    logger.info('------------------------------------------ \n')
    logger.info('Model Name: ' + model_name + '\n')
    logger.info('Number of hidden layers: ' + str(layers) + '\n')
    logger.info('Hidden layer width: ' + str(hl_size) + '\n')
    logger.info('Batch size: ' + str(batch_size) + '\n')
    logger.info('Embedding dimension: ' + str(dim) + '\n')
    logger.info('Learning rate: ' + str(learning_rate) + '\n')
    logger.info('Number of epochs: ' + str(epochs) + '\n')
    logger.info('Number of triplets: ' + str(number_of_triplets) + '\n')
    logger.info(' #### Training begins #### \n')
    logger.info('---------------------------\n')

    # Training begins
    train_time = 0
    for ep in range(epochs):
        # Epoch is one pass over the dataset
        epoch_loss = 0
        epoch_length = 0
        print('Digits: ', digits)
        for each_loader_type in ind_loaders.keys():
            ind_loader = ind_loaders[each_loader_type]
            logger.info('Going through:' + str(each_loader_type) + ' triplets')

            for batch_ind, trips in enumerate(ind_loader):
                sys.stdout.flush()
                trip = trips.squeeze().to(device).float()

                # Training time
                begin_train_time = time.time()
                # Forward pass
                embedded_a = emb_net(trip[:, :digits])
                embedded_p = emb_net(trip[:, digits:2 * digits])
                embedded_n = emb_net(trip[:, 2 * digits:])
                # Compute loss
                loss = criterion(embedded_a, embedded_p, embedded_n).to(device)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # End of training
                end_train_time = time.time()
                if batch_ind % 10 == 0:
                    logger.info('Epoch: ' + str(ep) + ' Mini batch: ' + str(batch_ind) +
                                '/' + str(len(ind_loader)) + ' Loss: ' + str(loss.item()))
                    sys.stdout.flush()  # Prints faster to the out file
                epoch_loss += loss.item()
                train_time = train_time + end_train_time - begin_train_time
            epoch_length += len(ind_loader)

        logger.info('Epoch: ' + str(ep) + ' - Average Epoch Loss:  ' + str(epoch_loss/epoch_length) +
                     ', Training time ' + str(train_time))
        sys.stdout.flush()  # Prints faster to the out file

        # Saving the results
        logger.info('Saving the models and the results')
        sys.stdout.flush()  # Prints faster to the out file

        experiment_name = experiment_name

        os.makedirs('checkpoints', mode=0o777, exist_ok=True)
        model_path = 'checkpoints/' + \
                     experiment_name + \
                     '.pt'
        torch.save({
            'epochs': ep,
            'model_state_dict': emb_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss:': epoch_loss,
        }, model_path)

        # Training time
        os.makedirs('training_time', mode=0o777, exist_ok=True)
        train_time_path = 'training_time/' + \
                          experiment_name + \
                          '.npy'
        np.save(train_time_path, train_time)

    # Compute the embedding of the data points.
    bin_array = data_utils.get_binary_array(n, digits)
    data_bin = torch.Tensor(bin_array).to(device)
    embeddings = emb_net(data_bin)  # Feed the binary array of indices to the network and generate embeddings: FP
    embedding_final = embeddings.cpu().detach().numpy()

    # Save the embeddings, triplet error and training_time separately for later use.
    # Embeddings
    os.makedirs('embeddings', mode=0o777, exist_ok=True)  # Create writable directory if it doesn't exist.
    embedding_path = 'embeddings/' + \
                     experiment_name + \
                     '.npy'
    np.save(embedding_path, embedding_final)

    return embedding_final


def create_and_train_triplet_network_withtime(experiment_name, model_name, dataset_name, logger, number_of_triplets,
                                     ind_loaders, n, dim, layers,
                                     learning_rate=5e-2, epochs=10,
                                     hl_size=100, batch_size=10000):
    """
    Description: Constructs the OENN network, defines an optimizer and trains the network on the data w.r.t triplet loss.
    :param experiment_name:
    :param model_name:
    :param dataset_name:
    :param ind_loader_selective:
    :param ind_loader_random:
    :param n: # points
    :param dim: # features/ dimensions
    :param layers: # layers
    :param learning_rate: learning rate of optimizer.
    :param epochs: # epochs
    :param hl_size: # width of the hidden layer
    :param batch_size: # batch size for training
    :param logger: # for logging
    :param number_of_triplets: #TODO
    :return:
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    digits = int(math.ceil(math.log2(n)))

    emb_net = define_model(model_name=model_name, digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    emb_net = emb_net.to(device)

    if torch.cuda.device_count() > 1:
        emb_net = DataParallel(emb_net)
        print('multi-gpu')

    # Optimizer
    optimizer = torch.optim.Adam(emb_net.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    criterion = criterion.to(device)

    logger.info('#### Dataset Selection #### \n')
    logger.info('dataset:', dataset_name)
    logger.info('#### Network and learning parameters #### \n')
    logger.info('------------------------------------------ \n')
    logger.info('Model Name: ' + model_name + '\n')
    logger.info('Number of hidden layers: ' + str(layers) + '\n')
    logger.info('Hidden layer width: ' + str(hl_size) + '\n')
    logger.info('Batch size: ' + str(batch_size) + '\n')
    logger.info('Embedding dimension: ' + str(dim) + '\n')
    logger.info('Learning rate: ' + str(learning_rate) + '\n')
    logger.info('Number of epochs: ' + str(epochs) + '\n')
    logger.info('Number of triplets: ' + str(number_of_triplets) + '\n')
    logger.info(' #### Training begins #### \n')
    logger.info('---------------------------\n')

    # Training begins
    train_time = 0
    for ep in range(epochs):
        # Epoch is one pass over the dataset
        epoch_loss = 0
        for each_loader_type in ind_loaders.keys():
            ind_loader = ind_loaders[each_loader_type]
            logger.info('Going through:' + str(each_loader_type) + ' triplets')

            for batch_ind, trips in enumerate(ind_loader):
                sys.stdout.flush()
                trip = trips.squeeze().to(device).float()

                # Training time
                begin_train_time = time.time()
                # Forward pass
                embedded_a = emb_net(trip[:, :digits])
                embedded_p = emb_net(trip[:, digits:2 * digits])
                embedded_n = emb_net(trip[:, 2 * digits:])
                # Compute loss
                loss = criterion(embedded_a, embedded_p, embedded_n).to(device)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # End of training
                end_train_time = time.time()
                if batch_ind % 100 == 0:
                    logger.info('Epoch: ' + str(ep) + ' Mini batch: ' + str(batch_ind) +
                                '/' + str(len(ind_loader)) + ' Loss: ' + str(loss.item()))
                    sys.stdout.flush()  # Prints faster to the out file
                epoch_loss += loss.item()
                train_time = train_time + end_train_time - begin_train_time

        # Saving the results
        logger.info('Saving the models and the results')
        sys.stdout.flush()  # Prints faster to the out file

        experiment_name = experiment_name

        os.makedirs('checkpoints', mode=0o777, exist_ok=True)
        model_path = 'checkpoints/' + \
                     experiment_name + \
                     '.pt'
        torch.save({
            'epochs': ep,
            'model_state_dict': emb_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss:': epoch_loss,
        }, model_path)

        # Training time
        os.makedirs('training_time', mode=0o777, exist_ok=True)
        train_time_path = 'training_time/' + \
                          experiment_name + \
                          '.npy'
        np.save(train_time_path, train_time)

    # Compute the embedding of the data points.
    bin_array = data_utils.get_binary_array(n, digits)
    data_bin = torch.Tensor(bin_array).to(device)
    embeddings = emb_net(data_bin)  # Feed the binary array of indices to the network and generate embeddings: FP
    embedding_final = embeddings.cpu().detach().numpy()

    # Save the embeddings, triplet error and training_time separately for later use.
    # Embeddings
    os.makedirs('embeddings', mode=0o777, exist_ok=True)  # Create writable directory if it doesn't exist.
    embedding_path = 'embeddings/' + \
                     experiment_name + \
                     '.npy'
    np.save(embedding_path, embedding_final)

    return embedding_final, train_time