# coding: utf-8
# !/usr/bin/env python
from torch.utils.data import DataLoader
import numpy as np
import argparse
import math
import os
import logging
import matplotlib.pyplot as plt

from train_utils import data_utils
from train_utils import training_routine
from data_select_utils import select_dataset
from logging_utils import logging_util

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for different experiments')
    parser.add_argument('-d', '--dataset_name', type=str, default='mnist', required=False,
                        help='Select the dataset, default is: mnist')
    parser.add_argument('-m', '--model_name', type=str, default='standard', required=False,
                        help='Select the model, default is standard')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, required=False,
                        help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-03, required=False,
                        help='Learning rate')
    parser.add_argument('-ep', '--epochs', type=int, default=10, required=False,
                        help='Number of epochs')
    parser.add_argument('-l', '--layers', type=int, default=3, required=False,
                        help='Number of layers')
    parser.add_argument('-met', '--metric', type=str, default='eu', required=False,
                        help='Distance metric to be used')
    parser.add_argument('-num_n', '--num_of_neighbors', type=int, default='50', required=False,
                        help='Number of neighbours to be used for knn and selective sampling, just used as a filler')
    parser.add_argument('-sam', '--sampling', type=str, default='random', required=False,
                        help='selective or random')
    args = parser.parse_args()
    return args


def main(args):
    data, labels = select_dataset(args.dataset_name, testing=False)
    n = data.shape[0]
    dim = data.shape[1]

    hl_size = np.int(np.ceil(120 + 8 * dim * math.log2(n)))
    number_of_triplets = np.int(np.ceil(7 * n * dim * math.log2(n)))
    epochs = args.epochs
    layers = args.layers
    learning_rate = args.learning_rate

    triplet_indices = data_utils.TripletDataset(data=data,
                                                labels=labels,
                                                num_trips=number_of_triplets,
                                                metric=args.metric,
                                                num_n=args.num_of_neighbors,
                                                batch_size=args.batch_size)
    all_triplets = triplet_indices.trips_data  # For computing triplet error
    batch_triplet_indices_loader = DataLoader(triplet_indices, batch_size=args.batch_size, shuffle=True, num_workers=4)
    loader = {'random': batch_triplet_indices_loader}

    # training the neural networks and get the embeddings
    experiment_name = args.dataset_name + \
                      '_model_' + args.model_name + \
                      '_sampling_' + str(args.sampling) + \
                      '_metric_' + str(args.metric) + '_' + str(args.num_of_neighbors) + 'n_' \
                      '_layers_' + str(layers) + \
                      '_dimension_' + str(dim) + \
                      '_bs_' + str(args.batch_size) + \
                      '_lr_' + str(learning_rate) + \
                      '_hl_size_' + str(hl_size) + \
                      '_epochs_' + str(epochs) + \
                      '_num_trips_' + str(number_of_triplets) + \
                      '_num_samples_' + str(n)

    os.makedirs('', mode=0o777, exist_ok=True)
    logging_path = '2d_reconstructions/' + experiment_name + '.log'
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiments: ' + experiment_name)

    embedding_final = training_routine.create_and_train_triplet_network(experiment_name=experiment_name,
                                                                        model_name='standard',
                                                                        dataset_name=args.dataset_name,
                                                                        ind_loaders=loader,
                                                                        n=n,
                                                                        dim=dim,
                                                                        layers=layers,
                                                                        learning_rate=learning_rate,
                                                                        epochs=epochs,
                                                                        hl_size=hl_size,
                                                                        batch_size=n,
                                                                        number_of_triplets=number_of_triplets,
                                                                        logger=logger)

    # compute the triplet error
    train_trip_error, _ = data_utils.triplet_error(embedding_final, all_triplets)
    logger.info('Training triplet error: ' + str(train_trip_error))

    random_triplet_indices = data_utils.gen_triplet_indices(n, 100000)
    random_triplets = data_utils.gen_triplet_data(data, random_triplet_indices, batch_size=10000)
    test_trip_error, _ = data_utils.triplet_error(embedding_final, random_triplets)
    logger.info('Generalization triplet error: ' + str(test_trip_error))

    # plot the stuff for a nice visualization
    colors_list = ['g', 'r', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    target_names = np.unique(labels)
    colors = colors_list[0:len(target_names)]
    target_ids = range(1, 2)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Reconstruction in 2D')

    # Plot the original figure colored by labels
    for i, c, label in zip(target_ids, colors, target_names):
        ax1.scatter(data[labels == i, 0], data[labels == i, 1], c=c, s=0.002, label=label)

    # Plot the embeddings colored by labels
    for i, c, label in zip(target_ids, colors, target_names):
        ax2.scatter(embedding_final[labels == i, 0], embedding_final[labels == i, 1], c=c, s=0.001, label=label)

    plt.show()
    fig.savefig(os.path.join('', experiment_name + '.eps'), format='eps')

if __name__ == "__main__":
    main(parse_args())



