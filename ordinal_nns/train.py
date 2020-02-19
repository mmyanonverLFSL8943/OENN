# coding: utf-8
# !/usr/bin/env python
# Imports
from torch.utils.data import DataLoader
import numpy as np
import argparse
import math
import os
import logging
import sys
from distutils import util

from train_utils import data_utils
from train_utils import training_routine
from data_select_utils import select_dataset
from logging_utils import logging_util


def get_neighbours(dataset_name):
    """
    Generating nearest neighbors on CPU typically takes quite long. In order to facilitate quick experimentation,
    we save the nearest neighbor indices for datasets and load them during experiments.
    :param dataset_name: Name of the dataset. For options, see data_select_utils.py or the documentation.
    :return: nearest neighbor indices for the given dataset.
    """
    file_name = os.path.join('indices',
                             dataset_name + '_neigbours_indices.npy')
    indices = np.load(file_name)
    return indices


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
    parser.add_argument('-hl', '--hl_scale', type=int, default=5, required=False,
                        help='Size of the hidden layer - 1 to 5 (from small to large)')
    parser.add_argument('-minD', '--min_dim', type=int, default=5, required=False,
                        help='Minimum embedding dimension')
    parser.add_argument('-maxD', '--max_dim', type=int, default=10, required=False,
                        help='Maximum embedding dimension')
    parser.add_argument('-tr', '--triplet_constant', type=float, required=False, default=1,
                        help='Scaling factor, number of triplets')
    parser.add_argument('-sam', '--sampling', type=str, default='random', required=False,
                        help='selective or random')
    parser.add_argument('-sr', '---sampling_ratio', type=float, default=1.0, required=False,
                        help='selective or random')
    parser.add_argument('-nn', '--compute_nn', default='False', help='Compute Nearest Neigbours or Not')
    parser.add_argument('-num_n', '--num_of_neighbors', type=int, default=50, help='select the number of neigbours')
    parser.add_argument('-met', '--metric', default='eu', help='Metric eu or cosine or knn')
    args = parser.parse_args()
    return args


def main(args):

    # get the dataset you want
    vec_data, labels = select_dataset(args.dataset_name)
    compute_nn = util.strtobool(args.compute_nn)

    # compute the number of data samples
    n = vec_data.shape[0]

    # Sampling can be chosen to be random or selective or mixed (a mix of both).

    # Random sampling procedure chooses the triplets completely randomly.

    # Selective sampling procedure refers to generating triplets (i, j, k) such that at least one of j or k arise
    # from the nearest neighbors of i.
    if args.sampling == 'mixed':
        # Sampling ratio determines the fraction of triplets that are chosen randomly (sampling_ratio = 1.0) vs
        # selectively
        assert(0 < args.sampling_ratio < 1)

    dim_range = np.array(range(args.min_dim, args.max_dim + 1))  # Range of dimensions to experiment over.
    for dim in dim_range:

        hl_size = int(120 + (1 * args.hl_scale * dim * math.log2(n)))  # Hidden layer size
        number_of_triplets = np.int(np.ceil(args.triplet_constant * n * math.log2(n) * dim))  # Number of triplets
        print(number_of_triplets)

        # select and define the params of the experiment
        if args.sampling == 'selective':
            # If sampling is selective
            num_neighbors = args.num_of_neighbors
            print('Sampling is chosen as selective')
            sys.stdout.flush()
            if compute_nn:
                # If neighbor indices are not saved then they are generated.
                neighbor_indices = data_utils.get_nearest_neighbors(vec_data, num_neighbors)
            else:
                print('Loading saved indices...')
                sys.stdout.flush()
                neighbor_indices = get_neighbours(args.dataset_name)

            triplet_indices = data_utils.SelectiveTripletDataset(data=vec_data, labels=labels,
                                                                 num_trips=number_of_triplets,
                                                                 batch_size=args.batch_size,
                                                                 neighbor_indices=neighbor_indices,
                                                                 metric=args.metric, num_n=args.num_of_neighbors)

            all_triplets = triplet_indices.trips_data  # For computing triplet error
            batch_triplet_indices_loader = DataLoader(dataset=triplet_indices,
                                                      batch_size=args.batch_size,
                                                      shuffle=True, num_workers=4)
            triplet_loaders = {'selective': batch_triplet_indices_loader}
        elif args.sampling == 'mixed':
            # If sampling is mixed
            num_selective_triplets = np.int(np.ceil((1 - args.sampling_ratio) * number_of_triplets))  # Number of selective trips
            num_random_triplets = np.int(np.ceil(args.sampling_ratio * number_of_triplets))  # Number of random trips

            num_neighbors = args.num_of_neighbors
            print('Sampling is chosen as Mixed')
            sys.stdout.flush()
            if compute_nn:
                neighbor_indices = data_utils.get_nearest_neighbors(vec_data, num_neighbors)
            else:
                print('Loading saved indices...')
                sys.stdout.flush()
                neighbor_indices = get_neighbours(args.dataset_name)

            # for selective sampling
            triplet_selective_indices = data_utils.SelectiveTripletDataset(data=vec_data, labels=labels,
                                                                           num_trips=num_selective_triplets,
                                                                           batch_size=args.batch_size,
                                                                           neighbor_indices=neighbor_indices,
                                                                           metric=args.metric,
                                                                           num_n=num_neighbors)

            all_selective_triplets = triplet_selective_indices.trips_data  # For computing triplet error
            batch_selective_triplet_loader = DataLoader(triplet_selective_indices, batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=4)

            # for random sampling
            triplet_random_indices = data_utils.TripletDataset(data=vec_data, labels=labels,
                                                               num_trips=num_random_triplets,
                                                               batch_size=args.batch_size,
                                                               metric=args.metric,
                                                               num_n=num_neighbors)
            all_random_triplets = triplet_random_indices.trips_data  # For computing triplet error
            batch_random_triplet_loader = DataLoader(triplet_random_indices, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=4)

            all_triplets = np.concatenate((all_random_triplets, all_selective_triplets), axis=0)
            triplet_loaders = {'random': batch_random_triplet_loader, 'selective': batch_selective_triplet_loader}

        else:
            # If sampling is random
            print('Sampling is chosen as random')
            sys.stdout.flush()

            num_neighbors = args.num_of_neighbors
            print('Data loader is being created')
            sys.stdout.flush()
            triplet_indices = data_utils.TripletDataset(data=vec_data, labels=labels,
                                                        num_trips=number_of_triplets,
                                                        batch_size=args.batch_size,
                                                        metric=args.metric,
                                                        num_n=num_neighbors)
            all_triplets = triplet_indices.trips_data  # For computing triplet error

            batch_triplet_indices_loader = DataLoader(triplet_indices, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=8)
            triplet_loaders = {'random': batch_triplet_indices_loader}

        experiment_name = args.dataset_name + \
                          '_model_' + args.model_name + \
                          '_sampling_' + args.sampling + '_' + str(args.sampling_ratio) + \
                          '_metric_' + args.metric + '_' + str(args.num_of_neighbors) + 'n_' \
                          '_layers_' + str(args.layers) + \
                          '_dimension_' + str(dim) + \
                          '_bs_' + str(args.batch_size) + \
                          '_lr_' + str(args.learning_rate) + \
                          '_hl_size_' + str(hl_size) + \
                          '_epochs_' + str(args.epochs) + \
                          '_num_trips_' + str(number_of_triplets)
        print(number_of_triplets)
        print(experiment_name)
        os.makedirs('logs', mode=0o777, exist_ok=True)
        logging_path = 'logs/' + experiment_name + '.log'
        logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
        logger.info('Name of Experiments: ' + experiment_name)

        embedding_final = training_routine.create_and_train_triplet_network(experiment_name=experiment_name,
                                                                            model_name=args.model_name,
                                                                            dataset_name=args.dataset_name,
                                                                            ind_loaders=triplet_loaders,
                                                                            n=n,
                                                                            dim=dim,
                                                                            layers=args.layers,
                                                                            learning_rate=args.learning_rate,
                                                                            epochs=args.epochs,
                                                                            hl_size=hl_size,
                                                                            batch_size=args.batch_size,
                                                                            number_of_triplets=number_of_triplets,
                                                                            logger=logger)

        random_triplet_indices = data_utils.gen_triplet_indices(n, 100000)
        random_triplets = data_utils.gen_triplet_data(vec_data, random_triplet_indices, 10000)
        test_trip_error, _ = data_utils.triplet_error(embedding_final, random_triplets)
        logger.info('Generalization triplet error: ' + str(test_trip_error))

        train_trip_error, _ = data_utils.triplet_error(embedding_final, all_triplets)
        logger.info('Training triplet error: ' + str(train_trip_error))


if __name__ == "__main__":
    main(parse_args())
