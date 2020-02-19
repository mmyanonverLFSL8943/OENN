# coding: utf-8
# !/usr/bin/env python

import numpy as np
import argparse
import math
import os
import logging

from train_utils import data_utils
from train_utils import testing_routine
from data_select_utils import select_dataset
from data_select_utils import select_test_dataset
from logging_utils import logging_util
from torch.utils.data import DataLoader


def get_neighbours(dataset_name):
    file_name = os.path.join('indices',
                             dataset_name + '_neigbours_indices.npy')
    indices = np.load(file_name)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for different experiments')

    parser.add_argument('-mp', '--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('-d', '--dataset', type=str, default='standard', required=False,
                        help='Select the dataset, default is mnist')
    parser.add_argument('-m', '--model_name', type=str, default='standard', required=False,
                        help='Select the model, default is standard')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-03, required=False,
                        help='Learning rate')
    parser.add_argument('-ep', '--epochs', type=int, default=10, required=False,
                        help='Number of epochs')
    parser.add_argument('-l', '--layers', type=int, default=3, required=False,
                        help='Number of layers')
    parser.add_argument('-bs', '--batch_size', type=int, default=50000, required=False,
                        help='Batch size')
    parser.add_argument('-hl', '--hl_scale', type=int, default=2, required=False,
                        help='Scaling factor, Size of the hidden layer')
    parser.add_argument('-dim', '--dimension', type=int, default=5, required=False,
                        help='Embedding dimension')
    parser.add_argument('-tr', '--num_triplets', type=float, required=False, default=1,
                        help='number of triplets')
    args = parser.parse_args()
    return args


def main(args):
    # get the dataset you want
    vec_data, labels = select_dataset(args.dataset)

    # rand_indices = np.random.permutation(len(vec_data.shape[0]))
    # train_data = vec_data[rand_indices[300:, :]]
    # test_data = vec_data[rand_indices[0:300, :]]
    test_data, test_labels = select_test_dataset(args.dataset)
    # test_data = test_data
    number_of_triplets = np.int(args.num_triplets)

    triplet_indices = data_utils.TripletDataset(data=vec_data, labels=labels,
                                                num_trips=number_of_triplets,
                                                batch_size=args.batch_size,
                                                metric='eu',
                                                num_n=50, test=True, test_data=test_data)
    unseen_triplets = triplet_indices.trips_data  # For computing triplet error

    batch_triplet_indices_loader = DataLoader(triplet_indices, batch_size=args.batch_size, shuffle=True,
                                              num_workers=8)

    # unseen_triplets = data_utils.gen_triplet_data_unseen(vec_data, test_data, number_of_triplets)
    test_n = test_data.shape[0]

    # compute the number of data samples
    n = vec_data.shape[0]
    dim = args.dimension

    hl_size = int(120 + (args.hl_scale * dim * math.log2(n)))  # Hidden layer size

    experiment_name = args.dataset + \
                      '_model_' + args.model_name + \
                      '_layers_' + str(args.layers) + \
                      '_dimension_' + str(dim) + \
                      '_testing_pts_' + str(test_n) + \
                      '_lr_' + str(args.learning_rate) + \
                      '_bs_' + str(args.batch_size) + \
                      '_hl_size_' + str(hl_size) + \
                      '_epochs_' + str(args.epochs) + \
                      '_num_trips_' + str(number_of_triplets) + '_testing_'

    os.makedirs('test_logs', mode=0o777, exist_ok=True)
    logging_path = 'test_logs/' + experiment_name + '.log'
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiments: ' + experiment_name)

    testing_routine.create_and_test_triplet_network(experiment_name=experiment_name,
                                                    batch_triplet_indices_loader=batch_triplet_indices_loader,
                                                    path_to_emb_net=args.model_path,
                                                    unseen_triplets=unseen_triplets,
                                                    dataset_name=args.dataset,
                                                    model_name=args.model_name,
                                                    logger=logger,
                                                    test_n=test_n,
                                                    n=n,
                                                    dim=dim,
                                                    layers=args.layers,
                                                    learning_rate=args.learning_rate,
                                                    epochs=args.epochs,
                                                    hl_size=hl_size
                                                    )


if __name__ == "__main__":
    main(parse_args())
