# coding: utf-8
# !/usr/bin/env python
from torch.utils.data import DataLoader
import numpy as np
import argparse
import math
import os
import sys
import logging

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
    parser.add_argument('-bs', '--batch_size', type=int, default=5000, required=False,
                        help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-03, required=False,
                        help='Learning rate')
    parser.add_argument('-ep', '--epochs', type=int, default=500, required=False,
                        help='Number of epochs')
    parser.add_argument('-l', '--layers', type=int, default=3, required=False,
                        help='Number of layers')
    parser.add_argument('-min_hl', '--min_hl', type=int, default=0, required=False,
                        help='min size of the hidden layer')
    parser.add_argument('-max_hl', '--max_hl', type=int, default=2, required=False,
                        help='max size of the hidden layer')
    parser.add_argument('-minD', '--min_dim', type=int, default=10, required=False,
                        help='Minimum embedding dimension')
    parser.add_argument('-maxD', '--max_dim', type=int, default=10, required=False,
                        help='Maximum embedding dimension')
    parser.add_argument('-tr', '--triplet_constant', type=float, required=False, default=1,
                        help='Scaling factor, number of triplets')
    parser.add_argument('-min_n', '--min_n', type=int, default=1000, required=True,
                        help='Minimum number of points')
    parser.add_argument('-max_n', '--max_n', type=int, default=1000, required=True,
                        help='Maximum number of points')
    parser.add_argument('-reps', '--repetitions', type=int, default=6, required=False,
                        help='Number of times to repeat the experiment')
    args = parser.parse_args()
    return args


def main(args):

    n_range = 2 ** np.arange(args.min_n, args.max_n + 1)  # Range of number of points to experiment over.
    dim_range = np.arange(args.min_dim, args.max_dim + 1, 5)  # Range of dimensions to experiment over.
    hl_range = 100 + 25*np.arange(args.min_hl, args.max_hl + 1)  # Range of hl_size
    reps = args.repetitions

    train_error_log_scores = np.ones(shape=(n_range.shape[0], dim_range.shape[0], hl_range.shape[0], reps), dtype=np.float32)

    n_counter = 0
    for n in n_range:
        dim_counter = 0
        for dim in dim_range:
            hl_counter = 0
            for hl_size in hl_range:
                # get the dataset you want and we need to keep track of the points, dimensions, hl_size
                vec_data, labels = select_dataset(dataset_name=args.dataset_name, input_dim=dim, n=n)
                n = vec_data.shape[0]

                # hl_size = int(120 + (1 * args.hl_scale * dim * math.log2(n)))  # Hidden layer size
                number_of_triplets = np.int(np.ceil(args.triplet_constant * n * math.log2(n) * dim))  # triplets
                print('Trips: ', number_of_triplets)
                # If sampling is random
                triplet_indices = data_utils.TripletDataset(data=vec_data,
                                                            labels=labels,
                                                            num_trips=number_of_triplets,
                                                            batch_size=args.batch_size,
                                                            metric='eu',
                                                            num_n=50)

                all_triplets = triplet_indices.trips_data  # For computing triplet error
                batch_triplet_indices_loader = DataLoader(dataset=triplet_indices,
                                                          batch_size=args.batch_size,
                                                          shuffle=True,
                                                          num_workers=8)

                triplet_loaders = {'random': batch_triplet_indices_loader}

                experiment_name = args.dataset_name + \
                                  '_n_' + str(n) + \
                                  '_model_' + args.model_name + \
                                  '_layers_' + str(args.layers) + \
                                  '_dimension_' + str(dim) + \
                                  '_bs_' + str(args.batch_size) + \
                                  '_lr_' + str(args.learning_rate) + \
                                  '_hl_size_' + str(hl_size) + \
                                  '_epochs_' + str(args.epochs) + \
                                  '_num_trips_' + str(number_of_triplets)

                os.makedirs('hyper_logs', mode=0o777, exist_ok=True)
                logging_path = 'hyper_logs/' + experiment_name + '.log'
                logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
                logger.info('Name of Experiments: ' + experiment_name)

                for rep_id in range(reps):
                    logger.info('n is ' + str(n) + '\n')
                    logger.info('dim is ' + str(dim) + '\n')
                    logger.info('rep is ' + str(rep_id) + '\n')
                    sys.stdout.flush()

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

                    train_trip_error, _ = data_utils.triplet_error(embedding_final, all_triplets)
                    logger.info('Training triplet error: ' + str(train_trip_error))

                    train_error_log_scores[n_counter, dim_counter, hl_counter, rep_id] = train_trip_error

                    matrix_saving_path = '_min_n_' + str(min(n_range)) + \
                                         '_max_n_' + str(max(n_range)) + \
                                         '_min_dim_' + str(min(dim_range)) + \
                                         '_max_dim_' + str(max(dim_range)) + \
                                         '_min_hlrange_' + str(min(hl_range)) + \
                                         '_max_hlrange_' + str(max(hl_range)) + \
                                         '_reps_' + str(args.repetitions)

                    os.makedirs('hyper_maps', mode=0o777, exist_ok=True)
                    np.save('hyper_maps/' + args.dataset_name + matrix_saving_path + '_' + 'train_error.npy', train_error_log_scores)

                hl_counter += 1
            dim_counter += 1
        n_counter += 1


if __name__ == "__main__":
    main(parse_args())
