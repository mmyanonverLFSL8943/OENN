# coding: utf-8
# !/usr/bin/env python
from torch.utils.data import DataLoader
import numpy as np
import argparse
import math
import os
import logging
import sys
import PIL
import time
import matplotlib.pyplot as plt

from tste import *
from train_utils import data_utils
from train_utils import training_routine
from data_select_utils import select_dataset
from logging_utils import logging_util

def get_neighbours(dataset_name):
    file_name = os.path.join('indices',
                             dataset_name + '_neigbours_indices.npy')
    indices = np.load(file_name)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for different experiments')
    parser.add_argument('-d', '--dataset_name', type=str, default='cover_type', required=False,
                        help='Select the dataset, default is: mnist')
    parser.add_argument('-m', '--model_name', type=str, default='standard', required=False,
                        help='Select the model, default is standard')
    parser.add_argument('-bs', '--batch_size', type=int, default=1000, required=False,
                        help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-03, required=False,
                        help='Learning rate')
    parser.add_argument('-ep', '--epochs', type=int, default=100, required=False,
                        help='Number of epochs')
    parser.add_argument('-l', '--layers', type=int, default=3, required=False,
                        help='Number of layers')
    parser.add_argument('-hl', '--hl_scale', type=int, default=1, required=False,
                        help='Size of the hidden layer - 1 to 5 (from small to large)')
    parser.add_argument('-minD', '--min_dim', type=int, default=10, required=False,
                        help='Minimum embedding dimension')
    parser.add_argument('-maxD', '--max_dim', type=int, default=10, required=False,
                        help='Maximum embedding dimension')
    parser.add_argument('-tr', '--triplet_constant', type=float, required=False, default=1,
                        help='Scaling factor, number of triplets')
    parser.add_argument('-sam', '--sampling', type=str, default='random', required=False,
                        help='selective or random')
    parser.add_argument('-nn', '--compute_nn', default=False, help='Compute Nearest Neigbours or Not')
    parser.add_argument('-rs', '--range_scale', type=int, default=100, help='the scale multiplied by np.arange(1,11)')

    # parser.add_argument('-')
    args = parser.parse_args()
    return args


def main(args):
    # get the dataset you want
    vec_data_total, labels = select_dataset(args.dataset_name, testing=False)
    print(vec_data_total.shape)
    vec_data = vec_data_total

    # compute the number of data samples

    # dim_range = 5 * np.array(range(args.min_dim, args.max_dim + 1))  # Range of dimensions to experiment over.
    train_errors = list()
    test_errors = list()
    times = list()
    dim = args.min_dim

    n = vec_data_total.shape[0]
    hl_size = int(120 + (1 * args.hl_scale * dim * math.log2(n)))  # Hidden layer size
    number_of_triplets = np.int(np.ceil(args.triplet_constant * n * math.log2(n) * dim))  # Number of triplets

    # select and define the params of the experi
    print('Sampling is chosen as random')

    triplet_indices = data_utils.TripletDataset(vec_data, labels, number_of_triplets, args.batch_size, "Eucl", 1)
    all_triplets = triplet_indices.trips_data  # For computing triplet error

    quadruplets = np.zeros((all_triplets.shape[0], 4), dtype = int)
    quadruplets[:,0:2] = all_triplets[:,0:2]
    quadruplets[:, 3] = all_triplets[:, 2]
    quadruplets[:, 2] = all_triplets[:, 0]

    print(quadruplets.shape)
    np.savetxt("dataForLOE.csv", quadruplets, '%7.0f', delimiter=",", )
    begin = time.time()
    res = os.system('Rscript runLOE.R')
    print('LOE res=',res)
    time_LOE = time.time() - begin

    embedding_LOE = np.genfromtxt('loeEmb.csv', delimiter=',')
    embedding_LOE = embedding_LOE[1:,1:]
    print(data_utils.triplet_error(embedding_LOE, all_triplets))
    batch_triplet_indices_loader = DataLoader(triplet_indices, batch_size=args.batch_size, shuffle=True,
                                              num_workers=4)
    ind_loaders = {'random': batch_triplet_indices_loader}

    experiment_name = args.dataset_name + \
                      '_model_' + args.model_name + \
                      '_sampling_' + args.sampling + \
                      '_layers_' + str(args.layers) + \
                      '_dimension_' + str(dim) + \
                      '_bs_' + str(args.batch_size) + \
                      '_lr_' + str(args.learning_rate) + \
                      '_hl_size_' + str(hl_size) + \
                      '_epochs_' + str(args.epochs) + \
                      '_num_trips_' + str(number_of_triplets)

    print(all_triplets.shape)
    sys.stdout.flush()
    begin = time.time()
    embedding_TSTE = tste(all_triplets, no_dims=dim, alpha=1)
    print(data_utils.triplet_error(embedding_TSTE, all_triplets))
    time_TSTE = time.time() - begin

    os.makedirs('comparison-results', mode=0o777, exist_ok=True)
    os.makedirs('logs', mode=0o777, exist_ok=True)
    logging_path = 'logs/' + experiment_name + '.log'
    logger = logging_util.my_custom_logger(logger_name=logging_path, level=logging.INFO)
    logger.info('Name of Experiments: ' + experiment_name)

    embedding_OENN, time_OENN = training_routine.create_and_train_triplet_network_withtime(
        experiment_name=experiment_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        ind_loaders=ind_loaders,
        n=n,
        dim=dim,
        layers=args.layers,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hl_size=hl_size,
        batch_size=args.batch_size,
        number_of_triplets=number_of_triplets,
        logger=logger)
    # times.append([time_OENN, time_TSTE, time_LOE])
    print('time', times)
    random_triplet_indices = data_utils.gen_triplet_indices(n, 10000)
    random_triplets = data_utils.gen_triplet_data(vec_data, random_triplet_indices, 10000)
    test_trip_error_OENN, _ = data_utils.triplet_error(embedding_OENN, random_triplets)
    test_trip_error_TSTE, _ = data_utils.triplet_error(embedding_TSTE, random_triplets)
    print(test_trip_error_OENN, test_trip_error_TSTE)

    # test_trip_error_LOE, _ = data_utils.triplet_error(embedding_LOE, random_triplets)

    # test_errors.append([test_trip_error_OENN, test_trip_error_TSTE, test_trip_error_LOE])

    # print('test_trip',test_errors)

    # logger.info('Generalization triplet error: ' + str(test_trip_error))

    train_trip_error_OENN, _ = data_utils.triplet_error(embedding_OENN, all_triplets)
    train_trip_error_TSTE, _ = data_utils.triplet_error(embedding_TSTE, all_triplets)
    print(train_trip_error_OENN, train_trip_error_TSTE)
    # train_trip_error_LOE, _ = data_utils.triplet_error(embedding_LOE, all_triplets)

    # train_errors.append([train_trip_error_OENN, train_trip_error_TSTE, train_trip_error_LOE])
    # logger.info('Training triplet error: ' + str(train_trip_error))
    # print('train_trip',train_errors)
    # file = 'comparison-results/' + args.dataset_name + '-OENN-TSTE-LOE.npz'
    # np.savez_compressed(file, times = times,
    #                     train_errors = train_errors,
    #                     test_errors = test_errors
    #                     )


if __name__ == "__main__":
    main(parse_args())
    # datasetname = 'usps'
    # comparison_files = np.load('comparison-results/'+str(datasetname)+'-OENN-TSTE-LOE.npz')
    #
    # times = np.array(comparison_files['times'])
    # n_range = 100 * np.array(range(1, 11))
    # fig, axes = plt.subplots(1,3,figsize = (10,3.3))
    # # print(times.shape)
    # axes[0].plot(n_range, times[:,0], marker = 'o')
    # axes[0].plot(n_range, times[:, 1], marker='x')
    # axes[0].plot(n_range, times[:, 2], marker='+')
    #
    # axes[0].set_xlabel('Number of items (n)')
    # axes[0].set_ylabel('Embedding time (sec)')
    # axes[0].legend(['OENN','TSTE','LOE'])
    # axes[0].set_title(str(datasetname) + '(dim = 25)')
    #
    # train_error = np.array(comparison_files['train_errors'])
    # axes[1].plot(n_range, train_error[:, 0], marker='o')
    # axes[1].plot(n_range, train_error[:, 1], marker='x')
    # axes[1].plot(n_range, train_error[:, 2], marker='+')
    # axes[1].set_xlabel('Number of items (n)')
    # axes[1].set_ylabel('Training error')
    # axes[1].legend(['OENN', 'TSTE', 'LOE'])
    # axes[1].set_title(str(datasetname) + '(dim = 25)')
    # axes[1].set_ylim([0, 0.25])
    #
    # test_error = np.array(comparison_files['test_errors'])
    # axes[2].plot(n_range, test_error[:, 0], marker='o')
    # axes[2].plot(n_range, test_error[:, 1], marker='x')
    # axes[2].plot(n_range, test_error[:, 2], marker='+')
    # axes[2].set_xlabel('Number of items (n)')
    # axes[2].set_ylabel('Test error')
    # axes[2].legend(['OENN', 'TSTE', 'LOE'])
    # axes[2].set_title(str(datasetname) + '(dim = 25)')
    # axes[2].set_ylim([0, 0.25])
    # plt.tight_layout()
    # fig.savefig('comparison-figs/'+datasetname+'-OENN-TSTE_MNIST_50.png')
    # plt.show()
