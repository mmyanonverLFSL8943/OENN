import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for different experiments')
    parser.add_argument('-f', '--file_name', type=str, required=True,
                        help='Select the dataset, default is: mnist')
    args = parser.parse_args()
    return args

def plot_experiment(heat_map, legend_values, exp_id):
    print(heat_map.shape, legend_values[0].shape, legend_values[1].shape)
    fig, ax = plt.plot()

    im = ax.imshow(np.flip(heat_map, axis=0))

    if exp_id==1:
        ax.set_xlabel('Dimensions', fontsize=13)
        ax.set_ylabel('Hidden Layer Size', fontsize=13)

    if exp_id==2:
        ax.set_xlabel('Number of Points', fontsize=13)
        ax.set_ylabel('Dimensions', fontsize=13)

    # ... and label them with the respective list entries
    ax.set_xticklabels(legend_values[0])
    ax.set_yticklabels(np.flip(legend_values[1], axis=0))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=16)
    fig.tight_layout()
    plt.show()
    return fig


def check_property(tokens, property_name):
    exists = False
    index_found = -1
    for index, element in enumerate(tokens):
        if element == property_name:
            exists = True
            index_found = index
            return exists, index_found
    return exists, index_found


def parse_config(file_path):
    experiment_name = os.path.basename(file_path)
    result_name = os.path.splitext(experiment_name)[0]

    tokens = result_name.split('_')

    if tokens[0]=='char' or 'usps':
        dataset_name = tokens[0] + tokens[1]
    else:
        dataset_name = tokens[0]

    exists, index = check_property(tokens, 'n')
    samples = int(tokens[index +1])

    exists, index = check_property(tokens, 'dim')
    dim = int(tokens[index + 1])

    exists, index = check_property(tokens, 'hl')
    hl = int(tokens[index + 1])

    exists_train, index = check_property(tokens, 'train')
    exists_test, index = check_property(tokens, 'test')

    if exists_train:
        exp_name = 'train'
    elif exists_test:
        exp_name = 'test'
    else:
        print('Wrongly formatted maps')
        exp_name = 'invalid'
        exit(0)
    heat_specs = {'dataset_name': dataset_name, 'samples': samples, 'dim': dim, 'hl': hl, 'exp_name': exp_name}
    return heat_specs

def main(args):

    # Experiments on all 5 datasets - MNIST-PC, CHAR-PC, USPS-PC, GMM, UNIFORM
    # 1. Vary hl_size(from 0 to 5), vary n( from 7 to 14) and compute triplet error. Keep dim = 2
    # 2. Vary each hl_size(from 1 to 8), vary dim( from 2 to 20), keep n = 512.
    # Check log files to see what converged and what information do we have and create the 3D matrix for each
    # experiment.
    # experiment specs for plot legend

    min_n, max_n = 9, 9  # 6, 15
    min_dim, max_dim = 2, 16
    min_hl, max_hl = 1, 8


    n_range = 2 ** np.arange(min_n, max_n + 1)  # Range of number of points to experiment over.
    dim_range = np.arange(min_dim, max_dim + 1, 2)  # Range of dimensions to experiment over.
    hl_range = 70 + 20 * np.arange(min_hl, max_hl + 1)  # Range of hl_size


    heat_specs = parse_config(args.file_name)

    exp_id = -1
    if heat_specs['samples'] == 1:
        exp_id = 1
    elif heat_specs['dim'] == 1:
        exp_id = 2

    # load heatmap matrix
    heat_matrix = np.load(args.file_name)
    print(heat_matrix.shape)

    # average over the number of repetitions
    heat_matrix = np.mean(heat_matrix, axis=3)

    # create 2D matrix
    if exp_id == 1:
        matrix = np.squeeze(heat_matrix[0, :-2, :])
        print(exp_id)
        print(matrix.shape, dim_range.shape, hl_range.shape)
        print(dim_range, hl_range)
        fig = plot_experiment(heat_map=matrix, legend_values=[dim_range, hl_range], exp_id=exp_id)

    elif exp_id ==2:
        matrix = np.squeeze(heat_matrix[:, 0, :])
        print(exp_id)
        print(matrix.shape, n_range.shape, dim_range.shape)
        fig = plot_experiment(heat_map=matrix, legend_values=[n_range, hl_range], exp_id=exp_id)


if __name__ == "__main__":
    main(parse_args())
