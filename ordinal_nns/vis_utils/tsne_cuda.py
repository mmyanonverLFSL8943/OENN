# coding: utf-8
# !/usr/bin/env python
import torch
import math
import numpy as np
from tsnecuda import TSNE
import torch.optim
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt
import argparse
import os

from data_select_utils import select_dataset
from train_utils import data_utils
from models.models import standard_model


def parse_args():
    # If embedding of a model is not already saved. Then use this code.
    parser = argparse.ArgumentParser(description='Parameters to t-SNE')
    parser.add_argument('-mp', '--model_path', type=str, required=True, help='Input path to the Model')
    parser.add_argument('-m', '--model', required=False, default='standard', help='model definition')
    parser.add_argument('-data', '--dataset_name', required=True, help='Name of the dataset')
    parser.add_argument('-hl', '--hl_size', type=int, required=True, help='Hidden layer size')
    parser.add_argument('-dim', '--dim', type=int, required=True, help='Embedding dimension')
    parser.add_argument('-l', '--layers', type=int, default=3, required=False, help='Number of layers')
    parser.add_argument('-p', '--perplexity', type=int, default=30, required=False,
                        help='Perplexity parameter for tSNE')

    args = parser.parse_args()
    return args


def define_model(model_name, digits, hl_size, dim, layers):
    # Constructing the Network
    if model_name == 'standard':
        emb_net = standard_model(digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    return emb_net


def main(args):

    # fetch the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # get the dataset you want
    vec_data, labels = select_dataset(args.dataset_name, testing=False)

    # compute the number of data samples
    n = vec_data.shape[0]
    # Input data to the network
    digits = int(math.ceil(math.log2(n)))
    bin_array = data_utils.get_binary_array(vec_data.shape[0], digits)  # Binary representation of the data

    dim = args.dim
    hl_size = args.hl_size
    layers = args.layers

    experiment_name = os.path.basename(args.model_path)
    result_name = os.path.splitext(experiment_name)[0] + '_perplexity_' + str(args.perplexity)

    # load the saved_model
    emb_net = define_model(model_name=args.model, digits=digits, hl_size=hl_size, dim=dim, layers=layers)
    emb_net = emb_net.to(device)

    if torch.cuda.device_count() > 1:
        emb_net = DataParallel(emb_net)
        print('multi-gpu')

    emb_net.load_state_dict(torch.load(args.model_path)) #['model_state_dict'])
    emb_net.eval()

    # Compute the embedding of the data points.
    data_bin = torch.Tensor(bin_array).to(device)
    embeddings = emb_net(data_bin)  # Feed the binary array of indices to the network and generate embeddings: FP
    embedding_final = embeddings.cpu().detach().numpy()

    # Compute the t-SNE embedding.
    embedded_x = TSNE(n_components=2, perplexity=args.perplexity).fit_transform(embedding_final)

    # Visualize the data
    target_names = np.unique(labels)
    target_ids = target_names

    f = plt.figure(figsize=(30, 30))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'Aqua', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(embedded_x[labels == i, 0], embedded_x[labels == i, 1], s=5, c=c, label=label, alpha=0.9)
    plt.legend()
    # plt.show()

    os.makedirs('tsne', mode=0o777, exist_ok=True)
    embeddings_with_labels = {'embeddings': embedded_x, 'labels':labels}
    np.save(os.path.join('tsne', result_name + '.npy'), embeddings_with_labels)
    f.savefig(os.path.join('tsne', result_name + '.eps'), format='eps')


if __name__ == "__main__":
    main(parse_args())
