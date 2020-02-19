# Imports
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from train_utils import data_utils
import torch.optim
import torch.nn as nn
import argparse
import math
import numpy as np


def parse_args():
    # If embedding of a model is not already saved. Then use this code.
    parser = argparse.ArgumentParser(description='Input path of the model')
    parser.add_argument('-inp', '--model_path', dest='model_path', type=str, required=False, help='Input path to the Model')
    parser.add_argument('-hl', '--hl_size', type=int, dest='hl_size', required=True, help='Hidden layer size')
    parser.add_argument('-d', '--dim', type=int, dest='dim', required=True, help='Embedding dimension')
    parser.add_argument('-lr', '--learning_rate', dest= 'lr', type=float, default=1e-3, required=False, help='Learning rate')
    parser.add_argument('-l', '--layers', type=int, dest='layers', default=3, required=False, help='Number of layers')

    args = parser.parse_args()
    return args


def define_model(digits, hl_size, dim, layers):
    # Constructing the Network
    first_layer = [nn.Linear(digits, hl_size), nn.ReLU()]
    hidden_layers = []
    last_layer = [nn.Linear(hl_size, dim)]
    i = 1
    while i < layers:
        hidden_layers.append(nn.Linear(hl_size, hl_size))

        hidden_layers.append(nn.ReLU())
        i += 1

    emb_net = nn.Sequential(*first_layer, *hidden_layers, *last_layer)
    print(emb_net)
    return emb_net

def main(args):
    # Download the data
    mnist = MNIST(root="./downloads", download=True)
    data = mnist.data.numpy()
    data = np.reshape(data, (data.shape[0], 28*28))
    n = data.shape[0]
    labels = mnist.targets.numpy()

    # CUDA
    device = torch.device("cpu")
    print(device)

    # Input data to the network
    digits = int(math.ceil(math.log2(n)))
    bin_array = data_utils.get_binary_array(n, digits)  # Binary representation of the data

    # model_params go here
    hl_size = args.hl_size # int(120 + (1 * 25 * math.log2(n)))  # Hidden layer size
    dim = args.dim # 25
    layers = args.layers

    # define model
    emb_net = define_model(digits=digits, dim=dim, hl_size=hl_size, layers=layers)

    # Load the model
    emb_net.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')['model_state_dict']
    print(checkpoint.keys())
    emb_net.load_state_dict(checkpoint)
    emb_net.eval() # for inference

    # Compute the embedding of the data points.
    data_bin = torch.Tensor(bin_array)
    embeddings = emb_net(data_bin)  # Feed the binary array of indices to the network and generate embeddings: FP
    embedding_final = embeddings.cpu().detach().numpy()

    # Initialize t-SNE
    t_sne = TSNE(n_components=2, random_state=0, perplexity=40)
    X_2d = t_sne.fit_transform(embedding_final)  # Compute the t-SNE embedding.

    # Visualize the data
    labels = mnist.targets.numpy()
    target_names = range(0, 10)

    target_ids = range(len(target_names))

    plt.figure(figsize=(40, 40))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=label)
    plt.legend()
    plt.show()
    plt.savefig('t_sne0.png')

if __name__ == '__main__':
    main(parse_args())