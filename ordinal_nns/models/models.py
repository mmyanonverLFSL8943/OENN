import torch.nn as nn


def standard_model(digits, hl_size, dim, layers):
    first_layer = [nn.Linear(digits, hl_size), nn.ReLU()]
    hidden_layers = []
    last_layer = [nn.Linear(hl_size, dim)]
    i = 1
    while i < layers:
        hidden_layers.append(nn.Linear(hl_size, hl_size))
        hidden_layers.append(nn.ReLU())
        i += 1

    emb_net = nn.Sequential(*first_layer, *hidden_layers, *last_layer)
    return emb_net