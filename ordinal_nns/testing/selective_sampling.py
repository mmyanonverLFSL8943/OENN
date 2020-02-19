import numpy as np
import os
def get_neighbours(dataset_name):
    file_name = os.path.join('/home/faiz/personal-projects/icml_things/ordinal_nns', 'indices',
                             dataset_name + '_neigbours_indices.npy')
    indices = np.load(file_name)
    return indices

def gen_selective_triplets(num_trips, neighbor_indices):
    """
    Description: Generate triplets according to nearest neighbor sampling strategy
    :param num_trips:
    :param neighbor_indices:
    :return:
    """
    n = neighbor_indices.shape[0]  # Total number of points
    num_neighbors = neighbor_indices.shape[1]  # Number of neighbors selected for each point.

    all_triplet_indices = np.zeros((num_trips, 3))  # Initializing the triplets
    # The first index is randomly sampled from all the points
    indices_one = np.random.randint(n, size=(num_trips, 1)).squeeze()

    # The second index is randomly sampled from the nearest neighbor of index 1.
    nearest_neighbors = neighbor_indices[indices_one[:], :]  # Arranges the nearest neighbors according to indices_one.
    if nearest_neighbors.shape[0] == num_trips & nearest_neighbors.shape[1] == num_neighbors:
        print('Dimensions of nearest neighbors are as expected.')

    random_nn_indices = np.random.randint(1, num_neighbors, size=(num_trips, 1)).squeeze()  # Exclude the first point.
    indices_two = nearest_neighbors[np.arange(len(nearest_neighbors)), random_nn_indices]

    # The third index is sampled from the set of points farther away from index 1 than index 2.
    indices_three = np.zeros((num_trips, 1))
    for i in range(num_trips):
        neg_samples = np.setdiff1d(np.array(range(n)), nearest_neighbors[i, 0:random_nn_indices[i]])
        indices_three[i] = neg_samples[np.random.randint(len(neg_samples), size=1)]
    indices_three = indices_three.squeeze()
    # The third index would be selected from any point far away from index 1 than index 2.
    all_triplet_indices[:, 0] = indices_one.astype(int)
    all_triplet_indices[:, 1] = indices_two.astype(int)
    all_triplet_indices[:, 2] = indices_three.astype(int)

    return all_triplet_indices.astype(int)

def gen_selective_triplets_v2(num_trips, neighbor_indices):
    n = neighbor_indices.shape[0]  # Total number of points
    num_neighbors = neighbor_indices.shape[1]  # Number of neighbors selected for each point.

    indices_one = np.random.randint(n, size=(num_trips, 1)).squeeze()
    print(indices_one.shape)





indices = get_neighbours(dataset_name='mnist')
trips = 1000
all_triplets = gen_selective_triplets_v2(trips, indices)
# print(all_triplets.shape)