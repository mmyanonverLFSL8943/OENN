import data_select_utils
import numpy as np

# mnist
data, labels = data_select_utils.select_dataset(dataset_name='mnist')
print('MNIST: ', data.shape, labels.shape)

# fmnist
data, labels = data_select_utils.select_dataset(dataset_name='fmnist')
print('FashionMNIST: ',data.shape, labels.shape)

# emnist
data, labels = data_select_utils.select_dataset(dataset_name='emnist')
print('EMNIST: ',data.shape, labels.shape, np.unique(labels))

# kmnist
data, labels = data_select_utils.select_dataset(dataset_name='kmnist')
print('KMNIST: ',data.shape, labels.shape, np.unique(labels))

# usps
data, labels = data_select_utils.select_dataset(dataset_name='usps')
print('USPS: ', data.shape, labels.shape)

# cover_type
data, labels = data_select_utils.select_dataset(dataset_name='cover_type')
print('CoverType: ', data.shape, labels.shape)

# charfonts
data, labels = data_select_utils.select_dataset(dataset_name='charfonts')
print('CharFonts: ', data.shape, labels.shape)

# coil20
data, labels = data_select_utils.select_dataset(dataset_name='coil20')
print('Coil-20: ', data.shape, labels.shape)

# news_groups
data, labels = data_select_utils.select_dataset(dataset_name='news')
print('News20: ', data.shape, labels.shape)

# char
data, labels = data_select_utils.select_dataset(dataset_name='char')
print('Char: ', data.shape, labels.shape)

# kdd
data, labels = data_select_utils.select_dataset(dataset_name='kdd_cup')
print('KDD: ', data.shape, labels.shape)

# uniform
data, labels = data_select_utils.select_dataset(dataset_name='blobs')
print('Blobs: ', data.shape, labels.shape)

# # create_new_char_dataset
# save_path = '/home/faiz/personal-projects/icml_things/TripletEmbedding/ordinal_nns/datasets/'
# print('Create split for a new char dataset')
# data, labels = data_select_utils.select_dataset(dataset_name='char')
# print('Char: ', data.shape, labels.shape)
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(
#     data, labels, test_size=0.2, shuffle=False)
# print('Train:', len(y_train), 'Test: ', len(y_test))
# mew_char_dataset = [X_train, y_train, X_test, y_test]
# np.save(save_path + 'char_x.npy', mew_char_dataset)
#
# testing_data = np.load(save_path + 'char_x.npy', allow_pickle=True)
# print(len(testing_data), testing_data[0].shape, testing_data[1].shape, testing_data[2].shape, testing_data[3].shape)