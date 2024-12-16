import os

import numpy as np


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('data/PeMS04/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('data/PeMS08/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join('data/PeMS03/pems03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('data/PeMS07/pems07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7M':
        data_path = os.path.join('data/PeMSD7M/PeMSD7M.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7L':
        data_path = os.path.join('data/PeMSD7L/PeMSD7L.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
