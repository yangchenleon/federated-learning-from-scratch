import os
import numpy as np

def read_data(dataset, idx, train=True):
    '''
    read [client] data from file
    '''
    if train:
        path = os.path.join('data', dataset, 'train', f'{idx}.npz')
    else:
        path = os.path.join('data', dataset, 'test', f'{idx}.npz')
    data = np.load(path, allow_pickle=True)['data'].item()
    return data['x'], data['y']