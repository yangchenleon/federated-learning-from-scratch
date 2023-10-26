data_dict = {
    'mnist': 'data/datasets',
    'fashion': 'data/datasets',
    'cifar10': 'data/datasets/CIFAR10/raw'
}
base = 'data/partition/'
par_dict = {
    'mnist': base + 'MNIST',
    'fashion': base + 'FashionMNIST',
    'cifar10': base + 'CIFAR10',
}

data_info = {
    'mnist': (3, 28*28, 10),
    'fashion': (1, 28*28, 10),
    'cifar10': (3, 32*32, 10),
}