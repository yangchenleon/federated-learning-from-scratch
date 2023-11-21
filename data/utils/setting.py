data_dict = {
    'mnist': 'data/datasets/',
    'fashion': 'data/datasets/',
    'cifar10': 'data/datasets/CIFAR10/raw/',
    'emnist': 'data/datasets/',
    'tinyimage': 'data/datasets/TinyImageNet/raw/', # not use actually
    'mnistm': 'data/datasets/MNISTM/raw',
}

base = 'data/partition/'
par_dict = {
    'mnist': base + 'MNIST/',
    'fashion': base + 'FashionMNIST/',
    'cifar10': base + 'CIFAR10/',
    'emnist': base + 'EMNIST/',
    'tinyimage': base + 'TinyImageNet/',
    'mnistm': base + 'MNISTM/',
}

data_info = {
    'mnist': (3, 28*28, 10),
    'fashion': (1, 28*28, 10),
    'cifar10': (3, 32*32, 10),
    'emnist': (3, 28*28, 47),
    'tinyimage': (3, 64*64, 200),
    'mnistm': (3, 28*28, 10),
}

MEAN = {
    "mnist": [0.1307, 0.1307, 0.1307],
    "fashion": [0.286, 0.286, 0.286], # imagenet by default
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "emnist": [0.1736, 0.1736, 0.1736],
    "tinyimage": [122.5119, 114.2915, 101.388],
    'mnistm': [0.485, 0.456, 0.406],

    "femnist": [0.9637],
    "medmnist": [124.9587],
    "medmnistA": [118.7546],
    "medmnistC": [124.424],
    "covid19": [125.0866, 125.1043, 125.1088],
    "celeba": [128.7247, 108.0617, 97.2517],
    "synthetic": [0.0],
    "svhn": [0.4377, 0.4438, 0.4728],
    "cinic10": [0.47889522, 0.47227842, 0.43047404],
    "domain": [0.485, 0.456, 0.406],
}

STD = {
    "mnist": [0.3015, 0.3015, 0.3015],
    "fashion": [0.3205, 0.3205, 0.3205],
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
    "emnist": [0.3248, 0.3248, 0.3248],
    "tinyimage": [58.7048, 57.7551, 57.6717],
    'mnistm': [0.229, 0.224, 0.225],

    "femnist": [0.155],
    "celeba": [67.6496, 62.2519, 61.163],
    "synthetic": [1.0],
    "svhn": [0.1201, 0.1231, 0.1052],
    "cinic10": [0.24205776, 0.23828046, 0.25874835],
    "domain": [0.229, 0.224, 0.225],
}
