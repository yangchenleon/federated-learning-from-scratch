from data.utils.dataset_utils import partition_data
from src.utils.train_utils import fix_random_seed
from src.utils.algorithm import ARG_DICT, ALGO_DICT
from data.utils.visualize import draw_distribution, draw_data_sample

if __name__ == "__main__":
    fix_random_seed(42)

    algorithm = 'fedavg'
    datasets = ['mnist', 'fashion', 'mnistm', 'cifar10']
    models = ['resnet18']
    args = ARG_DICT[algorithm].parse_args()
    for dataset in datasets:
        # partition_data(dataset, args, draw=True)
        draw_data_sample(dataset)
        draw_distribution(dataset, 'hist')
  
    # server = ALGO_DICT[algorithm](datasets, models, args)
    # server.train()
    # server.test()
    
    