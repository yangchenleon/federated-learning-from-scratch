import os, pickle, sys
import torch
from argparse import ArgumentParser

sys.path.append('')
from data.utils.datasets import DatasetDict, CustomSubset
from data.utils.setting import par_dict, data_dict

class Client(object):
    '''
    also as FedAvgClient
    '''
    def __init__(self, dataset, client_id, model, args, logger, device):
        # here dataset is actully dataset_name, using dict to acquire
        # because usually do not use class as parameter
        # beside, a clint only need a subset, only train/test set, not whole
        self.args = args
        self.device = device
        self.logger = logger
        self.model = model
        self.id = client_id
        self.dataset = dataset

        self.trainset = None
        self.trainset = None
        self.trainloader = None
        self.testloader = None
        
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), 
        #     lr=args.lr, 
        #     momentum=args.momentum, 
        #     weight_decay=args.weight_decay
        # )
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_dataset(self, transform=None, target_transform=None):
        '''
        default data_path is fixed in datasets, only set partition dir
        read partition and load train/test dataset
        '''
        dataset = DatasetDict[self.dataset](root=data_dict[self.dataset], transform=transform, target_transform=target_transform)
        with open(os.path.join(par_dict[self.dataset], "partition.pkl"), "rb") as f:
            partition = pickle.load(f)
        self.trainset = CustomSubset(dataset, partition[self.id]['train'])
        self.testset = CustomSubset(dataset, partition[self.id]['test'])
        
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # num_workers=self.args.num_workers
        )
        return self.trainset, self.testset
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float,  default=1e-2)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    args = parser.parse_args()
    
    dataset_name = 'mnist'
    client = Client(dataset_name, 0, None, args, None, None)
    trainset, testset = client.load_dataset(transform=None)
    print(trainset.data.shape)
