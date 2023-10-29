 
# 意识到没有必要transform不需要从外界获取，直接根据输入的model和dataset在通过访问字典就修改到适配的
def load_dataset(self, transform=None):
        '''
        default data_path is fixed in datasets, only set partition dir
        read partition and load train/test dataset
        wired huh! why not direct pass the datast class, because i want allow client to apply it's own transform, which can't be changed into dataset is created
        '''
        dataset = DatasetDict[self.dataset](transform=transform) 
        with open(os.path.join(par_dict[self.dataset], "partition.pkl"), "rb") as f:
            partition = pickle.load(f)
        self.trainset = CustomSubset(dataset, partition[self.id]['train'])
        self.testset = CustomSubset(dataset, partition[self.id]['test'])
        
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # num_workers=self.args.num_workers,
            drop_last=True, # When the current batch size is 1, the batchNorm2d modules in the model would raise error. So the latent size 1 data batches are discarded.
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # num_workers=self.args.num_workers,
            drop_last=False,
        )
        # only in customsubse and in uint8, use [index] to apply transform
        # print(self.trainset.data.shape, self.trainset.data.dtype)
        # print(next(iter(self.trainloader))[0].shape, next(iter(self.trainloader))[0].dtype)
        return self.trainset, self.testset
