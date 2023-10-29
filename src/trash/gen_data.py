import json
import numpy as np
import torch, torchvision
from torchvision import transforms
from utils.dataset_utils import partition_data, save_partition, draw_distribution

def gen_client_data(dataset_name, data_path, num_client, is_draw, setting):
    # image preprocessing, tensor and normalization (widely used in pytorch)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    
    # four file: t10k for test, train for train with labels and images
    train_data = getattr(torchvision.datasets, dataset_name)(
        root=data_path, train=True, download=True, transform=transform)
    test_data = getattr(torchvision.datasets, dataset_name)(
        root=data_path, train=False, download=True, transform=transform)
    
    dataset = torch.utils.data.ConcatDataset([train_data, test_data])
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    dataset_image, dataset_label = next(iter(dataset_loader))
    dataset_image, dataset_label = np.array(dataset_image), np.array(dataset_label)

    par_key = ['niid', 'balance', 'partition', 'alpha', 'num_class_client', 'least_samples']
    par_setting = {key: value for key, value in setting.items() if key in par_key}
    partition = partition_data((dataset_image, dataset_label), num_client, **par_setting)
    save_partition((dataset_image, dataset_label), partition, setting, path='data/_ClientData')
    if is_draw:
        name_class = train_data.classes
        draw_distribution((dataset_image, dataset_label), name_class, partition)
    return

if __name__ == '__main__':
    
    np.random.seed(42)
    dataset_name = 'MNIST' # use offical dataset name: MNIST, CIFAR10, CIFAR100, FashionMNIST
    data_path = 'data' # data/CIFAR10/raw when using cifar-10
    num_client = 10

    # use json file setting by default, also suport command line arguments
    setting = json.loads(open('datasets/setting.json').read())
    # partition_setting['niid'] = True if sys.argv[1] == "niid" else False
    setting['partition'] = 'pat'
    gen_client_data(dataset_name, data_path, num_client, is_draw=True, setting=setting)

    '''
    另外一篇的代码，实现思路是
    1. 关于参数输入：arg参数基础，name、path、setting都在里面，但是这里的arg参数是通过json文件来实现的，原来是通过sys.argv，问题不大
    2. 关于数据集获取：通过类继承封装了dataset，有一个basedataset，能按照torchvision里的dataset方法调用书香，不过貌似并没有继承torchvision里的dataset类，更像是其中的属性重新写了一遍，总之给每一种数据集写了一个dataset类。功能上本质是将train、test的数据进行拼接（但不是简单的[train， test]）
    3. 数据划分：和我的想法类似，狠心在于partition，实现上，我采用最垃圾的if、else，而且逻辑混用，这边用了arg对，形式上作划分，具体什么方法什么参数靠用户（不用应该没有更好的更体面的方法），重点在于设置了4种划分方法，每种方法都是一个独立的函数
    4. 细枝末节：最后的汇总上，有很乱，主要是数据集的不同处理，对于torchvision下载的可以直接套用上面的内容，而对于femnist、celeba、synthetic、domain，需要单独的函数执行划分；保存数据用的是pickle，我用的是numpy；关于split参数的作用没搞懂，有两种值：user、sample，哦，意思是一个全体参加，分别有测试和训练，前者是训练和测试用户，不划分数据集。
    
    从代码美观性，一个简单，一个看之高大上，更喜欢后面的写法，之后再改成这种写法。
    '''