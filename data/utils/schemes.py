import numpy as np
import random

def iid_partition(dataset, num_client):
    idx_sample_client = [[] for _ in range(num_client)]
    labels = dataset.targets # in case dont know, type here is tensor(dont know why yet), better transform into numpy array
    idxs = list(range(len(labels)))

    random.shuffle(idxs)
    num_sample_client = int(len(idxs)/num_client) # means balanced, wonder whether imbalanced iid exist
    for i in range(num_client):
        idx_sample_client[i] = idxs[num_sample_client*i:num_sample_client*(i+1)]

    return idx_sample_client

def dirichlet(dataset, num_client, alpha, least_samples):
    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    labels = dataset.targets
    num_class = len(dataset.classes)
    idx_sample_client =[[] for _ in range(num_client)]
    min_size = 0

    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_client)]
        for k in range(num_class):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_client))
            proportions = np.array([p*(len(idx_j)<len(labels)/num_client) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_client):
            idx_sample_client[j] = idx_batch[j]
    return idx_sample_client

def pathological(dataset, num_client, num_class_client, balance, least_samples):
    '''
    some problem with partion when num_client * num_class_clinet < num_class
    try num_class_client = max(int(np.ceil(num_class/num_client)), num_class_client)
    but without it still work, because the num_select is always 1, same effect as above.
    beside tried available_client, but seem can't set all client have same num_class_client
    it is useless while it may bring more trouble, so just leave it.
    and the implementation is not good, seems some class can never meet in a same client, not sure, can't statisticly prove it.
    
    modified shards partition, see "allocate_shards" for more details, seems when num_class_client==2, we call it pathological. whether, quite different from shards allocation
    '''
    labels = dataset.targets
    num_class = len(dataset.classes)
    idx_sample_client = [[] for _ in range(num_client)]

    # class data by class
    idxs = np.array(range(len(labels)))
    label2idx = [None for _ in range(num_class)]
    for i in range(num_class):
        label2idx[i] = idxs[labels==i]
    
    idx_client = np.array(range(num_client))
    cnt_class_client = np.array([num_class_client for _ in range(num_client)])
    # All the data is divided into num_client * class_per_client shards, with each class evenly distributed among these shards (dividing them by num_class). The number of shards in class corresponds to the number of clients selcected.
    # ceil出现的问题是，最后一个类别的shard client数量不够，一个class n个shard但只有<n个分，造成部分client多；而floor是过早分配完，部分client的shard不够，如果不要求所有客户类数量一样的可以选择，差不了多少
    num_select = int(np.ceil((num_client/num_class)*num_class_client))

    for i in range(num_class):
        # choose client for class i and update client list
        pri_selected_client = idx_client[cnt_class_client==max(cnt_class_client)]
        # to supportclass_per_client > 2 
        if len(pri_selected_client) >= num_select:
            selected_client = np.random.choice(pri_selected_client, num_select, replace=False)
        else:
            restavai_client = np.setdiff1d(idx_client, pri_selected_client)
            aft_selected_client = np.random.choice(restavai_client, num_select-len(pri_selected_client), replace=False)
            selected_client = np.concatenate((pri_selected_client, aft_selected_client))
        cnt_class_client[selected_client] -= 1

        # now we have selected clients and data(sample) of class i
        idx_sample = label2idx[i]
        num_sample, num_slected_client = len(idx_sample), len(selected_client)
        num_sample_client_avg = int(num_sample/num_slected_client)

        # choose samples for each client, whether balance or not
        if balance:
            num_sample_client = np.ones(num_slected_client, dtype=int) * num_sample_client_avg
            num_sample_client[:(num_sample - sum(num_sample_client))] += 1
        else:
            least_sample_client = least_samples / num_slected_client
            num_sample_client = np.random.randint(max(num_sample_client_avg/10, least_sample_client), num_sample_client_avg, num_slected_client-1)
            num_sample_client = np.append(num_sample_client, num_sample - sum(num_sample_client[:-1]))
        
        # assign sample to client with idx
        idx = 0
        np.random.shuffle(idx_sample)
        for client, num_sample_client in zip(selected_client, num_sample_client):
            idx_sample_client[client].extend(idx_sample[idx:idx+num_sample_client])
            idx += num_sample_client
        
    return idx_sample_client

def randomly_assign_classes(dataset, num_client, class_num_client):
    # 基本思想是client选择几种类，根据计数切分各类，极端情况有的类可能选不上，有的类很多人选而数量少，导致数量不均匀，但是类数量均匀
    idx_sample_client = [[] for _ in range(num_client)]
    labels = dataset.targets
    num_class = len(dataset.classes)

    # class data by class    
    label_list = list(range(num_class))
    label2idx = [np.where(labels == i)[0].tolist() for i in label_list]
    # same as above
    # idxs = np.array(range(len(labels)))
    # label2idx = [None for _ in label_list]
    # for i in range(num_class):
    #     label2idx[i] = idxs[labels==i]
    
    assigned_labels = [None for _ in range(num_client)]
    selected_times = np.zeros_like(label_list)
    for i in range(num_client):
        sampled_labels = random.sample(label_list, class_num_client)
        assigned_labels[i] = sampled_labels
        for j in sampled_labels:
            selected_times[j] += 1
    selected_times = [1 if i == 0 else i for i in selected_times]

    batch_sizes = [None for _ in label_list]
    for i in label_list:
        batch_sizes[i] = int(len(label2idx[i]) / selected_times[i])

    for i in range(num_client):
        for cls in assigned_labels[i]:
            batch_size = batch_sizes[cls]
            selected_idx = random.sample(label2idx[cls], batch_size)
            idx_sample_client[i].extend(selected_idx)
            # idx_sample_client[i] = np.concatenate([idx_sample_client[i], selected_idx]) 
            # origin way, require astype, cause concat empty list come out a float type
            label2idx[cls] = np.setdiff1d(label2idx[cls], selected_idx).tolist()
    return idx_sample_client

def allocate_shards(dataset, num_client, shard_num):
    # 在FL-Bench上改的，改的不多，但还是要提一嘴，按照类排序那里小改
    # 和我改的pat类似，不过相当于总shard数要平均到各个class，然后逐class分配，而非一整个分配，但平均shard不一定是整数，要么向上导致client的shard数多了，要么向下导致少了，不过各client的shard数还是近似平均的（一个shard的数量差别）。好处是一个shard只对应一个类，shard选择不会跨类别，同时因为没有放回采样，一个客户端有多少shard就有多少类。
    shards_total = num_client * shard_num
    size_of_shards = int(len(dataset) / shards_total)
    idx_sample_client = [[] for _ in range(num_client)]
    labels = dataset.targets

    sort_idxs = np.concatenate([np.where(labels == i)[0].tolist() for i in range(len(dataset.classes))]).tolist()

    # assign
    idx_shard = list(range(shards_total))
    for i in range(num_client):
        rand_set = random.sample(idx_shard, shard_num)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            idx_sample_client[i].extend(sort_idxs[rand * size_of_shards : (rand + 1) * size_of_shards])

    return idx_sample_client