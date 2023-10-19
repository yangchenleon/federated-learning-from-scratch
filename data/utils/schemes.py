import numpy as np

def iid_partition(dataset, num_client, balance, least_samples):
    labels = dataset.targets
    num_class = len(dataset.classes)
    idx_sample_client = [[] for _ in range(num_client)]

    idx_sample_client = pathological(dataset, num_client, num_class_client=num_class, balance=balance, least_samples=least_samples)

    return idx_sample_client

def dirichlet(dataset, num_client, alpha):
    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    labels = dataset.targets
    num_class = len(dataset.classes)
    idx_sample_client =[[] for _ in range(num_client)]

    min_size = 0
    K = num_class
    N = len(labels)

    while min_size < num_class:
        idx_batch = [[] for _ in range(num_client)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_client))
            proportions = np.array([p*(len(idx_j)<N/num_client) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_client):
            idx_sample_client[j] = idx_batch[j]
    return idx_sample_client

def pathological(dataset, num_client, num_class_client, balance, least_samples):
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
    available_client = np.array(range(num_client))

    for i in range(num_class):
        # choose client for class i and update client list
        num_select = int(np.ceil((num_client/num_class)*num_class_client))

        # All the data is divided into num_client * class_per_client shards, with each class evenly distributed among these shards (dividing them by num_class). The number of shards in class corresponds to the number of clients selcected.
        pri_selected_client = idx_client[cnt_class_client==max(cnt_class_client)]

        # to supportclass_per_client > 2 
        if pri_selected_client.size >= num_select:
            selected_client = np.random.choice(pri_selected_client, num_select, replace=False)
        else:
            restavai_client = np.setdiff1d(available_client, pri_selected_client)
            aft_selected_client = np.random.choice(restavai_client, num_select-len(pri_selected_client), replace=False)
            selected_client = np.concatenate((pri_selected_client, aft_selected_client))
        cnt_class_client[selected_client] -= 1
        available_client = idx_client[cnt_class_client>0]

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

def other():
    pass