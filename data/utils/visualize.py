import json
import numpy as np
import matplotlib.pyplot as plt

from data.utils.datasets import DatasetDict
from data.utils.setting import par_dict


def getinfo(dataset_name):
    par_file = f'{par_dict[dataset_name]}/stats.json'
    meta_file = f'{par_dict[dataset_name]}/args.json'
    with open(par_file, 'r') as f:
        data = json.load(f)
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    dataset = DatasetDict[dataset_name](transform=None)
    num_client = meta['num_client']
    num_class = len(dataset.classes)
    
    array = np.zeros(shape=(num_client, num_class))
    for i in range(num_client):
        for k in data[str(i)]['class_count'].keys():
            array[i][int(k)] = data[str(i)]['class_count'][k]
    
    info = {}
    info['labels'] = dataset.classes
    info['num_client'] = meta['num_client']
    info['partition'] = meta['partition']
    info['fullname'] = type(dataset).__name__[6:]
    
    return array, info

def heatmap(data, labels, save_dir):
    num_client, num_class = data.shape
    x = np.arange(num_client)
    y = np.arange(num_class)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data.T, cmap='Blues', aspect='auto') # auto is important incase class is too much

    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of samples')
    ax.set_title('Display Label Distribution on Different Clients')
    ax.set_xticks(x)
    ax.set_xticklabels(["%d" % (i + 1) for i in range(num_client)])

    step = 5 if num_class >= 100 else 1
    yticks = np.arange(0, num_class, step)
    yticklabels = [str(i+1) for i in range(0, num_class, step)] # incase class is too much
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')

    plt.tight_layout()

    plt.savefig(save_dir)

def histogram(data, labels, save_dir):
    num_client, num_class = data.shape
    x = np.arange(num_client)
    y = np.arange(num_class)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    # ax.get_figure().set_size_inches(12, 8)
    bar_width = 0.5

    cmap = plt.get_cmap('tab10')
    for i in range(num_class):
        ax.bar(x, data[:, i], bar_width, label=labels[i], color=cmap(i % 10), bottom=np.sum(data[:, :i], axis=1))
    ax.bar(len(x), 0, bar_width)

    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of samples')
    ax.set_title('Display Label Distribution on Different Clients')
    ax.set_xticks(x)
    ax.set_xticklabels(["%d" % (i + 1) for i in range(num_client)])
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_dir)

def scatter(data, labels, save_dir):
    num_client, num_class = data.shape
    x = np.repeat(np.arange(num_client), num_class)
    y = np.tile(np.arange(1, num_class + 1), num_client)
    sizes = data.flatten() / np.max(data) * 1000
    colors = data.flatten()

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(x, y, c=colors, s=sizes, cmap='Blues')
    ax.grid(True, linestyle='--', linewidth=0.5)

    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of samples')
    ax.set_title('Display Label Distribution on Different Clients')
    ax.set_xticks(np.arange(num_client))
    ax.set_xticklabels(["%d" % (i + 1) for i in range(num_client)])

    ax.set_ylim(0, num_class + 1)
    ax.set_yticks(np.arange(1, num_class + 1))
    ax.set_yticklabels(np.arange(0, num_class))

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')

    plt.savefig(save_dir)

def draw_distribution(dataset_name, method):
    data, info = getinfo(dataset_name)
    labels = info['labels']
    output_name = f"results/figures/{info['fullname']}_{info['num_client']}_{info['partition']}_{method}.png"

    if method == 'hist':
        histogram(data, labels, output_name)
    elif method == 'heat':
        heatmap(data, labels, output_name)
    elif method == 'dot':
        scatter(data, labels, output_name)

def draw_data_sample(dataset_name):
    dataset = DatasetDict[dataset_name](transform=None)
    plt.figure(figsize=(20, 10))
    size = 20
    random_indices = np.random.randint(0, len(dataset), size=size)
    for i in range(size):
        plt.subplot(4, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(dataset.data[random_indices[i]] / 255, cmap=plt.cm.binary)
        #plt.imshow(train_images[i].squeeze(), cmap=plt.cm.binary)
        plt.xlabel(dataset.classes[dataset.targets[random_indices[i]]])
    plt.savefig(f'results/figures/{type(dataset).__name__[6:]}.png')