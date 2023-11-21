# 直接知乎copy的代码，hist那块不是很看懂，这个也挺好的，不是说不能用，但是后来写的更有可读性

def draw_distribution(dataset, partition, args, path):
    labels = dataset.targets
    name_class = dataset.classes
    num_client = len(partition)
    num_class = len(name_class)

    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(num_class)]
    for client, idx_sample in enumerate(partition):
        for idx in idx_sample:
            label_distribution[labels[idx]].append(client)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, num_client + 1.5, 1),
             label=name_class, rwidth=0.5)
    plt.xticks(np.arange(num_client), ["Client %d" % c_id for c_id in range(num_client)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig(os.path.join(path, f'{type(dataset).__name__[6:]}_{args.num_client}_{args.partition}.png'))
    plt.show()