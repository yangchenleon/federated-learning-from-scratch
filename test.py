from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    choices=[
        "mnist",
        "cifar10",
        "cifar100",
        "synthetic",
        "femnist",
        "emnist",
        "fmnist",
        "celeba",
        "medmnistS",
        "medmnistA",
        "medmnistC",
        "covid19",
        "svhn",
        "usps",
        "tiny_imagenet",
        "cinic10",
        "domain",
    ],
    default="cifar10",
)
parser.add_argument("--iid", type=int, default=0)
parser.add_argument("-cn", "--client_num", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--split", type=str, choices=["sample", "user"], default="sample"
)
parser.add_argument("-f", "--fraction", type=float, default=0.5)
parser.add_argument("-c", "--classes", type=int, default=0)
parser.add_argument("-s", "--shards", type=int, default=0)
parser.add_argument("-a", "--alpha", type=float, default=0)
parser.add_argument("-ls", "--least_samples", type=int, default=40)

# For synthetic data only
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--dimension", type=int, default=60)

# For CIFAR-100 only
parser.add_argument("--super_class", type=int, default=0)

# For EMNIST only
parser.add_argument(
    "--emnist_split",
    type=str,
    choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
    default="byclass",
)

# For semantic partition only
parser.add_argument("-sm", "--semantic", type=int, default=0)
parser.add_argument("--efficient_net_type", type=int, default=0)
parser.add_argument("--gmm_max_iter", type=int, default=100)
parser.add_argument(
    "--gmm_init_params", type=str, choices=["random", "kmeans"], default="kmeans"
)
parser.add_argument("--pca_components", type=int, default=256)
parser.add_argument("--use_cuda", type=int, default=1)
args = parser.parse_args()
args.ggegege = 1111111
args_dict = vars(args)
print(args_dict)