from argparse import ArgumentParser


save_base = 'results/'
state_dir = save_base + 'checkpoints/'
figure_dir = save_base + 'figures/'
log_dir = save_base + 'logs/'

# 放到__init__()?
# ckpt_base = os.path.join(path, 'checkpoints')
# if os.path.exists(ckpt_base) is False:
#     os.makedirs(ckpt_base)

def get_argpaser():
    parser = ArgumentParser()
    parser.add_argument('-n', '--num_client', type=int, default=10)
    parser.add_argument('--balance', type=int, default=1) # actually only impl pat imbalance
    parser.add_argument('--partition', type=str, choices=['iid', 'pat', 'dir', 'mix', 'rad', 'srd'], default='dir')
    parser.add_argument('-a', '--alpha', type=float, default=0.8)
    parser.add_argument('-nc', '--num_class_client', type=int, default=7)
    parser.add_argument('-ls', '--least_samples', type=int, default=40)
    parser.add_argument('-ts', '--train_size', type=float, default=0.8)

    # mlp: 0.1, cnn: 0.001, alexnet/resnet: 0.01, 
    parser.add_argument('-pt', '--pretrained', type=int, default=0)
    parser.add_argument('-lr', '--lr', type=float,  default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-mom', '--momentum', type=float, default=0.9)
    parser.add_argument('-ne', '--num_epochs', type=int, default=5)

    parser.add_argument('-jt', '--join_ratio', type=float, default=0.6) # jt * ne is the number of epoch in one-client setting
    parser.add_argument('-gr', '--global_round', type=int, default=30)

    parser.add_argument('-ge', '--global_eval', type=int, default=1)
    parser.add_argument('-ce', '--client_eval', type=int, default=0)

    return parser

def fedavg_argpaser():
    parser = get_argpaser()
    return parser

def fedmd_argpaser():
    parser = get_argpaser()
    # parser.set_defaults(join_ratio=1.0)
    parser.add_argument('-pd', '--public_dataset', type=str, default='mnist')
    parser.add_argument('-ps', '--public_size', type=int, default=5000)
    parser.add_argument('-bpe', '--num_pretrain_epoch', type=int, default=5)
    parser.add_argument('--num_digest_epoch', type=int, default=2)
    parser.add_argument('-pbs', '--public_batch_size', type=int, default=32)
    return parser

def fedpkl_argparser():
    parser = fedmd_argpaser()
    # parser.set_defaults(join_ratio=1.0)
    parser.add_argument('--num_distil_epoch', type=int, default=2)
    parser.add_argument('-lt', '--local_temper', type=int, default=10)
    parser.add_argument('-gt', '--global_temper', type=int, default=10)
    parser.add_argument('-glr', '--global_lr', type=float, default=0.01)
    parser.add_argument('-nlme', '--num_learn_matrix_epoch', type=int, default=1)
    parser.add_argument('--rho', type=float, default=0.7)
    
    return parser

def fedproto_argpaser():
    parser = get_argpaser()
    parser.add_argument('--lamda', type=float, default=1.0)

    return parser