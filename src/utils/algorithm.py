from src.utils.setting import *
from src.server.fedavg import FedAvgServer
from src.server.fedmd import FedMDServer
from src.server.fedpkl import FedPKLServer


ARG_DICT = {
    'fedavg': fedavg_argpaser(),
    'fedmd': fedmd_argpaser(),
    'fedpkl': fedpkl_argparser(),
}

ALGO_DICT = {
    'fedavg': FedAvgServer,
    'fedmd': FedMDServer,
    'fedpkl': FedPKLServer,
}