from src.utils.setting import *
from src.server.fedavg import FedAvgServer
from src.server.fedmd import FedMDServer


ARG_DICT = {
    'fedavg': fedavg_argpaser(),
    'fedmd': fedmd_argpaser(),
}

ALGO_DICT = {
    'fedavg': FedAvgServer,
    'fedmd': FedMDServer,
}