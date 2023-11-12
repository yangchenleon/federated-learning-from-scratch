from src.utils.setting import *
from src.server.fedavg import FedAvgServer
from src.server.fedmd import FedMDServer
from src.server.fedpkl import FedPKLServer
from src.server.fedproto import FedProtoServer


ARG_DICT = {
    'fedavg': fedavg_argpaser(),
    'fedmd': fedmd_argpaser(),
    'fedpkl': fedpkl_argparser(),
    'fedproto': fedproto_argpaser(),
}

ALGO_DICT = {
    'fedavg': FedAvgServer,
    'fedmd': FedMDServer,
    'fedpkl': FedPKLServer,
    'fedproto': FedProtoServer,
}