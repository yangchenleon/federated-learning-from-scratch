import os
import torch
import pynvml
import numpy as np
from src.models.models import MLP, AlexNet

ModelDict = {
    'mlp': MLP,
    'alexnet': AlexNet
}

def get_best_device(use_cuda: bool) -> torch.device:
    """Dynamically select the vacant CUDA device for running FL experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    # This function is modified by the `get_best_gpu()` in https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/functional.py
    # Shout out to FedLab, which is an incredible FL framework!
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        assert max(gpu_ids) < torch.cuda.device_count()
    else:
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    gpu_memory = np.array(gpu_memory)
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")