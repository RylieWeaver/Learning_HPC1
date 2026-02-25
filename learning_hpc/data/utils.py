# General

# Torch
import torch

# Learning HPC1



def move_to(data, device):
    """
    This function is a recursive helper to move data to a given device.
    
    Mostly this just helps move data dicts, which are helpful to keep 
    track of which tensors represent what inputs.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to(v, device) for v in data)
    else:
        return data
