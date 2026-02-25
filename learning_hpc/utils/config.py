# General
import json
from copy import deepcopy
from pathlib import Path
import numpy as np

# Torch
import torch


# Learning HPC1



class Config:
    """
    The purpose of this class is to robustly save the arguments used
    to create configuration objects in a JSON-serializable format.

    Most of its complexity is just making sure the conversation is robust.
    """
    def _obj2dict(self, obj):
        # Config objects
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        # Basic primitives
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # Path
        if isinstance(obj, Path):
            return str(obj)

        # Python numeric containers
        if isinstance(obj, dict):
            return {str(k): self._obj2dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._obj2dict(v) for v in obj]
        
        # Common package types
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Try JSON encoding
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # Fallback: set as string placeholder
            return f"<nonserializable:{type(obj).__name__}>"

    def to_dict(self):
        out = deepcopy(self.__dict__)
        for k, v in list(out.items()):
            out[k] = self._obj2dict(v)
        return out
    
    def save(self, path: Path):
        cfg = self.to_dict()
        with path.open("w") as f:
            json.dump(cfg, f, indent=4)
