"""Load AIST++ pickles while only allowing numpy array reconstruction (no arbitrary code)."""
import pickle, sys
import numpy as np

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
}

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return getattr(np, name) if module == "numpy" else getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

def load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()

