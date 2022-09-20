import os
import pysimplelog
import numpy as np

np.random.seed(0xDEADBEEF)

__version__ = "0.0.1"

logger = pysimplelog.Logger()

__current_path = os.path.dirname(__file__)
default_curvox_dataset_path = os.path.abspath(os.path.join(__current_path, "../../", "data", "curvox_dataset"))
default_meshgraphnet_dataset_path = os.path.abspath(os.path.join(__current_path, "../../", "data", "meshgraphnet_dataset"))
default_training_path = os.path.abspath(os.path.join(__current_path, "../../", "data", "training"))
root_path = os.path.dirname(os.path.realpath(__file__))

__all__ = [
    "evaluation",
    "algorithms",
    "util",
]
