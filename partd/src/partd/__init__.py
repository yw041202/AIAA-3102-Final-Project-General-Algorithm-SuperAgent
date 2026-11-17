"""Part D tools. Final candidate selection belongs to the calling Agent."""
__version__ = "0.2.0"

from .training import train
from .prediction import predict

__all__ = ["train", "predict", "__version__"]
