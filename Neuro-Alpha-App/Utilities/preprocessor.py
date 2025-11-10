# pre_processor.py
import numpy as np

from .MindsAI.mindsai_filter_python.core import mindsai_python_filter as mai

class PreProcessor:
    """
    Replicate notebook's preprocessing:
    - Take [samples, channels] float32.
    - Transpose to [channels, samples].
    - Apply MindsAI filter (mindsai_python_filter).
    - Transpose back to [samples, channels].
    """
    def __init__(self, sr: int, tailoring_lambda: float = 1.25e-29):
        self.sr = sr
        self.tailoring_lambda = tailoring_lambda
        if mai is None:
            raise ImportError("mindsai_filter_python not available; install or vendor it to run preprocessing")

    def transform(self, chunk_samples_by_channels: np.ndarray) -> np.ndarray:
        """
        Input: [samples, channels] float32
        Output: model-ready array or same shape if model expects raw channels
        """
        x = chunk_samples_by_channels
        # Ensure 2D
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array [samples, channels], got {x.shape}")
        # Transpose to [channels, samples]
        x_T = x.T.astype(np.float32, copy=False)
        # MindsAI filter expects channels x time
        filtered_T = mai(x_T, self.tailoring_lambda)
        filtered = np.asarray(filtered_T, dtype=np.float32).T 

        return filtered
