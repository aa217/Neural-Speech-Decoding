# lstm_eeg_predictor.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessor import PreProcessor 

CLASS_NAMES = ["Food", "Water", "BG-Noise"] 

class EEG_LSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=48, num_layers=2, num_classes=3, dropout=0.60):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.RReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: [B, T, C]
        out, _ = self.lstm(x)                           # [B, T, H]
        scores = self.attn(out).squeeze(-1)             # [B, T]
        weights = torch.softmax(scores, dim=1)          # [B, T]
        out = (out * weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        out = self.ln(out)                              # [B, H]
        return self.fc(out)                             # [B, num_classes]


class SimplePredictor:
    """
    Minimal predictor wrapping your EEG_LSTM:
    - preprocess [T,C] with MindsAI
    - reshape to [1,T,C]
    - forward -> softmax -> return (probs, label)
    """
    def __init__(
        self,
        pth_path: str,
        sr: int,
        channel_order=None,
        input_size: int = 8,
        hidden_size: int = 48,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.60,
        device: str = "cpu",
        tailoring_lambda: float = 1.25e-29,
        class_names=None,
    ):
        self.device = torch.device(device)
        self.class_names = class_names or CLASS_NAMES

        # Preprocessor (matches notebook)
        self.pre = PreProcessor(sr=sr, tailoring_lambda=tailoring_lambda)

        # Build model and load state_dict
        self.model = EEG_LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)
        state = torch.load(pth_path, map_location=self.device)
        # Support both raw state_dict and {"state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True) 
        self.model.eval()

        self._no_grad = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

    def predict(self, chunk_TxC: np.ndarray):
        """
        chunk_TxC: np.ndarray [T, C] float32 (5 s window)
        returns: (probs np.ndarray [3], label str)
        """
        x = self.pre.transform(chunk_TxC)

        x_t = torch.from_numpy(x[None, ...]).float().to(self.device)

        with self._no_grad():
            logits = self.model(x_t)                   
            probs = F.softmax(logits, dim=-1)[0]       
            probs = probs.detach().cpu().numpy().astype(np.float32)

        y_idx = int(np.argmax(probs))
        return probs, self.class_names[y_idx]
