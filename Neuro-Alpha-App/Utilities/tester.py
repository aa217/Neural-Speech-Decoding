from dataclasses import dataclass
from multiprocessing import Queue, freeze_support
from pathlib import Path
import time
from typing import Optional

import numpy as np

try:  # allow both package and standalone usage
    from .streaming_process import StreamingProcess  # type: ignore
    from .lstm_eeg_model import SimplePredictor  # type: ignore
except ImportError:
    from streaming_process import StreamingProcess
    from lstm_eeg_model import SimplePredictor


DEFAULT_SERIAL = "/dev/cu.usbserial-FTB6SPL3"
DEFAULT_MODEL = str(
    Path(__file__).resolve().parent / "LSTM_Model" / "lstm_classifier_Water_Food_Bg_Noise.pth"
)


@dataclass
class TrialResult:
    trials: int
    avg_probs: Optional[np.ndarray]
    avg_chunk: Optional[np.ndarray] = None


def run_trials(
    trials: int = 10,
    serial_port: str = DEFAULT_SERIAL,
    num_channels: int = 8,
    window_seconds: float = 5.0,
    model_path: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> TrialResult:
    """
    Collect `trials` windows from StreamingProcess, run SimplePredictor,
    and return the averaged probabilities. Safe to import into app.py.
    """
    q = Queue(maxsize=8)
    producer = StreamingProcess(
        serial_port=serial_port,
        num_channels=num_channels,
        window_seconds=window_seconds,
        out_queue=q,
    )
    producer.start()
    producer.recording_flag.value = True

    predictor = None
    collected = 0
    sum_probs = np.zeros(3, dtype=np.float32)
    sum_chunk = None

    try:
        while collected < trials:
            if not producer.is_alive():
                raise RuntimeError("Producer exited unexpectedly")

            try:
                item = q.get(timeout=6.5)
            except Exception:
                if verbose:
                    print("Waiting for chunk...", flush=True)
                continue

            chunk = np.asarray(item["data"])
            sr = item["sr"]
            channels = item.get("channels")

            if predictor is None:
                predictor = SimplePredictor(
                    pth_path=model_path,
                    sr=sr,
                    channel_order=channels,
                    input_size=num_channels,
                    hidden_size=48,
                    num_layers=2,
                    num_classes=3,
                    dropout=0.60,
                    device="cpu",
                    tailoring_lambda=1.25e-29,
                    class_names=["Food", "Water", "None"],
                )

            probs, label = predictor.predict(chunk)
            sum_probs += probs
            sum_chunk = chunk if sum_chunk is None else sum_chunk + chunk
            collected += 1

            if verbose:
                stamp = time.strftime("%H:%M:%S")
                print(f"[Trial {collected:02d} @ {stamp}] pred={label} probs={np.round(probs, 3)}")

        avg_probs = (sum_probs / collected) if collected else None
        avg_chunk = (sum_chunk / collected) if (collected and sum_chunk is not None) else None
        if verbose:
            if avg_probs is not None:
                print(f"\nAveraged over {collected} trials: {np.round(avg_probs, 3)}")
                if avg_chunk is not None and verbose:
                    print(f"Averaged chunk shape: {avg_chunk.shape}")
            else:
                print("No trials completed; no average available.")
        return TrialResult(trials=collected, avg_probs=avg_probs, avg_chunk=avg_chunk)
    finally:
        producer.recording_flag.value = False
        producer.stop()
        producer.join(timeout=5.0)


def main():
    run_trials()


if __name__ == "__main__":
    freeze_support()
    main()
