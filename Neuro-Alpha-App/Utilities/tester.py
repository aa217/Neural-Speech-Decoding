from multiprocessing import Queue, freeze_support
from queue import Empty
from pathlib import Path
import time
import numpy as np

from .streaming_process import StreamingProcess
from .lstm_eeg_model import SimplePredictor

MODEL_PATH = Path(__file__).resolve().parent / "LSTM_Model" / "lstm_classifier_Water_Food_Bg_Noise.pth"


class TesterStream:
    """Reusable bridge between StreamingProcess and SimplePredictor."""

    def __init__(
        self,
        serial_port: str = "COM5",
        num_channels: int = 8,
        window_seconds: float = 5.0,
        model_path: Path | str = MODEL_PATH,
    ) -> None:
        self.serial_port = serial_port
        self.num_channels = num_channels
        self.window_seconds = window_seconds
        self.model_path = str(model_path)

        self._queue: Queue | None = None
        self._process: StreamingProcess | None = None
        self._predictor: SimplePredictor | None = None
        self._channel_order = None
        self._sr = None

    def start(self) -> None:
        if self._process:
            return
        self._queue = Queue(maxsize=8)
        self._process = StreamingProcess(
            serial_port=self.serial_port,
            num_channels=self.num_channels,
            window_seconds=self.window_seconds,
            out_queue=self._queue,
        )
        self._process.start()
        self._process.recording_flag.value = True

    def stop(self) -> None:
        if not self._process:
            return
        try:
            self._process.recording_flag.value = False
        except Exception:
            pass
        try:
            self._process.stop()
            self._process.join(timeout=5.0)
        except Exception:
            pass
        self._process = None
        self._queue = None
        self._predictor = None
        self._channel_order = None
        self._sr = None

    def next(self, timeout: float = 0.5):
        """
        Returns the next chunk + prediction dict:
        {
            "chunk": np.ndarray [T,C],
            "probs": np.ndarray [3],
            "label": str,
            "timestamp": float
        }
        """
        if not self._process or not self._queue:
            raise RuntimeError("TesterStream.start() must be called before next().")

        try:
            item = self._queue.get(timeout=timeout)
        except Empty:
            return None

        chunk = item["data"]
        if self._predictor is None:
            self._sr = item["sr"]
            self._channel_order = item.get("channels")
            self._predictor = SimplePredictor(
                pth_path=self.model_path,
                sr=self._sr,
                channel_order=self._channel_order,
                input_size=self.num_channels,
                hidden_size=48,
                num_layers=2,
                num_classes=3,
                dropout=0.60,
                device="cpu",
                tailoring_lambda=1.25e-29,
                class_names=["Food", "Water", "None"],
            )

        probs, label = self._predictor.predict(chunk)
        return {
            "chunk": chunk,
            "probs": probs,
            "label": label,
            "timestamp": time.time(),
            "sr": item.get("sr"),
            "channels": item.get("channels"),
        }

    def __del__(self):
        self.stop()


def main():
    stream = TesterStream()
    stream.start()
    try:
        while True:
            payload = stream.next(timeout=6.5)
            if payload is None:
                print("Waiting for chunk...", flush=True)
                time.sleep(0.5)
                continue
            chunk = payload["chunk"]
            print(chunk)
            probs = payload["probs"]
            label = payload["label"]
            print(f"[{time.strftime('%H:%M:%S')}] pred={label} probs={np.round(probs, 3)}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()


if __name__ == '__main__':
    freeze_support()
    main()
