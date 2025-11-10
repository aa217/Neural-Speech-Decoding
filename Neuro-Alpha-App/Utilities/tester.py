from multiprocessing import Queue, freeze_support
from streaming_process import StreamingProcess
import time
import numpy as np
from lstm_eeg_model import SimplePredictor

def main():
    q = Queue(maxsize=8)
    producer = StreamingProcess(serial_port="/dev/cu.usbserial-FTB6SPL3", num_channels=8, window_seconds=5.0, out_queue=q)
    producer.start()
    producer.recording_flag.value = True

    predictor = None

    try:
        while True:
            if not producer.is_alive():
                raise RuntimeError("Producer exited unexpectedly")

            try:
                item = q.get(timeout=6.5)
            except Exception:
                print("Waiting for chunk...", flush=True)
                continue

            chunk = item["data"]
            print(chunk)
            sr = item["sr"]
            channels = item.get("channels")

            if predictor is None:
                predictor = SimplePredictor(
                    pth_path="Neuro-Alpha-App/Utilities/LSTM_Model/lstm_classifier_Water_Food_Bg_Noise.pth",
                    sr=sr,
                    channel_order=channels,
                    input_size=8,
                    hidden_size=48,
                    num_layers=2,
                    num_classes=3,
                    dropout=0.60,
                    device="cpu",
                    tailoring_lambda=1.25e-29,
                    class_names=["Food", "Water", "None"],
                )


            probs, label = predictor.predict(chunk)
            print(f"[{time.strftime('%H:%M:%S')}] pred={label} probs={np.round(probs, 3)}")
            break
    except KeyboardInterrupt:
        pass
    finally:
        producer.recording_flag.value = False
        producer.stop()
        producer.join(timeout=5.0)

if __name__ == '__main__':
    freeze_support()
    main()
