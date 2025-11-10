from multiprocessing import Queue, freeze_support
from streaming_process import StreamingProcess
import time

def main():
    q = Queue(maxsize=8)
    producer = StreamingProcess(serial_port="COM16", num_channels=8, window_seconds=5.0, out_queue=q)
    producer.start()
    producer.recording_flag.value = True  # enable before waiting

    try:
        while True:
            if not producer.is_alive():
                raise RuntimeError("Producer exited unexpectedly")

            try:
                item = q.get(timeout=6.5)
                chunk = item["data"]
                print("Got chunk:", chunk.shape, "sr:", item["sr"], "chs:", len(item["channels"]))
            except Exception:
                print("Waiting for chunk...", flush=True)
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        producer.recording_flag.value = False
        producer.stop()
        producer.join(timeout=5.0)

if __name__ == '__main__':
    freeze_support()
    main()
