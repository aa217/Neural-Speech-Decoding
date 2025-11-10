# streaming_process.py
import time
import numpy as np
from multiprocessing import Process, Event, Value, Queue
from .neurokit_connector import NeuroPawnConnector


class StreamingProcess(Process):
    def __init__(
        self,
        serial_port: str,
        num_channels: int = 8,
        window_seconds: float = 5.0,
        out_queue: Queue = None,
        start_recording: bool = False,
        buffer_size: int = 450000,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.serial_port = serial_port
        self.num_channels = int(num_channels)
        self.window_seconds = float(window_seconds)
        self.buffer_size = int(buffer_size)
        self.out_queue = out_queue or Queue(maxsize=8)
        self.recording_flag = Value('b', start_recording)
        self._running = Event()
        self._running.set()

    def run(self):
        conn = NeuroPawnConnector(self.serial_port, self.num_channels, self.buffer_size)
        conn.start_stream()
        sr = conn.sr
        samples_per_win = max(1, int(self.window_seconds * sr))

        # In StreamingProcess.run()
        last_emit_ts = 0.0
        ...
        while self._running.is_set():
            if self.recording_flag.value:
                # Wait for full window
                while self._running.is_set():
                    if conn.board_shim.get_board_data_count() >= samples_per_win:
                        break
                    time.sleep(0.01)
                if not self._running.is_set():
                    break

                now = time.time()
                if now - last_emit_ts < self.window_seconds:
                    time.sleep(0.05)
                    continue

                eeg = conn.get_window(self.window_seconds)  
                if eeg.shape[1] >= samples_per_win and eeg.shape[0] > 0:
                    chunk = np.asarray(eeg.T, dtype=np.float32)
                    payload = {"sr": sr, "channels": conn.eeg_channels, "data": chunk, "t_emit": now}
                    try:
                        self.out_queue.put_nowait(payload)
                        print(f"[producer] {time.strftime('%H:%M:%S', time.localtime(now))} emitted {chunk.shape}", flush=True)
                        last_emit_ts = now
                    except Exception:
                        try:
                            _ = self.out_queue.get_nowait()
                            self.out_queue.put_nowait(payload)
                            last_emit_ts = now
                        except Exception:
                            pass
                time.sleep(0.01)
            else:
                time.sleep(0.05)


    def stop(self):
        self._running.clear()
