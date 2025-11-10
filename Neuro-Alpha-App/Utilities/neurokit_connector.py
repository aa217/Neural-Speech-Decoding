# neuro_pawn_connector.py
import time
from typing import List
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


class NeuroPawnConnector:
    def __init__(self, serial_port: str, num_channels: int = 8, buffer_size: int = 450000):
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.num_channels = int(num_channels)
        self.buffer_size = int(buffer_size)

        self.board_shim = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, self.params)
        self.board_id = self.board_shim.get_board_id()
        self.eeg_channels: List[int] = self.board_shim.get_exg_channels(self.board_id)
        self.sr: int = self.board_shim.get_sampling_rate(self.board_id)
        if len(self.eeg_channels) > self.num_channels:
            self.eeg_channels = self.eeg_channels[: self.num_channels]

        self._streaming = False

    def _safe_config(self, cmd: str):
        """
        Send ASCII command via bytes to avoid UTF-8 decode issues on Windows.
        Ignore any response content.
        """
        try:
            payload = cmd.encode('ascii', errors='strict')
            self.board_shim.config_board_with_bytes(payload)
        except Exception:
            try:
                _ = self.board_shim.config_board(cmd)
            except Exception:
                pass  # ignore any response decoding error

    def start_stream(self):
        # Prepare, stabilize serial, configure, then start stream
        self.board_shim.prepare_session()
        time.sleep(0.5)

        for x in range(1, self.num_channels + 1):
            time.sleep(0.1)
            self._safe_config(f"chon_{x}_12")
            time.sleep(0.1)
            self._safe_config(f"rldadd_{x}")
            time.sleep(0.1)

        self.board_shim.start_stream(self.buffer_size)
        self._streaming = True
        time.sleep(1.0)

    def stop_stream(self):
        if self._streaming:
            self.board_shim.stop_stream()
            self.board_shim.release_session()
            self._streaming = False

    def get_window(self, seconds: float) -> np.ndarray:
        assert self._streaming, "Stream not started"
        n = max(1, int(seconds * self.sr))
        data = self.board_shim.get_current_board_data(n)
        if data.shape[1] == 0:
            return np.empty((len(self.eeg_channels), 0))
        return data[self.eeg_channels, :]
