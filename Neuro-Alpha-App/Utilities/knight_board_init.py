import brainflow as bf
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

class KnightBoard:
    def __init__(self, serial_port: str, num_channels: int):
        """Initialize and configure the Knight Board."""
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.num_channels = num_channels
        
        # Initialize board
        self.board_shim = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, self.params)
        self.board_id = self.board_shim.get_board_id()
        self.eeg_channels = self.board_shim.get_exg_channels(self.board_id)
        self.sr = self.board_shim.get_sampling_rate(self.board_id)
        self.num_points = 200 

    def start_stream(self, buffer_size: int = 450000):
        """Start the data stream from the board."""
        self.board_shim.prepare_session()
        self.board_shim.start_stream(buffer_size)
        print("Stream started.")
        time.sleep(2)
        for x in range(1, self.num_channels + 1):
            time.sleep(0.5)
            cmd = f"chon_{x}_12"
            self.board_shim.config_board(cmd)
            print(f"sending {cmd}")
            time.sleep(1)
            rld = f"rldadd_{x}"
            self.board_shim.config_board(rld)
            print(f"sending {rld}")
            time.sleep(0.5)

    def stop_stream(self):
        """Stop the data stream and release resources."""
        self.board_shim.stop_stream()
        self.board_shim.release_session()
        print("Stream stopped and session released.")
