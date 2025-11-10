# app.py
import time
from collections import deque
from typing import Deque, List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

WORDS = ["YES", "NO", "LEFT", "RIGHT", "UP", "DOWN"]
FS = 256  # Hz
WINDOW_SECONDS = 1.5
BUFFER_LEN = int(FS * WINDOW_SECONDS)
SAMPLES_PER_REFRESH = 24  # how many new samples to pull before each rerun


class NeuropawnEEGSimulator:
    """Lightweight Neuropawn-like EEG simulator (single synthetic channel)."""

    def __init__(self, fs: int = FS, noise_std: float = 8.0):
        self.fs = fs
        self.noise_std = noise_std
        self._step = 0

    def next_sample(self) -> float:
        t = self._step / self.fs
        alpha = 60 * np.sin(2 * np.pi * 10 * t)
        beta = 30 * np.sin(2 * np.pi * 18 * t + 0.8)
        drift = 5 * np.sin(2 * np.pi * 1 * t)
        noise = np.random.normal(0.0, self.noise_std)
        self._step += 1
        return alpha + beta + drift + noise


class SimpleWordDecoder:
    """Placeholder model that maps EEG energy to one of the target words."""

    def __init__(self, labels: List[str]):
        self.labels = labels

    def __call__(self, signal: np.ndarray) -> str:
        spectrum = np.abs(np.fft.rfft(signal))
        band_energy = np.log10(np.sum(spectrum**2) + 1e-6)
        idx = int(abs(band_energy * 3)) % len(self.labels)
        return self.labels[idx]


def init_state() -> None:
    if "simulator" not in st.session_state:
        st.session_state.simulator = NeuropawnEEGSimulator()
    if "decoder" not in st.session_state:
        st.session_state.decoder = SimpleWordDecoder(WORDS)
    if "buffer" not in st.session_state:
        st.session_state.buffer: Deque[float] = deque([0.0] * BUFFER_LEN, maxlen=BUFFER_LEN)
    if "running" not in st.session_state:
        st.session_state.running = False
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = "—"


def toggle_stream() -> None:
    st.session_state.running = not st.session_state.running


def draw_waveform(signal: np.ndarray, placeholder: st.delta_generator.DeltaGenerator) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, color="#00bcd4", linewidth=1)
    ax.set_ylim(-120, 120)
    ax.set_title("Simulated Neuropawn EEG (1.5 s window)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(alpha=0.2)
    placeholder.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Neuropawn Live Decoder", layout="wide")
    init_state()

    st.title("Neuropawn Live Decoder")
    st.caption("Streaming simulated EEG from a Neuropawn board and decoding the predicted word.")

    sidebar_status = st.sidebar.empty()
    st.sidebar.button(
        "▶️ Start stream" if not st.session_state.running else "⏹ Stop stream",
        on_click=toggle_stream,
        use_container_width=True,
    )

    chart_placeholder = st.empty()
    prediction_placeholder = st.empty()

    if st.session_state.running:
        sidebar_status.success("Streaming simulated Neuropawn EEG…")
        for _ in range(SAMPLES_PER_REFRESH):
            st.session_state.buffer.append(st.session_state.simulator.next_sample())
        window = np.asarray(st.session_state.buffer)
        st.session_state.last_prediction = st.session_state.decoder(window)
        draw_waveform(window, chart_placeholder)
        prediction_placeholder.markdown(
            f"""
            <div style="text-align:center; background-color:#262730; color:white; 
                        padding:20px; border-radius:12px;">
                <p style="margin:0; font-size:16px;">Predicted Word</p>
                <p style="margin:0; font-size:48px; font-weight:700;">{st.session_state.last_prediction}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.05)
        st.rerun()
    else:
        sidebar_status.info("Stream paused. Press Start to generate live data.")
        window = np.asarray(st.session_state.buffer)
        draw_waveform(window, chart_placeholder)
        prediction_placeholder.markdown(
            f"""
            <div style="text-align:center; background-color:#37474f; color:white; 
                        padding:20px; border-radius:12px;">
                <p style="margin:0; font-size:16px;">Predicted Word</p>
                <p style="margin:0; font-size:48px; font-weight:700;">{st.session_state.last_prediction}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()


if __name__ == "__main__":
    main()
