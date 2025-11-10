"""
Streamlit UI mock for NATHACKS.

Goals (per request):
1. Start/Stop controls only manage UI state (no backend dependency).
2. Three placeholders ("Food", "Water", "Background Noise") show class probabilities
   once Start is pressed.
3. Pressing Start also reveals fake EEG data from 8 channels so the page looks alive.
"""

import time
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ---- Constants ---------------------------------------------------------------
PROB_LABELS: List[str] = ["Food", "Water", "Background Noise"]
CHANNELS = 8
SAMPLE_RATE = 250  # Hz
WINDOW_SECONDS = 2.0
SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)


# ---- Helpers -----------------------------------------------------------------
def generate_mock_probs() -> Dict[str, float]:
    raw = np.random.rand(len(PROB_LABELS))
    probs = raw / raw.sum()
    return {label: float(val) for label, val in zip(PROB_LABELS, probs)}


def generate_mock_eeg(samples: int = SAMPLES, channels: int = CHANNELS) -> np.ndarray:
    """Simple multi-channel sine + noise signal for visualization."""
    t = np.linspace(0, WINDOW_SECONDS, samples, endpoint=False)
    waves = []
    for idx in range(channels):
        freq = 8 + idx * 0.9  # slightly different tone per channel
        wave = 0.8 * np.sin(2 * np.pi * freq * t)
        modulation = 0.3 * np.sin(2 * np.pi * 1.5 * t + idx)
        noise = 0.25 * np.random.randn(samples)
        waves.append(wave + modulation + noise)
    return np.stack(waves, axis=1)


def refresh_mock_stream(state: "UIState") -> None:
    state.running = True
    state.word_probs = generate_mock_probs()
    state.eeg_data = generate_mock_eeg()
    top_label = max(state.word_probs, key=state.word_probs.get)
    state.transcript = f"Predicted: {top_label}"
    state.last_update = time.strftime("%H:%M:%S")


# ---- Session state -----------------------------------------------------------
class UIState:
    def __init__(self) -> None:
        self.running = False
        self.word_probs: Dict[str, float] = {label: 0.0 for label in PROB_LABELS}
        self.eeg_data: np.ndarray | None = None
        self.transcript = "Press Start to decode."
        self.last_update = "Never"


if "ui_state" not in st.session_state:
    st.session_state.ui_state = UIState()

STATE: UIState = st.session_state.ui_state


# ---- Page chrome -------------------------------------------------------------
st.set_page_config(page_title="NATHACKS — EEG Demo", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --card-bg: rgba(255,255,255,0.04);
      --card-border: rgba(255,255,255,0.12);
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .bigtext { font-size: 28px; font-weight: 700; }
    .small { color: #9ca3af; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Layout ------------------------------------------------------------------
st.markdown(
    """
    <div class='card'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
          <div class='bigtext'>Live Decoding</div>
          <div class='small'>Start to see mock predictions + 8-channel EEG.</div>
        </div>
        <span class='small'>{}</span>
      </div>
    </div>
    """.format("Running" if STATE.running else "Idle"),
    unsafe_allow_html=True,
)

controls = st.columns([1, 1, 1])
with controls[0]:
    if st.button("Start", use_container_width=True):
        refresh_mock_stream(STATE)
with controls[1]:
    if st.button("Stop", use_container_width=True, disabled=not STATE.running):
        STATE.running = False
        STATE.transcript = "Stopped."
with controls[2]:
    st.markdown(
        f"<div class='card' style='margin-bottom:0;'>"
        f"<div class='small'>Last update: {STATE.last_update}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# Probability placeholders
prob_cols = st.columns(len(PROB_LABELS))
for col, label in zip(prob_cols, PROB_LABELS):
    value = STATE.word_probs.get(label, 0.0) * 100
    col.markdown(
        f"<div class='card'>"
        f"<div class='small'>{label}</div>"
        f"<div class='bigtext'>{value:05.2f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# Visual row
viz_left, viz_right = st.columns([2, 1])
with viz_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("8-channel EEG", divider=False)
    if STATE.running and STATE.eeg_data is not None:
        df = pd.DataFrame(STATE.eeg_data, columns=[f"Ch {i+1}" for i in range(CHANNELS)])
        st.line_chart(df, height=280, use_container_width=True)
    else:
        st.info("Press Start to stream placeholder EEG data.")
    st.markdown("</div>", unsafe_allow_html=True)

with viz_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Transcript", divider=False)
    st.write(STATE.transcript)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    f"Mock settings • channels: {CHANNELS} • sample rate: {SAMPLE_RATE} Hz • window: {WINDOW_SECONDS}s"
)
