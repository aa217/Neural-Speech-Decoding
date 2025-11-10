"""
NATHACKS Streamlit UI — refreshed mock demo.

Current behavior:
- Test mode ON (default): generates fake EEG + probabilities and streams them while
  Start is pressed.
- Test mode OFF: placeholder messaging (device integration will be re-added later).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES: List[str] = ["Food", "Water", "Background Noise"]
CHANNELS = 8
SAMPLE_RATE = 125  # Hz (mock data only for now)
WINDOW_SECONDS = 5
SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
WARMUP_SECONDS = 1.5


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
def generate_mock_probs() -> Dict[str, float]:
    raw = np.random.rand(len(CLASSES))
    probs = raw / raw.sum()
    return {label: float(val) for label, val in zip(CLASSES, probs)}


def generate_mock_eeg() -> np.ndarray:
    t = np.linspace(0, WINDOW_SECONDS, SAMPLES, endpoint=False)
    data = []
    for ch in range(CHANNELS):
        base = np.sin(2 * np.pi * (8 + ch) * t)
        mod = 0.4 * np.sin(2 * np.pi * (2 + ch * 0.2) * t + ch)
        noise = 0.35 * np.random.randn(SAMPLES)
        data.append(base + mod + noise)
    return np.stack(data, axis=1)


# ---------------------------------------------------------------------------
# Streamlit state
# ---------------------------------------------------------------------------
@dataclass
class UIState:
    running: bool = False
    test_mode: bool = True
    word_probs: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CLASSES})
    eeg_data: Optional[np.ndarray] = None
    transcript: str = "Press Start to begin."
    last_update: str = "Never"
    status_msg: str = ""
    warmup_start: Optional[float] = None


if "ui_state" not in st.session_state:
    st.session_state.ui_state = UIState()
STATE: UIState = st.session_state.ui_state


# ---------------------------------------------------------------------------
# UI setup
# ---------------------------------------------------------------------------
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

st.sidebar.header("Input Source")
STATE.test_mode = st.sidebar.checkbox("Test mode (fake data)", value=STATE.test_mode)
if not STATE.test_mode:
    st.sidebar.info("Device mode will be re-enabled soon. Keep Test mode on for now.", icon="ℹ️")


def run_mock_cycle() -> None:
    STATE.word_probs = generate_mock_probs()
    STATE.eeg_data = generate_mock_eeg()
    top_label = max(STATE.word_probs, key=STATE.word_probs.get)
    STATE.transcript = f"Predicted: {top_label}"
    STATE.last_update = time.strftime("%H:%M:%S")
    STATE.warmup_start = None


def is_warming_up() -> bool:
    if not STATE.running or not STATE.test_mode:
        return False
    if STATE.warmup_start is None:
        return False
    return (time.time() - STATE.warmup_start) < WARMUP_SECONDS


# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------
if STATE.running and is_warming_up():
    status_badge = "Warming up"
else:
    status_badge = "Running" if STATE.running else ("Ready" if STATE.eeg_data is not None else "Idle")
mode_text = "Simulated EEG stream" if STATE.test_mode else "Device-backed stream (coming soon)"
st.markdown(
    f"""
    <div class='card'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
          <div class='bigtext'>Live Decoding</div>
          <div class='small'>{mode_text}</div>
        </div>
        <span class='small'>{status_badge}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_btn, right_btn, meta = st.columns([1, 1, 1])
with left_btn:
    if st.button("Start", use_container_width=True, disabled=STATE.running):
        if STATE.test_mode:
            STATE.running = True
            STATE.status_msg = ""
            STATE.warmup_start = time.time()
            STATE.word_probs = {c: 0.0 for c in CLASSES}
            STATE.eeg_data = None
            STATE.transcript = "Model warming up..."
            STATE.last_update = "Calibrating..."
        else:
            STATE.status_msg = "Device mode not available yet. Please enable Test mode."
with right_btn:
    if st.button("Stop", use_container_width=True, disabled=not STATE.running):
        STATE.running = False
        STATE.status_msg = "Stopped."
        STATE.warmup_start = None
with meta:
    st.markdown(
        f"<div class='card' style='margin-bottom:0;'>"
        f"<div class='small'>Last update: {STATE.last_update}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

if STATE.status_msg:
    st.info(STATE.status_msg)


# ---------------------------------------------------------------------------
# Probability cards
# ---------------------------------------------------------------------------
prob_cols = st.columns(3)
for col, label in zip(prob_cols, CLASSES):
    if is_warming_up():
        display = "<div class='bigtext'>…</div>"
    else:
        value = STATE.word_probs.get(label, 0.0) * 100
        display = f"<div class='bigtext'>{value:05.2f}%</div>"
    col.markdown(
        f"<div class='card'>"
        f"<div class='small'>{label}</div>"
        f"{display}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------
viz_left, viz_right = st.columns([2, 1])
with viz_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("8-channel EEG", divider=False)
    if is_warming_up():
        st.info("Calibrating… generating a fresh window.")
    elif STATE.eeg_data is not None:
        df = pd.DataFrame(STATE.eeg_data, columns=[f"Ch {i+1}" for i in range(CHANNELS)])
        st.line_chart(df, height=280, use_container_width=True)
    else:
        st.info("Press Start to stream fake EEG data.")
    st.markdown("</div>", unsafe_allow_html=True)

with viz_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Transcript", divider=False)
    st.write(STATE.transcript)
    st.markdown("</div>", unsafe_allow_html=True)


st.caption(f"Mock setup • channels: {CHANNELS} • sample rate: {SAMPLE_RATE} Hz • window: {WINDOW_SECONDS}s")


# ---------------------------------------------------------------------------
# Auto-refresh loop (test mode only)
# ---------------------------------------------------------------------------
if STATE.running:
    if STATE.test_mode:
        delay = 0.4 if is_warming_up() else 0.7
        if not is_warming_up():
            run_mock_cycle()
        time.sleep(delay)
        rerun = getattr(st, "rerun", None)
        if callable(rerun):
            rerun()
    else:
        # Device mode placeholder
        STATE.running = False
        STATE.status_msg = "Device mode support will return soon."
