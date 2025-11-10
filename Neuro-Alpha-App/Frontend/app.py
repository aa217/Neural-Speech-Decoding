"""
NATHACKS Streamlit UI — refreshed mock demo.

Current behavior:
- Test mode ON (default): generates fake EEG + probabilities and streams them while
  Start is pressed.
- Test mode OFF: placeholder messaging (device integration will be re-added later).
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import backend helper
try:
    from Utilities.tester import run_trials, TrialResult
except Exception:
    run_trials = None
    TrialResult = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSES: List[str] = ["Food", "Water", "Background Noise"]
CHANNELS = 8
SAMPLE_RATE = 125  # Hz (mock data only for now)
WINDOW_SECONDS = 5
SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
def generate_mock_probs(focus_label: str) -> Dict[str, float]:
    focus_prob = np.random.uniform(0.60, 0.70)
    remaining = max(0.0, 1.0 - focus_prob)
    other_labels = [c for c in CLASSES if c != focus_label]
    weights = np.random.rand(len(other_labels))
    weights = weights / weights.sum() if weights.sum() else np.full(len(other_labels), 1/len(other_labels))
    probs = {focus_label: focus_prob}
    for label, weight in zip(other_labels, weights):
        probs[label] = remaining * float(weight)
    return probs


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
    test_mode: bool = False
    word_probs: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CLASSES})
    eeg_data: Optional[np.ndarray] = None
    transcript: str = "Press Start to begin."
    last_update: str = "Never"
    status_msg: str = ""
    focus_label: str = CLASSES[0]
    device_enabled: bool = run_trials is not None


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
    if not STATE.device_enabled:
        st.sidebar.error("Device mode unavailable — Utilities backend missing.", icon="⚠️")
    else:
        st.sidebar.info("Device mode enabled — Start will run `run_trials` and display averaged data.", icon="ℹ️")
current_index = CLASSES.index(STATE.focus_label) if STATE.focus_label in CLASSES else 0
STATE.focus_label = st.sidebar.selectbox("Top prediction word", CLASSES, index=current_index)


def run_mock_cycle() -> None:
    STATE.word_probs = generate_mock_probs(STATE.focus_label)
    STATE.eeg_data = generate_mock_eeg()
    top_label = max(STATE.word_probs, key=STATE.word_probs.get)
    STATE.transcript = f"Predicted: {top_label}"
    STATE.last_update = time.strftime("%H:%M:%S")
    STATE.status_msg = "Recording complete — snapshot ready."


def capture_device_snapshot() -> None:
    if run_trials is None:
        STATE.status_msg = "Backend utilities unavailable."
        return
    STATE.status_msg = "Recording from device…"
    try:
        result = run_trials(verbose=False)
    except Exception as exc:
        STATE.status_msg = f"Device recording failed: {exc}"
        STATE.eeg_data = None
        STATE.word_probs = {c: 0.0 for c in CLASSES}
        return

    if result.avg_probs is not None:
        STATE.word_probs = {label: float(result.avg_probs[idx]) for idx, label in enumerate(CLASSES)}
        top_idx = int(np.argmax(result.avg_probs))
        STATE.transcript = f"Detected: {CLASSES[top_idx]}"
        probs_text = ", ".join(f"{CLASSES[i]}: {result.avg_probs[i]:.2f}" for i in range(len(CLASSES)))
        STATE.status_msg = f"Device snapshot ready — avg probs [{probs_text}]"
    else:
        STATE.word_probs = {c: 0.0 for c in CLASSES}
        STATE.transcript = "No predictions collected."
        STATE.status_msg = "Device snapshot ready — no probabilities available."

    if result.avg_chunk is not None:
        STATE.eeg_data = result.avg_chunk
    else:
        STATE.eeg_data = None

    STATE.last_update = time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------
status_badge = "Recording…" if STATE.running else ("Ready" if STATE.eeg_data is not None else "Idle")
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
            STATE.word_probs = {c: 0.0 for c in CLASSES}
            STATE.eeg_data = None
            STATE.transcript = "Recording EEG… please hold still."
            STATE.last_update = "Recording…"
        else:
            if not STATE.device_enabled:
                STATE.status_msg = "Backend unavailable. Please install Utilities dependencies or switch to Test mode."
            else:
                STATE.running = True
                STATE.word_probs = {c: 0.0 for c in CLASSES}
                STATE.eeg_data = None
                STATE.transcript = "Recording from device…"
                STATE.last_update = "Recording…"
                capture_device_snapshot()
                STATE.running = False
with right_btn:
    if st.button("Stop", use_container_width=True, disabled=not STATE.running):
        STATE.running = False
        if STATE.test_mode:
            run_mock_cycle()
        else:
            if not STATE.device_enabled:
                STATE.status_msg = "Backend unavailable."
            else:
                capture_device_snapshot()
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
    if STATE.running:
        display = "<div class='bigtext'>Recording…</div>"
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
    if STATE.running:
        st.info("Recording EEG… please hold still.")
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
