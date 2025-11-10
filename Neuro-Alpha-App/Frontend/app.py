"""
Streamlit UI for NATHACKS.

Two data sources are available:
1. Test mode (default): generate fake probabilities + EEG to demo the UI.
2. Device mode: pull real windows + predictions from Utilities.tester.TesterStream.
"""

import sys
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---- Optional backend hook ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Utilities.tester import TesterStream, main as tester_main  # noqa: F401
except Exception as exc:  # pragma: no cover - import best-effort
    tester_main = None
    TesterStream = None
    _tester_import_error = exc
else:  # pragma: no cover
    _tester_import_error = None


# ---- Constants ---------------------------------------------------------------
PROB_LABELS: List[str] = ["Food", "Water", "Background Noise"]
CHANNELS = 8
SAMPLE_RATE = 250  # Hz (mock)
WINDOW_SECONDS = 2.0  # mock window
SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
DEFAULT_SERIAL = "/dev/cu.usbserial-FTB6SPL3"
DEFAULT_WINDOW_BACKEND = 5.0


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
    state.word_probs = generate_mock_probs()
    state.eeg_data = generate_mock_eeg()
    top_label = max(state.word_probs, key=state.word_probs.get)
    state.transcript = f"Predicted: {top_label}"
    state.last_update = time.strftime("%H:%M:%S")


def ensure_backend_stream(state: "UIState") -> tuple[bool, Optional[str]]:
    if TesterStream is None:
        msg = "Backend utilities unavailable."
        if _tester_import_error:
            msg += f" ({_tester_import_error})"
        return False, msg
    if state.backend is None:
        state.backend = TesterStream(
            serial_port=state.serial_port,
            num_channels=CHANNELS,
            window_seconds=state.backend_window,
        )
        try:
            state.backend.start()
        except Exception as exc:  # pragma: no cover
            state.backend = None
            return False, f"Failed to start backend: {exc}"
    return True, None


def update_from_backend(state: "UIState", payload: Optional[dict]) -> None:
    if not payload:
        return
    chunk = np.asarray(payload.get("chunk"))
    if chunk.ndim == 2:
        state.eeg_data = chunk
    probs = payload.get("probs", np.zeros(len(PROB_LABELS)))
    state.word_probs = {
        "Food": float(probs[0]) if len(probs) > 0 else 0.0,
        "Water": float(probs[1]) if len(probs) > 1 else 0.0,
        "Background Noise": float(probs[2]) if len(probs) > 2 else 0.0,
    }
    label = payload.get("label", "Unknown")
    state.transcript = f"Detected: {label}"
    ts = payload.get("timestamp", time.time())
    state.last_update = time.strftime("%H:%M:%S", time.localtime(ts))


def capture_snapshot(state: "UIState") -> tuple[bool, Optional[str]]:
    state.recording_active = True
    if state.test_mode:
        refresh_mock_stream(state)
        state.recording_active = False
        return True, None

    ok, err = ensure_backend_stream(state)
    if not ok:
        state.recording_active = False
        return False, err

    err = None
    payload = None
    if state.backend:
        try:
            payload = state.backend.next(timeout=state.backend_window + 0.5)
        except Exception as exc:
            err = f"Failed to read chunk: {exc}"
    if payload:
        update_from_backend(state, payload)
    else:
        err = err or "No data received from device."

    if state.backend:
        state.backend.stop()
        state.backend = None

    state.recording_active = False
    return (err is None), err


# ---- Session state -----------------------------------------------------------
class UIState:
    def __init__(self) -> None:
        self.recording_active = False
        self.word_probs: Dict[str, float] = {label: 0.0 for label in PROB_LABELS}
        self.eeg_data: np.ndarray | None = None
        self.transcript = "Press Start to capture a snapshot."
        self.last_update = "Never"
        self.test_mode = False
        self.serial_port = DEFAULT_SERIAL
        self.backend_window = DEFAULT_WINDOW_BACKEND
        self.backend: Optional[object] = None

    def clear(self) -> None:
        self.word_probs = {label: 0.0 for label in PROB_LABELS}
        self.eeg_data = None
        self.transcript = "Recording cleared."
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
st.sidebar.header("Input Source")
STATE.test_mode = st.sidebar.checkbox("Test mode (fake data)", value=STATE.test_mode)
if STATE.test_mode and STATE.backend:
    STATE.backend.stop()
    STATE.backend = None

STATE.serial_port = st.sidebar.text_input("Serial port", value=STATE.serial_port)
STATE.backend_window = st.sidebar.slider(
    "Window (sec)", min_value=1.0, max_value=10.0, value=STATE.backend_window, step=0.5
)

status_badge = (
    "Capturing…" if STATE.recording_active else ("Snapshot ready" if STATE.eeg_data is not None else "Idle")
)
stream_mode = "Simulated EEG snapshot" if STATE.test_mode else "Device-backed EEG snapshot"
st.markdown(
    f"""
    <div class='card'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
          <div class='bigtext'>Live Decoding</div>
          <div class='small'>{stream_mode}</div>
        </div>
        <span class='small'>{status_badge}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

controls = st.columns([1, 1, 1])
with controls[0]:
    if st.button("Start Recording", use_container_width=True, disabled=STATE.recording_active):
        ok, err = capture_snapshot(STATE)
        if not ok and err:
            st.error(err)
with controls[1]:
    if st.button("Clear", use_container_width=True):
        STATE.clear()
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
    if STATE.eeg_data is not None:
        df = pd.DataFrame(STATE.eeg_data, columns=[f"Ch {i+1}" for i in range(CHANNELS)])
        st.line_chart(df, height=280, use_container_width=True)
    else:
        st.info(
            "Press Start to capture {} data.".format(
                "fake" if STATE.test_mode else "device-driven"
            )
        )
    st.markdown("</div>", unsafe_allow_html=True)

with viz_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Transcript", divider=False)
    st.write(STATE.transcript)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Channels: {} • mock sample rate: {} Hz • mock window: {}s • backend window: {}s".format(
        CHANNELS, SAMPLE_RATE, WINDOW_SECONDS, STATE.backend_window
    )
)
