# app.py — NATHACKS simple Python frontend (Streamlit)
# ---------------------------------------------------
# Visuals mirror the earlier React mock: tabs (Demo/Data/Train/Settings),
# cards, big transcript, signal & latency metrics, spectrogram + waveform.
# This app can run in two modes:
#  1) Device mode (uses Utilities.streaming_process & BrainFlow)
#  2) Simulate mode (no device required)
#
# How it works:
#  - We read EEG windows (T x C) via a Queue-like producer (device or simulator)
#  - We run the provided LSTM model (Utilities/lstm_eeg_model.SimplePredictor)
#  - We display live predictions and simple metrics
#
# Folder layout expected:
#  - Place this folder next to your 'Utilities' folder (from your zip)
#    e.g.:
#      project_root/
#        Utilities/
#        nathacks_streamlit/ (this folder)
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# If you don't have the device or BrainFlow drivers, use Simulate mode.

import os
import sys
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---- Attempt to import from ../Utilities ------------------------------------
DEFAULT_UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Utilities"))
if os.path.isdir(DEFAULT_UTILS_PATH) and DEFAULT_UTILS_PATH not in sys.path:
    sys.path.insert(0, DEFAULT_UTILS_PATH)

# Lazy flags for optional modules
_have_backend = True
try:
    from streaming_process import StreamingProcess  # device producer
    from lstm_eeg_model import SimplePredictor, CLASS_NAMES
except Exception as e:
    _have_backend = False
    _backend_err = e

# ---- Streamlit page config & CSS --------------------------------------------
st.set_page_config(
    page_title="NATHACKS — Live EEG Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
:root {
  --card-bg: rgba(255,255,255,0.03);
  --card-border: rgba(255,255,255,0.10);
}
.block-container { padding-top: 1.2rem; }
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 16px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 4px 8px;
  background: rgba(255,255,255,0.10);
  color: #e5e7eb;
  font-size: 12px;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 2px 8px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.1);
  font-size: 11px;
}
.bigtext {
  font-size: 28px;
  font-weight: 700;
}
.small {
  color: #9ca3af;
  font-size: 12px;
}
.hr { border-top: 1px solid rgba(255,255,255,0.1); margin: 12px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---- Helpers -----------------------------------------------------------------
@dataclass
class AppState:
    connected: bool = False
    decoding: bool = False
    transcript: str = ""
    signal_quality: float = 0.0
    latency_ms: float = 0.0
    window_seconds: float = 5.0
    sr: int = 250
    channels: int = 8
    session_tokens: List[str] = field(default_factory=list)
    producer_thread: Optional[threading.Thread] = None
    producer_stop: threading.Event = field(default_factory=threading.Event)
    q: Queue = field(default_factory=lambda: Queue(maxsize=8))
    predictor: Optional[object] = None
    simulate: bool = True
    model_pth: str = ""

if "state" not in st.session_state:
    st.session_state.state = AppState()

S = st.session_state.state

# ---- Producer implementations ------------------------------------------------
def device_producer(q: Queue, stop_evt: threading.Event, serial_port: str, channels: int, window_seconds: float):
    """Wrap StreamingProcess (multiprocessing) as a thread controller."""
    if not _have_backend:
        st.error(f"Backend imports failed: {_backend_err}")
        return
    from multiprocessing import Queue as MPQueue, freeze_support
    freeze_support()
    mp_q = MPQueue(maxsize=8)
    proc = StreamingProcess(serial_port=serial_port, num_channels=channels, window_seconds=window_seconds, out_queue=mp_q)
    proc.start()
    proc.recording_flag.value = True
    try:
        while not stop_evt.is_set():
            try:
                payload = mp_q.get(timeout=0.1)  # {sr, channels, data: [T,C], t_emit}
                q.put_nowait(payload)
            except Exception:
                pass
            time.sleep(0.01)
    finally:
        proc.recording_flag.value = False
        try:
            proc.stop()
            proc.join(timeout=3.0)
        except Exception:
            pass

def simulate_producer(q: Queue, stop_evt: threading.Event, sr: int, channels: int, window_seconds: float):
    T = int(sr * window_seconds)
    t = np.arange(T) / sr
    w1, w2 = 10.0, 22.0
    while not stop_evt.is_set():
        base = 0.7*np.sin(2*np.pi*w1*t) + 0.4*np.sin(2*np.pi*w2*t)
        noise = 0.6*np.random.randn(T)
        sig = (base + noise).astype(np.float32)
        data = np.stack([np.roll(sig, i*13) for i in range(channels)], axis=1).astype(np.float32)  # [T,C]
        payload = {"sr": sr, "channels": list(range(channels)), "data": data, "t_emit": time.time()}
        try:
            q.put_nowait(payload)
        except Exception:
            try:
                _ = q.get_nowait()
                q.put_nowait(payload)
            except Exception:
                pass
        time.sleep(max(0.01, window_seconds/2))

# ---- Prediction --------------------------------------------------------------
def ensure_predictor(model_pth: str, sr: int, channels: int):
    if not _have_backend:
        raise RuntimeError(f"Backend imports failed: {_backend_err}")
    if S.predictor is not None:
        return S.predictor
    # class names come from Utilities.lstm_eeg_model
    S.predictor = SimplePredictor(
        pth_path=model_pth,
        sr=sr,
        channel_order=None,
        input_size=channels,
        hidden_size=48,
        num_layers=2,
        num_classes=len(CLASS_NAMES),
        dropout=0.60,
        device="cpu",
        tailoring_lambda=1.25e-29,
        class_names=CLASS_NAMES,
    )
    return S.predictor

def quick_signal_quality(chunk_TxC: np.ndarray) -> float:
    # crude metric: normalized std across channels (0..100)
    if chunk_TxC.size == 0:
        return 0.0
    s = float(np.std(chunk_TxC))
    score = max(0.0, min(100.0, 20.0 * s))  # tuned for demo
    return score

# ---- UI: Sidebar -------------------------------------------------------------
st.sidebar.header("NATHACKS — Control Panel")
S.simulate = st.sidebar.checkbox("Simulate mode (no device)", value=True)
serial_port = st.sidebar.text_input("Serial Port (device mode)", value="COM16" if os.name == "nt" else "/dev/tty.usbserial" )
S.window_seconds = st.sidebar.slider("Window (sec)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
S.sr = int(st.sidebar.number_input("Sample Rate (Hz)", min_value=100, max_value=1000, value=250, step=50))
S.channels = int(st.sidebar.number_input("Channels", min_value=4, max_value=64, value=8, step=1))
default_pth = os.path.abspath(os.path.join(DEFAULT_UTILS_PATH, "LSTM_Model", "lstm_classifier_Water_Food_Bg_Noise.pth"))
S.model_pth = st.sidebar.text_input("Model .pth path", value=default_pth)
st.sidebar.write("\n")
if st.sidebar.button("Connect" if not S.connected else "Disconnect", use_container_width=True):
    if not S.connected:
        # start producer
        S.producer_stop = threading.Event()
        if S.simulate:
            S.producer_thread = threading.Thread(target=simulate_producer, args=(S.q, S.producer_stop, S.sr, S.channels, S.window_seconds), daemon=True)
        else:
            if not _have_backend:
                st.sidebar.error("Backend not available; turn on simulate mode.")
            else:
                S.producer_thread = threading.Thread(target=device_producer, args=(S.q, S.producer_stop, serial_port, S.channels, S.window_seconds), daemon=True)
        if S.producer_thread:
            S.producer_thread.start()
            S.connected = True
            S.decoding = False
            S.transcript = ""
            S.session_tokens = []
    else:
        # stop producer
        S.producer_stop.set()
        S.connected = False
        S.decoding = False

st.sidebar.write("\n")
if st.sidebar.button("Start" if not S.decoding else "Pause", use_container_width=True, disabled=not S.connected):
    S.decoding = not S.decoding

# ---- Tabs --------------------------------------------------------------------
tabs = st.tabs(["Demo", "Data", "Train", "Settings"])

# ---- DEMO TAB ---------------------------------------------------------------
with tabs[0]:
    st.markdown(
        """
<div class='card'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <div class='bigtext'>Live Decoding</div>
      <div class='small'>Non‑invasive EEG → label prediction in real‑time.</div>
    </div>
    <span class='badge'>Showcase Mode</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Metric cards row
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.caption("EEG Device")
        st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<div class='small'>Battery 78% • Firmware 1.2.3</div>"
                    f"<span class='pill'>{'Connected' if S.connected else 'Disconnected'}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.caption("Signal Quality")
        st.markdown(f"<div class='bigtext'>{int(S.signal_quality)}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>higher is better</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.caption("Latency")
        st.markdown(f"<div class='bigtext'>{int(S.latency_ms)} ms</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>model end‑to‑end</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Viz row
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("EEG Spectrogram (mock)", divider=False)
        fig, ax = plt.subplots()
        ax.set_ylabel("freq"); ax.set_xlabel("time")
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Waveform (mock)", divider=False)
        fig2, ax2 = plt.subplots()
        ax2.set_ylabel("amp"); ax2.set_xlabel("samples")
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Transcript", divider=False)
        text_sz = 28
        st.markdown(f"<div style='border:1px solid var(--card-border);border-radius:12px;padding:12px;font-weight:600;font-size:{text_sz}px;'>"
                    f"{S.transcript if S.transcript else ('listening…' if (S.connected and S.decoding) else 'Press Start to decode')}"
                    f"</div>", unsafe_allow_html=True)
        ccc1, ccc2 = st.columns([1,1])
        with ccc1:
            if st.button("Clear", use_container_width=True):
                S.transcript = ""
                S.session_tokens = []
        with ccc2:
            st.button("Speak (TTS)", use_container_width=True, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Control bar
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    left, right = st.columns([2,1])
    with left:
        st.write("Device ready. Press Start." if (S.connected and not S.decoding) else ("Decoding active. Show your best phrase!" if (S.connected and S.decoding) else "Connect device to begin."))
    with right:
        st.button("Save Demo", use_container_width=True, disabled=not S.connected)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- DATA TAB ---------------------------------------------------------------
with tabs[1]:
    st.subheader("Datasets", divider=False)
    st.caption("Manage recent windows (preview only in this prototype)")
    if hasattr(S, 'last_chunks'):
        for i, (shape, ts) in enumerate(S.last_chunks[-4:][::-1]):
            st.markdown(f"- Chunk {i+1}: shape={shape}, time={time.strftime('%H:%M:%S', time.localtime(ts))}")
    else:
        st.info("No data yet. Connect and Start on the Demo tab.")

# ---- TRAIN TAB --------------------------------------------------------------
with tabs[2]:
    st.subheader("Training", divider=False)
    st.caption("Kick off and monitor model runs (placeholder)")
    if st.button("Start training"):
        for i in range(11):
            st.progress(i/10.0, text=f"Epoch {i}/10 — val acc {(26+i*6.2):.1f}%")
            time.sleep(0.4)

# ---- SETTINGS TAB -----------------------------------------------------------
with tabs[3]:
    st.subheader("Settings", divider=False)
    st.caption("Appearance & integrations (placeholder)")
    st.write({"sr": S.sr, "channels": S.channels, "window": S.window_seconds})
    st.write({"simulate": S.simulate, "model_pth": S.model_pth})

# ---- Main event loop: pull from queue, predict, update UI -------------------
def ensure_history():
    if not hasattr(S, 'last_chunks'):
        S.last_chunks = []

def process_one_chunk():
    try:
        payload = S.q.get(timeout=0.001)  # {sr, channels, data:[T,C], t_emit}
    except Empty:
        return False

    t0 = time.perf_counter()
    chunk = np.asarray(payload.get("data"), dtype=np.float32)
    S.signal_quality = quick_signal_quality(chunk)
    probs = None
    label = None
    if S.decoding and _have_backend and os.path.isfile(S.model_pth):
        predictor = ensure_predictor(S.model_pth, sr=S.sr, channels=S.channels)
        probs, label = predictor.predict(chunk)  # returns (np.array, str)
    S.latency_ms = (time.perf_counter() - t0) * 1000.0

    if label:
        S.session_tokens.append(label)
        S.transcript = " ".join(S.session_tokens[-12:])

    ensure_history()
    S.last_chunks.append((list(chunk.shape), payload.get("t_emit", time.time())))
    if len(S.last_chunks) > 32:
        S.last_chunks = S.last_chunks[-32:]
    return True

did = False
if S.connected:
    for _ in range(2):
        did = process_one_chunk() or did

time.sleep(0.02)

