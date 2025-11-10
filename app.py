import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from neuropawn_sim import get_live_eeg
from model import predict_word

st.set_page_config(layout="wide", page_title="NeuroAlpha Live EEG")

st.markdown("<h1 style='text-align:center;'>🧠 NeuroAlpha Brainwave Decoder</h1>", unsafe_allow_html=True)
chart_placeholder = st.empty()
prediction_placeholder = st.empty()

# Initialize buffer
fs = 256  # Hz
window = 256  # 1 second of data
signal = np.zeros(window)

# Start real-time update loop
while True:
    # Get simulated EEG sample
    new_sample = get_live_eeg()
    signal = np.roll(signal, -1)
    signal[-1] = new_sample

    # Plot EEG waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, lw=1)
    ax.set_ylim(-100, 100)
    ax.set_title("Live EEG Signal", fontsize=14)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude (µV)")
    chart_placeholder.pyplot(fig)
    plt.close(fig)

    # Predict word
    predicted_word = predict_word(signal)

    # Display word with dynamic color
    color_map = {
        "YES": "#2ecc71",
        "NO": "#e74c3c",
        "LEFT": "#3498db",
        "RIGHT": "#9b59b6",
        "UP": "#f1c40f",
        "DOWN": "#e67e22"
    }
    color = color_map.get(predicted_word, "#ffffff")

    prediction_placeholder.markdown(
        f"""
        <div style='text-align:center; background-color:{color}; padding:20px; border-radius:10px;'>
            <h2 style='color:white;'>🗣️ Predicted Word:</h2>
            <h1 style='color:white; font-size:50px;'>{predicted_word}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(0.1)
