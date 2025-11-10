# NeuroAlpha - Imagined Speech Decoder

> v2 branch - contains the live demo showcased at demo day. Device mode is wired through `Utilities/`, while the Streamlit frontend lives in `Neuro-Alpha-App/Frontend`.

## Getting Started

1. **Clone & checkout the demo branch**
   ```bash
   git clone https://github.com/<org-or-user>/Neural-Speech-Decoding.git
   cd Neural-Speech-Decoding
   git switch v2
   ```

2. **Create / activate the environment (conda or venv)**
   ```bash
   conda create -n nsd python=3.11.14 -y
   conda activate nsd
   pip install -r requirements.txt
   ```
   > **macOS note:** BrainFlow + MindsAI dependencies can be finicky on Apple hardware. If you hit driver issues, reinstall the BrainFlow runtime from their release page and re-run `pip install -r requirements.txt` in the same shell you use for Streamlit.

3. **Verify backend dependencies**  
   From `Neuro-Alpha-App/`, run:
   ```bash
   python -c "from Utilities.tester import run_trials; print('ok')"
   ```
   If you see `ok`, the BrainFlow + MindsAI stack is available. If not, make sure you activated the same env you used for `pip install`.

## Running the app

1. Connect the Neuropawn headset (USB serial) and note the port (e.g., `/dev/cu.usbserial-FTB6SPL3` on macOS or `COM7` on Windows). Use `ls /dev/cu.*` or Windows Device Manager if you're unsure which port the headset enumerated on.
2. Start the frontend:
   ```bash
   cd Neuro-Alpha-App/Frontend
   streamlit run app.py
   ```
3. In the sidebar:
   - Leave "Test mode" checked if you just want to see the mock experience.
   - Uncheck "Test mode" for the real demo, set the serial port, and press **Start** to begin recording. The backend averages 10 trials and displays the aggregated probabilities and normalized 8-channel EEG snapshot once **Stop** is pressed.

## Inspiration

Every year, millions of people lose their ability to speak due to ALS, Parkinson's, or other neurodegenerative diseases. These individuals remain fully conscious. Aware of everything around them yet unable to communicate even the simplest needs like "yes", "no", "water", "food", or even "help". We wanted to give them part of their voice back, to help them regain independence and connection using brain computer interface technology.

That's how NeuroAlpha was born, a system that can translate thoughts into words in real time.

## What it does

NeuroAlpha is a working prototype designed to decode imagined speech, the words a person is thinking, into text. This differs from attempted speech, where a person mentally tries to speak or move speech muscles without producing sound; imagined speech involves no articulatory movement at all. Using EEG brainwave data from the Neuropawn headset, our model detects neural activity patterns associated with specific words and predicts what the user is thinking in real time.

## How we built it

Limited datasets: While a few small imagined-speech EEG datasets exist, most are research-grade and collected using wet-electrode or high-density systems. Their formats and electrode placements weren't compatible with our affordable Neuropawn hardware, so we had to design and record our own dataset from scratch. We ran dozens of sessions where participants wore dry spike electrodes on the frontal lobe and imagined four key words - Yes, No, Water, and Food - plus background-noise segments, sampling at 125 Hz. During training we found the model confused Yes vs. No (likely due to overlapping cortical activation), so the production demo focuses on three classes: Food, Water, and Background Noise. With that simplification and an LSTM + residual stack, we hit roughly 70 % accuracy. The last piece was a Streamlit interface that pairs the decoder with live visualization so you can see brain activity and predictions in real time.

## Challenges we ran into

- No existing dataset: We had to design our own experiment and collect all training data manually.  
- Signal noise: EEG data is extremely sensitive. We had to reduce interference using bias fibers and electrode adjustments.  
- Hardware limitations: Dry electrodes need good scalp contact; hair density affected signal quality.  
- Subject variability: Each person's brain patterns differ, so individual calibration was necessary.

## Accomplishments we're proud of

- Built the first working imagined-speech-to-text model with affordable hardware.  
- Achieved ~70 % accuracy distinguishing between three classes (Food, Water, Background Noise).  
- Took the first step toward giving people with speech impairments a non-invasive way to communicate.

## What we learned

- EEG data is highly individual, meaning large-scale data will be key for a generalized decoder.  
- Words that activate the same parts of the brain (e.g., Yes vs. No) are hard to classify; the frontal placement was effective, but more electrodes or smarter encoders could let us reintroduce those words later.  
- And most importantly, we learned that it is possible to read imagined speech. This isn't science fiction anymore!

## Next steps

1. **Expand dataset & vocabulary** - Collect more EEG data across diverse users to improve generalization and enable decoding of a larger vocabulary.  
2. **Refine neural decoding models** - Experiment with transformer-based temporal models and attention-driven EEG encoders for higher accuracy.  
3. **Polish the real-time demo** - Harden the device pipeline, add calibration per user, and push toward edge deployment.

## Built with

`brainflow` · `numpy` · `pandas` · `python` · `pytorch` · `streamlit`

## Try it out

- Mock demo: keep Test mode on and click **Start** to watch the simulated EEG + probabilities update every few seconds.  
- Device demo: uncheck Test mode, set your Neuropawn serial port, press **Start**, and let the recorder average 10 trials to display a real EEG snapshot and predicted word.

Questions? Ping the team or open an issue. Let's keep giving people their voices back.

## Presentation
https://docs.google.com/presentation/d/1PmumbUgBneHLZSiKB_XiU8ok0VtoRsKsS3Wz_qaZzoA/edit?slide=id.p#slide=id.p
