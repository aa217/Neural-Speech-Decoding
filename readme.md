# Miniconda environment (Python 3.11.14) — quick setup

## 1) Create and activate the environment
```bash
# Create an environment named "nsd" with Python 3.11.14
conda create -n nsd python=3.11.14 -y

conda activate nsd

pip install -r requirements.txt