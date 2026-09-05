# Installation

Use Python 3.9 or newer. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The project uses PyTorch, NumPy, safetensors, FastAPI, Uvicorn, and pytest. Runtime code does not install packages or download models. Install dependencies and copy model files while online, then inference can run offline.
