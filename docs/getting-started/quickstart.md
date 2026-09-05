# Quickstart

```bash
python examples/create_test_model.py --output /tmp/model.safetensors
python -m stream_runtime prepare /tmp/model.safetensors --output /tmp/prepared
python -m stream_runtime inspect /tmp/model.safetensors
python -m stream_runtime plan /tmp/prepared --ram-budget 1M
python -m stream_runtime run /tmp/prepared --ram-budget 1M --trace
```

Start a server with `python -m stream_runtime serve --model /tmp/prepared --ram-budget 1M --api-key local`.
