# Model Preparation

`prepare` reads safetensors metadata and uses the generic adapter to group matching `.weight` and `.bias` names into logical nodes. It does not call `AutoModel.from_pretrained` or `load_file`. The generic adapter is best effort and does not represent general Hugging Face support.
