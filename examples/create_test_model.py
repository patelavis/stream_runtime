import torch
import os
from safetensors.torch import save_file

def create_test_model(output_path: str, num_layers: int = 5, hidden_dim: int = 1024):
    """Creates a large mock model in .safetensors format."""
    # Create tensors that collectively exceed a small memory budget (e.g., 1MB)
    model = {}
    for i in range(num_layers):
        # Weight shape: [Out, In]
        w_shape = (hidden_dim, hidden_dim)
        model[f"layer_{i}.weight"] = torch.randn(w_shape)
        model[f"layer_{i}.bias"] = torch.randn(hidden_dim)

    save_file(model, output_path)
    print(f"Created test model at {output_path} with {num_layers} layers.")

if __name__ == "__main__":
    # Create a 50MB model (approx. hidden_dim=4096 * num_layers=3)
    create_test_model("test_model.safetensors", num_layers=3, hidden_dim=4096)
