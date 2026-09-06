import argparse
from typing import Optional
from .runtime.engine import RuntimeEngine
from .storage.tensor_store import TensorStore
from .graph.graph import ModelAnalyzer
import sys

def parse_budget(budget_str: str) -> int:
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if budget_str.isdigit():
        return int(budget_str)

    for unit, multiplier in units.items():
        if budget_str.endswith(unit):
            number = budget_str[:-len(unit)]
            return int(number) * multiplier
    return int(budget_str)

def main():
    parser = argparse.ArgumentParser(description="Stream Runtime CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("model_path", help="Path to prepared manifest")
    run_parser.add_argument("--ram-budget", type=str, default="512M", help="RAM budget (e.g. 512M, 1G)")

    args = parser.parse_args()

    if args.command == "run":
        budget = parse_budget(args.ram_budget)
        print(f"Initializing runtime with {args.ram_budget} budget...")
        engine = RuntimeEngine(args.model_path, budget)
        # Placeholder for real input data
        dummy_input = torch.randn(1, 512) # Assume in_dim=512
        output = engine.run(dummy_input)
        print("Inference successful.")

if __name__ == "__main__":
    main()
