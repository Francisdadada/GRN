from __future__ import annotations

import argparse

from grn.evaluate import evaluate_from_config
from grn.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GRN checkpoints and save overlays.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
