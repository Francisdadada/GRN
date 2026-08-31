from __future__ import annotations

import argparse

from grn.training.trainer import train_from_config
from grn.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRN semi-supervised segmentation.")
    parser.add_argument("--config", default="configs/train_0.05.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_config(load_config(args.config))


if __name__ == "__main__":
    main()
