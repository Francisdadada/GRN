# GRN Semi-Supervised Medical Image Segmentation

This repository turns the original Generative Reinforcement Network research code into an MLE-style project: configurable training, reusable model modules, evaluation metrics, Dockerized inference, a FastAPI serving layer, tests, and deployment notes.

GRN jointly trains:

- a generator `G` that transforms input images,
- a segmentation model `S` that predicts anatomical classes,
- a PatchGAN discriminator `D` used during adversarial training.

The project is designed as a portfolio demo for medical image segmentation with limited labels.

## Engineering Highlights

- Config-driven training and evaluation with YAML files.
- Reusable Python package under `src/grn`.
- CLI entrypoints for training, evaluation, and single-image prediction.
- FastAPI inference service with `/health`, `/predict-mask`, and `/predict-overlay`.
- Dockerfile and Kubernetes manifests for containerized serving.
- AWS/SageMaker deployment notes for cloud training.
- Unit tests for metrics, model shape, and API health.
- Original research scripts are preserved for traceability.

## Repository Layout

```text
configs/                 YAML configs and API environment example
deployment/              AWS and Kubernetes deployment examples
docker/                  Docker assets
docs/                    Architecture, training, inference, MLOps notes
scripts/                 CLI entrypoints
src/grn/                 Reusable GRN package
tests/                   Lightweight tests for CI
jointtraining_*.py       Original research training script
joint_training_*.py      Original validation/inference script
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

For GPU training, install the PyTorch build that matches your CUDA version before installing the project dependencies.

## Train

```bash
python scripts/train.py --config configs/train_0.05.yaml
```

Checkpoints are saved to:

```text
artifacts/checkpoints/grn_0.05/
```

The config supports both `labeled_*` and `labelled_*` spellings because the current dataset folders use `labelled_img`, `labelled_msk`, and `unlabelled_img`.

## Evaluate

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

This computes IoU, Dice, ASD, and HD95 with confidence intervals and saves prediction/ground-truth overlays.

## Single-Image Prediction

```bash
python scripts/predict.py ^
  --image dataset/new_test_img/example.jpg ^
  --generator model_weights/0.05/0.05joingtraining_aug_noenhance/best_G.pth ^
  --segmenter model_weights/0.05/0.05joingtraining_aug_noenhance/best_S.pth ^
  --output outputs/example_overlay.png ^
  --overlay
```

## Serve API

```bash
set GRN_GENERATOR_WEIGHTS=model_weights/0.05/0.05joingtraining_aug_noenhance/best_G.pth
set GRN_SEGMENTER_WEIGHTS=model_weights/0.05/0.05joingtraining_aug_noenhance/best_S.pth
python scripts/serve_api.py
```

Then open:

```text
http://localhost:8000/docs
```

## Docker

```bash
docker build -f docker/Dockerfile -t grn-segmentation .
docker run -p 8000:8000 ^
  -e GRN_GENERATOR_WEIGHTS=/app/artifacts/checkpoints/best_G.pth ^
  -e GRN_SEGMENTER_WEIGHTS=/app/artifacts/checkpoints/best_S.pth ^
  -v %cd%/model_weights:/app/model_weights ^
  grn-segmentation
```

## Tests

```bash
pytest
ruff check .
```
