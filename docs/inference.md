# Inference

## Batch Evaluation

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

Evaluation computes:

- IoU
- Dice
- average surface distance
- HD95
- per-class confidence intervals

It also saves visual overlays for predicted and ground-truth masks.

## Single Image

```bash
python scripts/predict.py --image path/to/image.jpg --generator path/to/best_G.pth --segmenter path/to/best_S.pth --overlay
```

## API

```bash
python scripts/serve_api.py
```

Endpoints:

- `GET /health`
- `POST /predict-mask`
- `POST /predict-overlay`
