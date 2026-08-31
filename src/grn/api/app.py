from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from grn.inference.overlays import blend_overlay, colorize_mask
from grn.inference.predictor import GRNPredictor

app = FastAPI(
    title="GRN Segmentation API",
    description="FastAPI inference service for Generative Reinforcement Network segmentation.",
    version="0.1.0",
)

_predictor: GRNPredictor | None = None


def get_predictor() -> GRNPredictor:
    global _predictor
    if _predictor is not None:
        return _predictor

    generator_path = os.getenv("GRN_GENERATOR_WEIGHTS", "artifacts/checkpoints/best_G.pth")
    segmenter_path = os.getenv("GRN_SEGMENTER_WEIGHTS", "artifacts/checkpoints/best_S.pth")
    if not Path(generator_path).exists() or not Path(segmenter_path).exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model weights are not available. Set GRN_GENERATOR_WEIGHTS and "
                "GRN_SEGMENTER_WEIGHTS or place weights in artifacts/checkpoints."
            ),
        )

    _predictor = GRNPredictor(
        generator_weights=generator_path,
        segmenter_weights=segmenter_path,
        num_classes=int(os.getenv("GRN_NUM_CLASSES", "7")),
        device=os.getenv("GRN_DEVICE"),
    )
    return _predictor


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _read_image(file: UploadFile) -> Image.Image:
    content = await file.read()
    try:
        return Image.open(io.BytesIO(content)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc


@app.post("/predict-mask")
async def predict_mask(file: UploadFile = File(...)) -> Response:
    image = await _read_image(file)
    mask = get_predictor().predict_image(image)
    output = io.BytesIO()
    colorize_mask(mask).save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")


@app.post("/predict-overlay")
async def predict_overlay(file: UploadFile = File(...)) -> Response:
    image = await _read_image(file)
    mask = get_predictor().predict_image(image)
    overlay = blend_overlay(image.convert("RGBA").resize((mask.shape[1], mask.shape[0])), np.asarray(mask))
    output = io.BytesIO()
    overlay.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
