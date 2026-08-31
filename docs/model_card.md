# Model Card

## Model

Generative Reinforcement Network for 2D multi-class medical image segmentation.

## Intended Use

Research and portfolio demonstration for semi-supervised segmentation under limited labeled data.

## Not Intended For

Clinical diagnosis, treatment decisions, or production medical use without regulatory review, validation, monitoring, and data governance.

## Inputs

Single-channel 2D images resized to `256x256`.

## Outputs

Seven-class segmentation masks by default. Class definitions should be documented alongside the dataset release.

## Training Data

The local dataset is not included in public version control. Configs assume the dataset is mounted under `dataset/`.

## Evaluation

The evaluation command reports IoU, Dice, ASD, and HD95 per class with confidence intervals.

## Limitations

- Performance depends on the domain and labeling protocol of the local dataset.
- The current demo uses 2D slices rather than full 3D volumetric context.
- Deployment examples are templates and need environment-specific security, storage, and monitoring configuration.
