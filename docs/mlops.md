# MLOps Notes

This project is intentionally small, but it includes the same boundaries used in larger ML systems.

## Reproducibility

- YAML configs capture data paths, model settings, and training hyperparameters.
- Checkpoints are written to an artifact directory instead of the repository root.
- `seed_everything` centralizes random seed setup.

## Experiment Tracking

A next production step is adding MLflow:

- log every config file,
- log training and validation losses,
- log per-class metrics,
- log sample overlays,
- register the best model checkpoint.

The optional `mlops` dependency group includes `mlflow`.

## Containerization

The Dockerfile packages the API service. Training can also run inside Docker with mounted data and GPU access.

## Cloud Path

The recommended AWS path for this project is:

1. Store images, masks, and model artifacts in S3.
2. Run training as a SageMaker training job.
3. Push the inference Docker image to ECR.
4. Deploy inference through SageMaker Endpoint or ECS.
5. Monitor request latency, error rate, and prediction drift.

## Kubernetes Path

The Kubernetes manifests in `deployment/k8s` deploy the API as a simple service. In a real system, model weights would be mounted from object storage or a model registry.
