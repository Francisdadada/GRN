# AWS Deployment Sketch

This folder documents the cloud path rather than forcing a large infrastructure dependency into the demo repo.

Recommended AWS services:

- S3 for datasets and checkpoints.
- SageMaker Training Jobs for GPU training.
- ECR for Docker images.
- SageMaker Endpoint or ECS for inference serving.
- CloudWatch for logs and service metrics.

Example local-to-S3 artifact upload:

```bash
python deployment/aws/upload_artifacts.py --artifact-dir artifacts/checkpoints/grn_0.05 --s3-uri s3://my-bucket/grn/checkpoints/grn_0.05
```

For an interview, the key point is the boundary: local training and cloud training use the same config, while only storage and execution environment change.
