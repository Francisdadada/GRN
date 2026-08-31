from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload model artifacts to S3.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--s3-uri", required=True, help="Example: s3://bucket/prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    parsed = urlparse(args.s3_uri)
    if parsed.scheme != "s3":
        raise ValueError("--s3-uri must start with s3://")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    for file_path in artifact_dir.rglob("*"):
        if file_path.is_file():
            key = f"{prefix}/{file_path.relative_to(artifact_dir).as_posix()}".strip("/")
            s3.upload_file(str(file_path), bucket, key)
            print(f"Uploaded {file_path} to s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
