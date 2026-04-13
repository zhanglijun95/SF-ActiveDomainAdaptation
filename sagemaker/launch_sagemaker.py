"""Launch a SageMaker training job for SFADA DAOD.

Usage:
    pip install sagemaker boto3        # one-time, on your cloud desktop
    python sagemaker/launch_sagemaker.py

Env vars you can override:
    SM_ROLE          – SageMaker execution role ARN  (see --help)
    SM_INSTANCE_TYPE – default ml.g6.12xlarge (2× L40S not available as a
                       native SM instance; g6.12xlarge = 4× L4 is closest.
                       See note below.)
"""

import argparse
import subprocess
import sys

import boto3
import sagemaker
from sagemaker.estimator import Estimator


# ---------------------------------------------------------------------------
# NOTE on L40S:
#   SageMaker does not yet offer a 2×L40S instance.  Closest options:
#     • ml.g6.12xlarge  – 4× NVIDIA L4  (each 24 GB)
#     • ml.g5.12xlarge  – 4× NVIDIA A10G (each 24 GB)
#     • ml.p4d.24xlarge – 8× A100 40 GB  (overkill but available)
#   If your team has reserved capacity or a custom instance pool with L40S,
#   set SM_INSTANCE_TYPE accordingly.
# ---------------------------------------------------------------------------

REPO_ROOT = "/home/ljzhang/code/SFADA"
ECR_REPO_NAME = "sfada-daod"
IMAGE_TAG = "latest"


def _get_account_and_region():
    sts = boto3.client("sts")
    account = sts.get_caller_identity()["Account"]
    region = boto3.session.Session().region_name or "us-west-2"
    return account, region


def _build_and_push_image(account: str, region: str) -> str:
    """Build Docker image and push to ECR. Returns the full image URI."""
    ecr_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"

    print(f"[1/3] Creating ECR repo (if needed): {ECR_REPO_NAME}")
    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.create_repository(repositoryName=ECR_REPO_NAME)
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass

    print(f"[2/3] Building Docker image …")
    subprocess.check_call(
        ["docker", "build", "-t", ecr_uri, "-f", "sagemaker/Dockerfile", "."],
        cwd=REPO_ROOT,
    )

    print(f"[3/3] Pushing to ECR: {ecr_uri}")
    login_cmd = subprocess.check_output(
        ["aws", "ecr", "get-login-password", "--region", region]
    ).decode().strip()
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin",
         f"{account}.dkr.ecr.{region}.amazonaws.com"],
        input=login_cmd.encode(), check=True,
    )
    subprocess.check_call(["docker", "push", ecr_uri])
    return ecr_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role", default=None,
        help="SageMaker execution role ARN. If not set, reads SM_ROLE env var "
             "or tries sagemaker.get_execution_role().",
    )
    parser.add_argument(
        "--instance-type", default="ml.g5.12xlarge",
        help="SageMaker instance type (default: ml.g5.12xlarge, 4×A10G).",
    )
    parser.add_argument(
        "--config",
        default="configs/daod/round_cityscapes_to_foggy_cityscapes_dino.yaml",
        help="Config path relative to project root.",
    )
    parser.add_argument(
        "--s3-data",
        default="s3://lijun-domainadaptation-sagemaker/data/cityscapes/",
        help="S3 URI of the dataset.",
    )
    parser.add_argument(
        "--s3-output",
        default="s3://lijun-domainadaptation-sagemaker/sagemaker-output/",
        help="S3 URI for training output artifacts.",
    )
    parser.add_argument(
        "--s3-source-ckpt",
        default=None,
        help="S3 URI of the pre-trained source checkpoint directory.",
    )
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Docker build/push (reuse existing ECR image).")
    args = parser.parse_args()

    import os
    role = args.role or os.environ.get("SM_ROLE")
    if not role:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            sys.exit(
                "ERROR: Could not determine SageMaker role.\n"
                "Ask your team admin for the execution role ARN and pass it via:\n"
                "  --role arn:aws:iam::<ACCOUNT>:role/<ROLE_NAME>\n"
                "or set SM_ROLE env var."
            )

    account, region = _get_account_and_region()

    if args.skip_build:
        image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"
        print(f"Reusing existing image: {image_uri}")
    else:
        image_uri = _build_and_push_image(account, region)

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        output_path=args.s3_output,
        hyperparameters={
            "config": args.config,
            "s3_sync_uri": f"{args.s3_output.rstrip('/')}/intermediate/",
            "s3_sync_interval": "30",
        },
        environment={
            "NCCL_DEBUG": "INFO",
        },
        max_run=3600 * 24 * 7,
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region)),
    )

    print(f"\nLaunching SageMaker training job:")
    print(f"  image:    {image_uri}")
    print(f"  instance: {args.instance_type}")
    print(f"  data:     {args.s3_data}")
    print(f"  config:   {args.config}")
    print(f"  output:   {args.s3_output}")

    inputs = {"data": args.s3_data}
    if args.s3_source_ckpt:
        inputs["source_ckpt"] = args.s3_source_ckpt

    estimator.fit(inputs)


if __name__ == "__main__":
    main()
