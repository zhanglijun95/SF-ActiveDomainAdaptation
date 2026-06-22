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
import hashlib
import re
import subprocess
import sys
import time
import random
from pathlib import Path

import boto3
import sagemaker
from botocore.exceptions import ClientError
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


def _job_part(value: str) -> str:
    part = re.sub(r"[^a-z0-9-]+", "-", str(value).lower().replace("_", "-")).strip("-")
    return part or "job"


def _job_slug(config_path: str) -> str:
    stem = Path(config_path).stem.lower()
    known_patterns = (
        "query_revival_multiview",
        "query_revival_scorer",
        "query_recovery_multiview",
        "query_recovery_scorer",
        "oracle_filter_recover",
        "oracle_classwise",
        "oracle_filter",
        "oracle_recover",
        "soft_query",
    )
    for pattern in known_patterns:
        if pattern in stem:
            return pattern.replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]+", "-", stem.replace("_", "-")).strip("-")
    return slug[:32].strip("-") or "job"


def _make_job_name(
    config_path: str,
    *,
    job_name_token: str = "",
    now_ns: int | None = None,
    rand: int | None = None,
) -> str:
    config_hash = hashlib.sha1(config_path.encode("utf-8")).hexdigest()[:8]
    token = _job_part(job_name_token)[:24] if job_name_token else config_hash
    now = time.time_ns() if now_ns is None else int(now_ns)
    rand_value = random.randint(0, 99999) if rand is None else int(rand)
    suffix = f"{token}-{config_hash}-{now}-{rand_value:05d}"
    job_prefix = f"sfada-{_job_slug(config_path)}"
    max_prefix_len = max(1, 63 - len(suffix) - 1)
    return f"{job_prefix[:max_prefix_len].rstrip('-')}-{suffix}"


def _is_create_training_job_throttle(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    code = str(error.get("Code", ""))
    message = str(error.get("Message", ""))
    throttle_codes = {
        "ThrottlingException",
        "TooManyRequestsException",
        "RequestLimitExceeded",
        "ThrottledException",
    }
    return code in throttle_codes or (
        "Rate exceeded" in message and "CreateTrainingJob" in str(exc)
    )


def _fit_with_create_retry(
    estimator: Estimator,
    inputs: dict[str, str],
    *,
    job_name: str,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
) -> None:
    """Retry SageMaker control-plane throttling while preserving blocking fit."""

    for attempt in range(1, int(max_attempts) + 1):
        try:
            estimator.fit(inputs, job_name=job_name)
            return
        except ClientError as exc:
            if attempt >= int(max_attempts) or not _is_create_training_job_throttle(exc):
                raise
            delay = min(float(max_delay), float(base_delay) * (2 ** (attempt - 1)))
            delay = random.uniform(delay * 0.5, delay * 1.5)
            print(
                "[sagemaker-launch][retry] "
                f"CreateTrainingJob throttled for {job_name}; "
                f"attempt={attempt}/{max_attempts}, sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


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
        "--s3-target-data",
        default=None,
        help=(
            "Optional second dataset channel for cross-dataset DAOD configs. "
            "For Cityscapes->BDD100K, pass the BDD100K S3 root here."
        ),
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
    parser.add_argument(
        "--s3-init-ckpt",
        default=None,
        help=(
            "Optional S3 URI for the detector initialization checkpoint used by "
            "source training, e.g. a detrex model-zoo .pth file or directory."
        ),
    )
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip Docker build/push (reuse existing ECR image).")
    parser.add_argument(
        "--job-name-token",
        default="",
        help="Optional unique token included in the SageMaker job name. "
             "Generated batch files pass the registry ID here.",
    )
    parser.add_argument(
        "--job-scoped-sync",
        action="store_true",
        help=(
            "Upload periodic intermediate files under intermediate/<job_name>/. "
            "By default, keep the historical layout under intermediate/baselines/."
        ),
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Run configs/daod/* with the DAOD oracle target-finetuning entrypoint.",
    )
    parser.add_argument(
        "--create-retries",
        type=int,
        default=12,
        help="Retry count for SageMaker CreateTrainingJob throttling.",
    )
    parser.add_argument(
        "--create-retry-base-delay",
        type=float,
        default=20.0,
        help="Initial retry delay in seconds for CreateTrainingJob throttling.",
    )
    parser.add_argument(
        "--create-retry-max-delay",
        type=float,
        default=300.0,
        help="Maximum retry delay in seconds for CreateTrainingJob throttling.",
    )
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

    job_name = _make_job_name(args.config, job_name_token=args.job_name_token)

    if args.job_scoped_sync:
        s3_sync_uri = f"{args.s3_output.rstrip('/')}/intermediate/{job_name}/"
    else:
        s3_sync_uri = f"{args.s3_output.rstrip('/')}/intermediate/"

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        output_path=args.s3_output,
        hyperparameters={
            "config": args.config,
            "s3_sync_uri": s3_sync_uri,
            "s3_sync_interval": "30",
            "oracle": "1" if args.oracle else "0",
        },
        environment={
            "NCCL_DEBUG": "INFO",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONUNBUFFERED": "1",
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
    if args.s3_target_data:
        inputs["target_data"] = args.s3_target_data
    if args.s3_source_ckpt:
        inputs["source_ckpt"] = args.s3_source_ckpt
    if args.s3_init_ckpt:
        inputs["init_ckpt"] = args.s3_init_ckpt

    _fit_with_create_retry(
        estimator,
        inputs,
        job_name=job_name,
        max_attempts=max(1, int(args.create_retries)),
        base_delay=max(1.0, float(args.create_retry_base_delay)),
        max_delay=max(1.0, float(args.create_retry_max_delay)),
    )


if __name__ == "__main__":
    main()
