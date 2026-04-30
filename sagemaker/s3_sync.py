"""Background thread that periodically syncs a local directory to S3."""

import os
import threading
import time
from pathlib import Path


def _sync_to_s3(local_dir: str, s3_bucket: str, s3_prefix: str):
    import boto3
    s3 = boto3.client("s3")
    local_path = Path(local_dir)
    if not local_path.exists():
        return 0
    count = 0
    for f in local_path.rglob("*"):
        if not f.is_file():
            continue
        key = f"{s3_prefix}/{f.relative_to(local_path)}"
        try:
            s3.upload_file(str(f), s3_bucket, key)
            count += 1
        except Exception as e:
            print(f"[S3 sync] failed to upload {f}: {e}")
    return count


def sync_to_s3_once(local_dir: str, s3_uri: str) -> int:
    """Synchronize ``local_dir`` to ``s3_uri`` immediately."""
    if not s3_uri.startswith("s3://"):
        print(f"[S3 sync] invalid s3_uri: {s3_uri}, skipping")
        return 0

    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    count = _sync_to_s3(local_dir, bucket, prefix)
    print(f"[S3 sync] uploaded {count} files from {local_dir} to {s3_uri}")
    return count


def start_s3_sync(local_dir: str, s3_uri: str, interval_minutes: int = 30):
    """Start a daemon thread that syncs local_dir to s3_uri every interval_minutes."""
    if not s3_uri.startswith("s3://"):
        print(f"[S3 sync] invalid s3_uri: {s3_uri}, skipping")
        return

    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    def _loop():
        while True:
            time.sleep(interval_minutes * 60)
            try:
                sync_to_s3_once(local_dir, s3_uri)
            except Exception as e:
                print(f"[S3 sync] error: {e}")

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print(f"[S3 sync] started: {local_dir} -> {s3_uri} every {interval_minutes}min")
