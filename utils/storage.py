# utils/storage.py
import io, os, hashlib
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from config import FIG_STORE_PATH, FIG_PUBLIC_BASE_URL, THUMB_MAX_DIM
import datetime

try:
    import boto3  # optional
except Exception:
    boto3 = None

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def _timestamp_slug() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _thumb(image: Image.Image) -> Image.Image:
    img = image.copy()
    img.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM))
    return img

def _local_paths(base_hex: str) -> Tuple[str, str]:
    # spread files into two-level hex dirs to avoid giant folders
    root = Path(FIG_STORE_PATH) / base_hex[:2] / base_hex[2:4] / base_hex
    _ensure_dir(str(root))
    return str(root / "full.png"), str(root / "thumb.png")

def _public_url(local_path: str) -> str:
    if not FIG_PUBLIC_BASE_URL:
        # local file path fallback for dev
        return local_path
    # keep the path part under FIG_STORE_PATH
    rel = os.path.relpath(local_path, FIG_STORE_PATH).replace("\\", "/")
    return f"{FIG_PUBLIC_BASE_URL.rstrip('/')}/{rel}"

def save_image_bytes(img_bytes: bytes) -> dict:
    """
    Saves full-res PNG + thumbnail. Returns metadata:
    { 'sha256', 'full_path', 'thumb_path', 'full_url', 'thumb_url', 'width', 'height' }
    """
    h = _sha256_bytes(img_bytes)
    full_path, thumb_path = _local_paths(h)

    # Skip if files already exist (idempotent)
    if not os.path.exists(full_path):
        with Image.open(io.BytesIO(img_bytes)) as im:
            im.convert("RGB").save(full_path, format="PNG")
            t = _thumb(im)
            t.save(thumb_path, format="PNG")
            width, height = im.size
    else:
        with Image.open(full_path) as im:
            width, height = im.size

    return {
        "sha256": h,
        "full_path": full_path,
        "thumb_path": thumb_path,
        "full_url": _public_url(full_path),
        "thumb_url": _public_url(thumb_path),
        "width": width,
        "height": height,
    }

# Optional: S3 upload helper (use only if FIG_PUBLIC_BASE_URL points at your bucket/CDN)
def upload_to_s3(local_path: str, bucket: str, key: str, acl: str = "public-read") -> Optional[str]:
    if boto3 is None:
        return None
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ACL": acl, "ContentType": "image/png"})
    return f"s3://{bucket}/{key}"
