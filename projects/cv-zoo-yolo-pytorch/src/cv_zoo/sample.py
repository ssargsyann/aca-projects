"""Explicit retrieval of the notebook's public sample, pinned by content hash."""
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

SOURCE_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
EXPECTED_SHA256 = "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866"


def fetch_sample(output_path):
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Sample destination already exists; choose a new path.")
    with urlopen(SOURCE_URL, timeout=30) as response:
        content = response.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Downloaded sample differs from the recorded image checksum.")
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return {"source": SOURCE_URL, "sha256": EXPECTED_SHA256, "saved": str(output_path)}
