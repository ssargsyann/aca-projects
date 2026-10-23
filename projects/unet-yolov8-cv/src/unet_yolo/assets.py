"""Explicit official asset retrieval into a fresh local directory."""

import io
import zipfile
from pathlib import PurePosixPath
from urllib.request import urlopen

from .common import new_directory, sha256, write_json

SOURCES = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    "coco8.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
    "traffic_video.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
}
EXPECTED = {
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    "coco8.zip": "54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9",
    "traffic_video.mp4": "d31e0ebf194cc16e50eea784e7c3b0b1bb7976a4fb9b3fec26cb92a6704cca34",
}


def fetch(output):
    output = new_directory(output)
    records = []
    for filename, url in SOURCES.items():
        with urlopen(url, timeout=60) as response:
            content = response.read(50 * 1024 * 1024 + 1)
        if len(content) > 50 * 1024 * 1024:
            raise ValueError("Download exceeds the explicit asset size limit")
        path = output / filename
        path.write_bytes(content)
        if sha256(path) != EXPECTED[filename]:
            raise ValueError(
                f"Download does not match the verified checksum: {filename}"
            )
        if filename.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
                if sum(member.file_size for member in members) > 100 * 1024 * 1024:
                    raise ValueError("Archive exceeds uncompressed size limit")
                for member in members:
                    name = PurePosixPath(member.filename)
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or "\\" in member.filename
                        or name.parts[0] != "coco8"
                    ):
                        raise ValueError("Unexpected archive member path")
                archive.extractall(output)
        records.append(
            {
                "file": filename,
                "url": url,
                "sha256": sha256(path),
                "bytes": len(content),
            }
        )
    write_json(output / "download-manifest.json", records)
    return records
