"""Explicit pinned GPT-2 retrieval; validation is separate from model loading."""

import hashlib
from pathlib import Path

from .common import new_output, sha256, write_json

MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHT_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
FILES = {
    "config.json": ("10c66461e4c109db5a2196bff4bb59be30396ed8", 665),
    "merges.txt": ("226b0752cac7789c48f0cb3ec53eda48b7be36cc", 456318),
    "tokenizer.json": ("4b988bccc9dc5adacd403c00b4704976196548f8", 1355256),
    "tokenizer_config.json": ("be4d21d94f3b4687e5a54d84bf6ab46ed0f8defd", 26),
    "vocab.json": ("1f1d9aaca301414e7f6c9396df506798ff4eb9a6", 1042301),
    "README.md": ("a16a55fda99d2f2e7b69cce5cf93ff4ad3049930", 8092),
    "model.safetensors": (WEIGHT_SHA256, 548105171),
}


def verify_model(directory):
    directory = Path(directory)
    records = []
    for filename, (expected, size) in FILES.items():
        path = directory / filename
        if not path.is_file() or path.stat().st_size != size:
            raise FileNotFoundError(
                f"Missing or incomplete pinned asset: {filename}; run fetch-model explicitly"
            )
        digest = sha256(path)
        if filename == "model.safetensors":
            valid = digest == expected
        else:
            content = path.read_bytes()
            valid = (
                hashlib.sha1(f"blob {size}\0".encode() + content).hexdigest()
                == expected
            )
        if not valid:
            raise ValueError(f"Pinned model asset checksum mismatch: {filename}")
        records.append(
            {
                "file": filename,
                "bytes": size,
                "sha256": digest,
                "upstream_identity": expected,
                "identity_type": "sha256"
                if filename == "model.safetensors"
                else "git_blob_sha1",
            }
        )
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "files": records,
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
    }


def fetch_model(output):
    output = new_output(output)
    from huggingface_hub import hf_hub_download

    for filename in FILES:
        print(f"Downloading pinned {filename}...", flush=True)
        hf_hub_download(
            MODEL_ID, filename, revision=REVISION, local_dir=str(output), token=False
        )
    manifest = verify_model(output)
    write_json(output / "model-manifest.json", manifest)
    return manifest
