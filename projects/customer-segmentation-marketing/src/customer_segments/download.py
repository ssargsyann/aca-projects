"""Fetch the explicitly versioned public source via Kaggle's library."""
from pathlib import Path
import shutil


def download(output_path):
    import kagglehub
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Destination already exists; existing local data is preserved.")
    downloaded = Path(kagglehub.dataset_download(
        "imakash3011/customer-personality-analysis/versions/1", path="marketing_campaign.csv",
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, output_path)
    return output_path
