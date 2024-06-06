import logging
import requests
import shutil
from pathlib import Path
import fiftyone as fo

LOG = logging.getLogger()


def download_file(url: str, filename: Path):
    """
    Download a file from url to filename.

    Args:
        url (str): the url to download
        filename (Path): the destination file on disk
    """
    if filename.exists():
        LOG.info(f"File {filename} already exists. Skipping download.")
    else:
        LOG.info(f"Downloading {filename} from {url} ...")
        with requests.get(url, stream=True) as r:
            with open(filename, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        LOG.info("Download complete.")


def export_yolo_data(
    samples: fo.DatasetView,
    export_dir: Path,
    classes: list[str],
    label_field="ground_truth",
    split: list[str] | None = None,
):

    if type(split) == list:
        splits = split
        for split in splits:
            export_yolo_data(samples, export_dir, classes, label_field, split)
    else:
        if split is None:
            split_view = samples
            split = "val"
        else:
            split_view = samples.match_tags(split)

        split_view.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=split,
        )
