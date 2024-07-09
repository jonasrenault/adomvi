import csv
import logging
from pathlib import Path

import cv2
import fiftyone as fo
import numpy as np

from adomvi.yolo.utils import _uncenter_boxes

LOG = logging.getLogger(__name__)


def load_search_2_dataset(dataset_dir: Path) -> fo.Dataset:
    """
    Load The Seach_2 dataset from disk into a fiftyone Dataset

    Args:
        dataset_dir (Path): dataset directory

    Returns:
        fo.Dataset: the dataset
    """
    # Read metadata
    with open(dataset_dir / "meta.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        metas = [row for row in reader]

    samples = []
    for meta in metas:
        imagefp = dataset_dir / f'images/IMG{meta["Image"].zfill(4)}.jpg'
        maskfp = dataset_dir / f'masks/mask{meta["Image"].zfill(2)}.jpg'
        mask = cv2.imread(maskfp, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape

        # Get normalized bounding box coordinates from X, Y, W, H
        bbox = np.array(
            [
                [
                    int(meta["X"]) / width,
                    int(meta["Y"]) / height,
                    int(meta["W"]) / width,
                    int(meta["H"]) / height,
                ]
            ]
        )
        # X and Y are the target's center, convert them to top left coordinates
        _uncenter_boxes(bbox)

        # Only keep mask values inside the bounding box
        x, y, w, h = bbox[0]
        x *= width
        y *= height
        w *= width
        h *= height
        mask = mask[int(y) : int(y + h), int(x) : int(x + w)]

        sample = fo.Sample(filepath=imagefp)
        sample["ground_truth"] = fo.Detections(
            detections=[
                fo.Detection(
                    label=meta["Target"],
                    bounding_box=bbox[0].tolist(),
                    mask=mask,
                )
            ]
        )
        sample["distance"] = meta["Distance"]
        samples.append(sample)

    dataset = fo.Dataset()
    dataset.add_samples(samples)
    return dataset
