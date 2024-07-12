import logging
import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from adomvi.utils import (
    load_labels,
    convert_yolo_to_voc,
    convert_voc_to_yolo,
    save_yolo_labels,
)

LOG = logging.getLogger(__name__)


def func_augmentation(
    image: np.ndarray,
    bboxes: list,
    scale: tuple = (0.1, 0.9),
    translate_percent: tuple = (-0.3, 0.3),
    max_holes: int = 10,
    max_height: int = 200,
    max_width: int = 200,
    min_holes: int = 5,
    min_height: int = 50,
    min_width: int = 50,
    p_affine: float = 1.0,
    p_coarse_dropout: float = 1.0,
    p_weather: float = 1.0,
):
    """
    List of functions to apply on images and their labels

    Args:
        image (np.ndarray): Image array.
        bboxes (list): List of bounding boxes in YOLO format.
        scale (tuple): Scale range for affine transformation.
        translate_percent (tuple): Translation range for affine transformation.
        max_holes (int): Maximum number of holes for coarse dropout.
        max_height (int): Maximum height of holes for coarse dropout.
        max_width (int): Maximum width of holes for coarse dropout.
        min_holes (int): Minimum number of holes for coarse dropout.
        min_height (int): Minimum height of holes for coarse dropout.
        min_width (int): Minimum width of holes for coarse dropout.
        p_affine (float): Probability for affine transformation.
        p_coarse_dropout (float): Probability for coarse dropout.
        p_weather (float): Probability for weather transformations.
        p (float): Probability for overall transformations.

    Returns:
        np.ndarray: Augmented image and their labels.
    """
    transform = A.Compose(
        [
            A.Affine(
                scale=scale,
                translate_percent=translate_percent,
                rotate=None,
                p=p_affine,
            ),
            A.CoarseDropout(
                max_holes=max_holes,
                max_height=max_height,
                max_width=max_width,
                min_holes=min_holes,
                min_height=min_height,
                min_width=min_width,
                p=p_coarse_dropout,
            ),
            A.OneOf(
                [
                    A.RandomSnow(p=p_weather),
                    A.RandomRain(p=p_weather),
                    A.RandomFog(p=p_weather),
                ],
                p=p_weather,
            ),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    augmented = transform(
        image=image, bboxes=bboxes, class_labels=["label"] * len(bboxes)
    )
    return augmented["image"], augmented["bboxes"]


def apply_augmentation(image_path: Path, label_path: Path, **augmentation_params):
    """
    Apply augmentation to an image and its labels.

    Args:
        image_path (Path): Path to the image file.
        label_path (Path): Path to the label file.
        augmentation_params (dict): Parameters to pass to the augmentation function.

    Returns:
        tuple: Augmented image and augmented bounding boxes in YOLO format.
    """
    image = cv2.imread(image_path)
    bboxes = load_labels(label_path)
    voc_bboxes = convert_yolo_to_voc(image.shape, bboxes)

    aug_image, aug_bboxes = func_augmentation(image, voc_bboxes, **augmentation_params)
    aug_bboxes = convert_voc_to_yolo(image.shape, aug_bboxes)

    # Save augmented image
    ext = image_path.suffix
    aug_image_path = str(image_path).replace(ext, f"_augmented{ext}")
    cv2.imwrite(aug_image_path, aug_image)

    # Save augmented labels
    aug_label_path = str(label_path).replace(".txt", f"_augmented.txt")
    save_yolo_labels(aug_label_path, aug_bboxes)

    return aug_image, aug_bboxes


def augment_dataset(dataset_path: Path, **augmentation_params):
    """
    Augment the dataset by applying the augmentation function to each image and its corresponding labels.

    Args:
        dataset_path (Path): Path to the dataset.
        augmentation_params (dict): Parameters to pass to the augmentation function.
    """
    split_dir = ["test", "train"]
    for split_name in split_dir:
        image_path = dataset_path / f"images/{split_name}"
        label_path = dataset_path / f"labels/{split_name}"

        image_paths = sorted(image_path.glob("*"))
        label_paths = sorted(label_path.glob("*"))

        for image_path, label_path in zip(image_paths, label_paths):
            apply_augmentation(image_path, label_path)
    LOG.info(f"Images and labels augmented and saved into {dataset_path}")
