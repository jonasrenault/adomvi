import logging
import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

LOG = logging.getLogger(__name__)


def load_labels(label_path: Path):
    """
    Load labels from a YOLO format txt file.

    Args:
        label_path (Path): Path to the label file.

    Returns:
        list: List of bounding boxes.
    """
    with open(label_path, "r") as file:
        labels = file.readlines()
    bboxes = []
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        bboxes.append([x_center, y_center, width, height, class_id])
    return bboxes


def yolo_to_voc(image_shape: np.ndarray, bboxes: list):
    """
    Convert labels from YOLO format to Pascal VOC format.

    Args:
        image_shape (tuple): Shape of the image in the format (height, width).
        bboxes (list): List of bounding boxes in YOLO format.

    Returns:
        list: List of bounding boxes in Pascal VOC format.
    """
    h, w = image_shape[:2]
    voc_bboxes = []
    for bbox in bboxes:
        x_center, y_center, width, height, class_id = bbox
        x_min = (x_center - width / 2) * w
        x_max = (x_center + width / 2) * w
        y_min = (y_center - height / 2) * h
        y_max = (y_center + height / 2) * h

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        voc_bboxes.append([x_min, y_min, x_max, y_max, class_id])

    return voc_bboxes


def voc_to_yolo(image_shape: np.ndarray, bboxes: list):
    """
    Convert labels from Pascal VOC format to YOLO format.

    Args:
        image_shape (tuple): Shape of the image in the format (height, width).
        bboxes (list): List of bounding boxes in Pascal VOC format.

    Returns:
        list: List of bounding boxes in YOLO format.
    """
    h, w = image_shape[:2]
    yolo_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, class_id = bbox
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h
        yolo_bboxes.append([class_id, x_center, y_center, width, height])
    return yolo_bboxes


def save_yolo_labels(label_path: Path, bboxes: list):
    """
    Save labels in YOLO format.

    Args:
        label_path (Path): Path to the label file.
        bboxes (list): List of bounding boxes in YOLO format.
    """
    with open(label_path, "w") as file:
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox

            # stop 6 digits after the decimal point
            file.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )


def apply_augmentation(image_path: Path, label_path: Path, augmentation_func):
    """
    Apply augmentation to an image and its labels.

    Args:
        image_path (Path): Path to the image file.
        label_path (Path): Path to the label file.
        augmentation_func (function): Augmentation function to apply.

    Returns:
        tuple: Augmented image and augmented bounding boxes in YOLO format.
    """
    image = cv2.imread(image_path)
    bboxes = load_labels(label_path)
    voc_bboxes = yolo_to_voc(image.shape, bboxes)

    aug_image, aug_bboxes = augmentation_func(image, voc_bboxes)
    aug_bboxes = voc_to_yolo(image.shape, aug_bboxes)

    # Save augmented image
    ext = image_path.suffix
    aug_image_path = str(image_path).replace(ext, f"_augmented{ext}")
    cv2.imwrite(aug_image_path, aug_image)

    # Save augmented labels
    aug_label_path = str(label_path).replace(".txt", f"_augmented.txt")
    save_yolo_labels(aug_label_path, aug_bboxes)

    return aug_image, aug_bboxes


def func_augmentation(image: np.ndarray, bboxes: list):
    """
    List of functions to apply on images and ther labels

    Args:
        image (np.ndarray): Image array.
        bboxes (list): List of bounding boxes in YOLO format.

    Returns:
        np.ndarray: image and ther labels
    """
    transform = A.Compose(
        [
            A.Affine(
                scale=(0.1, 0.9),
                translate_percent=(-0.3, 0.3),
                rotate=None,
                p=1.0,
            ),
            A.CoarseDropout(
                max_holes=10,
                max_height=200,
                max_width=200,
                min_holes=5,
                min_height=50,
                min_width=50,
                p=1.0,
            ),
            A.OneOf(
                [A.RandomSnow(p=1.0), A.RandomRain(p=1.0), A.RandomFog(p=1.0)], p=1.0
            ),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    augmented = transform(
        image=image, bboxes=bboxes, class_labels=["label"] * len(bboxes)
    )
    return augmented["image"], augmented["bboxes"]


def augment_dataset(dataset_path: Path, augmentation_func):
    """
    Augment the dataset by applying the augmentation function to each image and its corresponding labels.

    Args:
        dataset_path (Path): Path to the dataset.
        augmentation_func (callable): Augmentation function to apply to each image and its labels.
    """

    split_dir = ["test", "train"]
    for split_name in split_dir:
        image_path = dataset_path / f"images/{split_name}"
        label_path = dataset_path / f"labels/{split_name}"

        image_paths = sorted(image_path.glob("*"))
        label_paths = sorted(label_path.glob("*"))

        for image_path, label_path in zip(image_paths, label_paths):
            apply_augmentation(image_path, label_path, augmentation_func)

    LOG.info(f"Image and label saved to {dataset_path}")
