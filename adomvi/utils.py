import logging
import shutil
from pathlib import Path

import fiftyone as fo
import requests
from moviepy.editor import VideoFileClip

LOG = logging.getLogger(__name__)


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


def convert_video_avi_to_mp4(
    input_path: str | Path, output_path: str | Path, fps: int = 30
):
    """
    Convert a video to a specified frame rate using moviepy.

    Args:
        input_path (str | Path): The path to the input video file.
        output_path (str | Path): The path to the output video file.
        fps (int, optional): The frames per second for the output video. Defaults to 30.
    """
    clip = VideoFileClip(str(input_path)).set_fps(fps)
    clip.write_videofile(str(output_path), codec="libx264")


def cleanup_existing_dataset(name: str):
    """
    Delete existing dataset before the creation

    Args:
        name (str):  dataset name
    """
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
        dataset.delete()
        LOG.info(f"Dataset '{name}' deleted.")
    else:
        LOG.info(f"Dataset '{name}' does not exist.")


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


def convert_yolo_to_voc(image_shape: np.ndarray, bboxes: list):
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


def convert_voc_to_yolo(image_shape: np.ndarray, bboxes: list):
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
