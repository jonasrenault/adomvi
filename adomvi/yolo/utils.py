from collections.abc import Iterable
from pathlib import Path
from moviepy.editor import VideoFileClip

import shutil
import fiftyone as fo
import numpy as np
import numpy.typing as npt
import IPython.display as display
import base64


def export_yolo_data(
    samples: fo.DatasetView,
    export_dir: Path,
    classes: list[str],
    label_field="ground_truth",
    split: list[str | None] | str | None = None,
    overwrite: bool = False,
):
    """
    Export a fiftyone DatasetView to a directory in Yolov5Dataset Format.

    Args:
        samples (fo.DatasetView): the dataset view to export
        export_dir (Path): the export directory
        classes (list[str]): the list of classes to export
        label_field (str, optional): the label field to export.
            Defaults to "ground_truth".
        split (list[str | None] | str | None, optional): the split to export.
            Defaults to None.
        overwrite(bool, optional): delete export_dir if exists. Defaults to False.
    """
    if export_dir.exists() and overwrite:
        shutil.rmtree(export_dir)

    if not isinstance(split, Iterable):
        split = [split]

    for s in split:
        if s is None:
            s = "val"
            split_view = samples
        else:
            split_view = samples.match_tags(s)

        split_view.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=classes,
            split=s,
        )


def _read_yolo_detections_file(predictions_file: Path) -> npt.NDArray[np.float_]:
    """
    Read a Yolo detection prediction file into a numpy array

    Args:
        predictions_file (Path): the predictions .txt file

    Returns:
        npt.NDArray[np.float_]: the predicted bounding boxes
    """
    detections = []
    if not predictions_file.exists():
        return np.array([])

    with open(predictions_file) as f:
        lines = [line.rstrip("\n").split(" ") for line in f]

    for line in lines:
        detection = [float(val) for val in line]
        detections.append(detection)
    return np.array(detections)


def _uncenter_boxes(boxes: npt.NDArray[np.float_]):
    """
    YOLOv8 represents bounding boxes in a centered format with coordinates
    [center_x, center_y, width, height], whereas FiftyOne stores bounding
    boxes in [top-left-x, top-left-y, width, height] format.

    This function converts from center coords to corner coords

    Args:
        boxes (npt.NDArray[np.float_]): the coordinates
    """
    boxes[:, 0] -= boxes[:, 2] / 2.0
    boxes[:, 1] -= boxes[:, 3] / 2.0


def _get_class_labels(
    predicted_classes: npt.NDArray[np.float_], class_list: list[str]
) -> list[str]:
    """
    Convert a list of class predictions (indices) to a list of class labels (strings)

    Args:
        predicted_classes (npt.NDArray[np.float_]): predicted class indices
        class_list (list[str]): class list

    Returns:
        list[str]: predicted class labels
    """
    labels = (predicted_classes).astype(int)
    labels = [class_list[label] for label in labels]
    return labels


def _convert_yolo_detections_to_fiftyone(
    yolo_detections: npt.NDArray[np.float_], class_list: list[str]
) -> fo.Detections:
    """
    Convert YoloV8 detections as numpy arrays to FiftyOne Detections object

    Args:
        yolo_detections (npt.NDArray[np.float_]): yolo detections
        class_list (list[str]): class list

    Returns:
        fo.Detections: Detections object
    """
    detections = []
    if yolo_detections.size == 0:
        return fo.Detections(detections=detections)

    boxes = yolo_detections[:, 1:-1]
    _uncenter_boxes(boxes)

    confs = yolo_detections[:, -1]
    labels = _get_class_labels(yolo_detections[:, 0], class_list)

    for label, conf, box in zip(labels, confs, boxes):
        detections.append(
            fo.Detection(label=label, bounding_box=box.tolist(), confidence=conf)
        )
    return fo.Detections(detections=detections)


def add_yolo_detections(
    test_view: fo.DatasetView,
    prediction_field: str,
    predictions_dir: Path,
    class_list: list[str],
):
    """
    Add detections predicted with a yolo model to a Fiftyone View

    Args:
        test_view (fo.DatasetView): the test view
        prediction_field (str): the prediction field to store detections in the test view
        predictions_dir (Path): the predictions directory
        class_list (list[str]): the class list
    """
    test_filepaths = test_view.values("filepath")
    predictions_files = [
        predictions_dir / Path(fp).with_suffix(".txt").name for fp in test_filepaths
    ]
    yolo_detections = [
        _read_yolo_detections_file(predictions_file)
        for predictions_file in predictions_files
    ]
    detections = [
        _convert_yolo_detections_to_fiftyone(yd, class_list) for yd in yolo_detections
    ]
    test_view.set_values(prediction_field, detections)


def convert_video_avi_to_mp4(
        input_path: Path,
        output_path: Path,
        fps: int = 30
    ):
    """
    Convert a video to a specified frame rate using moviepy.

    Args:
        input_path (str | Path): The path to the input video file.
        output_path (str | Path): The path to the output video file.
        fps (int, optional): The frames per second for the output video. Defaults to 30.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    clip = VideoFileClip(str(input_path))
    
    clip = clip.set_fps(fps)
    
    clip.write_videofile(str(output_path), codec='libx264')


def display_video(output_path: Path):
    """
    Display the video in a Jupyter notebook using HTML.

    Args:
        output_path (Path): The path to the output video file.
    Returns:
        IPython.display.HTML: An HTML object that display the video in the notebook.
    """
    # Convert the video file to a data URL
    video_path = output_path.resolve()
    video_url = f'data:video/mp4;base64,{base64.b64encode(video_path.read_bytes()).decode()}'

    # Display the video in the notebook
    return display.HTML(
        f"""
        <video controls width="640">
            <source src="{video_url}" type="video/mp4">
        </video>
        """
    )
