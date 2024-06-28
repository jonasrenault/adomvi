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
    Delete existing dataset before the création

    Args:
        name (str):  dataset name
    """
    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
        dataset.delete()
        LOG.info(f"Dataset '{name}' deleted.")
    else:
        LOG.info(f"Dataset '{name}' does not exist.")
