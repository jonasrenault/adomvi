import logging
import shutil
from pathlib import Path

LOG = logging.getLogger(__name__)


def restructure_dataset(roboflow_dir: Path):
    """
    Restructure dataset by copying jpg images to the 'data' directory
    and xml files to the 'labels' directory from the source directory.

    Args:
        roboflow_dir (Path): The source directory containing the dataset.
    """

    # Create target subdirectories if they don't exist
    (roboflow_dir / "data").mkdir(parents=True, exist_ok=True)
    (roboflow_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Define the source subdirectories
    source_subdirs = ["test", "train", "valid"]

    # Copy jpg and xml files from source to target
    for source_subdir in source_subdirs:
        source_path = roboflow_dir / source_subdir

        # Copy jpg files to 'data'
        for jpg_file in source_path.glob("*.jpg"):
            shutil.copy(jpg_file, roboflow_dir / "data")

        # Copy xml files to 'labels'
        for xml_file in source_path.glob("*.xml"):
            shutil.copy(xml_file, roboflow_dir / "labels")

        # Delete the original subdirectory
        shutil.rmtree(source_path)

    LOG.info("Dataset dir restructured successfully.")


def delete_images_without_labels(dataset):
    """
    Delete images without labels from dataset roboflow

    Args:
        dataset (_type_): The dataset from which to delete samples without labels
    """
    samples_to_delete = []
    for sample in dataset:
        if not sample.ground_truth.detections:
            samples_to_delete.append(sample.id)

    if samples_to_delete:
        dataset.delete_samples(samples_to_delete)
