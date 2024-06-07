import logging
import re
import shutil
import tarfile
from pathlib import Path

from adomvi.utils import download_file

LOG = logging.getLogger()


def download_class_names(download_dir: Path) -> dict[str, str]:
    """
    Download class ids and names for ImageNet.

    Args:
        download_dir (Path): directory where files will be downloaded

    Returns:
        dict[str, str]: dict of class id to class name
    """
    download_dir.mkdir(exist_ok=True)
    id_file = download_dir / "imagenet21k_wordnet_ids.txt"
    name_file = download_dir / "imagenet21k_wordnet_lemmas.txt"

    download_file(
        "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt", id_file
    )
    download_file(
        "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt",
        name_file,
    )

    with open(id_file, "r") as f:
        ids = f.readlines()

    with open(name_file, "r") as f:
        names = f.readlines()

    classes = {ids[i].strip(): names[i].strip() for i in range(len(ids))}
    return classes


def find_class_by_text(classes: dict[str, str], query: str) -> dict[str, str]:
    """
    Filter a dict of class id to class name by string query.

    Args:
        classes (dict[str, str]): dict of class id to class name
        query (str): a string query

    Returns:
        dict[str, str]: the filtered dict
    """
    filtered = {
        id: lemma
        for id, lemma in classes.items()
        if re.search(query, lemma, re.IGNORECASE)
    }
    return filtered


def download_annotations(class_ids: list[str], dataset_dir: Path) -> list[str]:
    """
    Download ImageNet annotations for given class ids, into dataset_dir.

    Args:
        class_ids (list[str]): the class ids
        dataset_dir (Path): the dataset directory

    Returns:
        list[str]: list of classes with annotations available
    """
    # Download zipfile with detections for all classes
    annotations_file = dataset_dir / "bboxes_annotations.tar.gz"
    annotations_dir = dataset_dir / "bboxes_annotations"
    download_file(
        "https://image-net.org/data/bboxes_annotations.tar.gz", annotations_file
    )

    # Extract annotations
    with tarfile.open(annotations_file, "r:gz") as tf:
        tf.extractall(annotations_dir)

    # Extract annotations for each class
    annoted_classes = []
    for class_id in class_ids:
        class_label_dir = dataset_dir / "labels" / class_id
        if class_label_dir.exists():
            LOG.info(
                f"Annotations directory {class_label_dir} already exists. Skipping extract."
            )
        else:
            annotations_class_file = annotations_dir / f"{class_id}.tar.gz"
            if annotations_class_file.exists():
                with tarfile.open(annotations_class_file, "r:gz") as tf:
                    tf.extractall(annotations_dir)
                shutil.move(annotations_dir / "Annotation" / class_id, class_label_dir)
                LOG.info(f"Extracted annotations for {class_id} to {class_label_dir}")
                annoted_classes.append(class_id)
            else:
                LOG.info(f"There are not annotations for class {class_id}.")

    # Delete annotations directory
    LOG.info("Deleting annotations dir.")
    shutil.rmtree(annotations_dir)
    return annoted_classes


def download_imagenet_detections(class_ids: list[str], dataset_dir: Path):
    """
    Download ImageNet images and annotations for given class ids into dataset_dir

    Args:
        class_ids (list[str]): class_ids to download
        dataset_dir (Path): the directory to save images into
    """
    # Create dataset_dir
    dataset_dir.mkdir(exist_ok=True)
    data_dir = dataset_dir / "data"
    data_dir.mkdir(exist_ok=True)
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    annoted_classes = download_annotations(class_ids, dataset_dir)

    # Download synset images for each class with annotations
    for class_id in annoted_classes:
        class_dir = data_dir / class_id
        if class_dir.exists():
            LOG.info(f"Directory {class_dir} already exists. Skipping download.")
        else:
            tarfilename = dataset_dir / f"{class_id}.tar"
            url = f"https://image-net.org/data/winter21_whole/{class_id}.tar"
            download_file(url, tarfilename)
            with tarfile.open(tarfilename) as tf:
                tf.extractall(class_dir)
            LOG.info(f"Extracted {class_dir}.")


def cleanup_labels_without_images(dataset_dir: Path):
    """
    Remove labels without images from dataset dir

    Args:
        dataset_dir (Path): the ImageNet dataset directory
    """
    data_dir = dataset_dir / "data"
    labels_dir = dataset_dir / "labels"
    classes = [path.name for path in data_dir.iterdir() if path.is_dir()]
    for class_id in classes:
        images = {
            path.stem for path in (data_dir / class_id).iterdir() if not path.is_dir()
        }
        labels = {
            path.stem for path in (labels_dir / class_id).iterdir() if not path.is_dir()
        }
        LOG.info(f"Deleting {len(labels.difference(images))} labels without images")
        for label_id in labels.difference(images):
            filename = labels_dir / class_id / (label_id + ".xml")
            filename.unlink()
