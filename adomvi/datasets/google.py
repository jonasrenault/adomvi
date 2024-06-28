import logging
import tarfile
from pathlib import Path

from adomvi.utils import download_file

LOG = logging.getLogger()


def download_google_dataset(url: str, google_dir: Path):
    """Download and unrar the Google dataset from the given URL.

    Args:
        url (str): the URL to download.
    """
    google_dir.mkdir(exist_ok=True)
    tarfilename = google_dir / "military-vehicles-dataset.tar.gz"

    download_file(url, tarfilename)
    with tarfile.open(tarfilename) as tf:
        tf.extractall(google_dir)
    tarfilename.unlink()
    LOG.info(f"Extracted to {google_dir}")
