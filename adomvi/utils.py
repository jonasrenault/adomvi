import logging
import shutil
from pathlib import Path

import requests

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
