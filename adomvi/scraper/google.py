import logging
import sys
import time
import json
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
import requests
from PIL import Image
from uuid import uuid4

LOGGER = logging.getLogger(__name__)


class GoogleImageScraper:
    KEYWORDS = ["art", "model", "3D", "toy", "RC", "jouet", "jeu", "miniature"]

    def __init__(
        self,
        save_dir: Path,
        search_term: str = "t90",
        max_images: int = 10,
        headless: bool = True,
        min_resolution: tuple[int, int] = (640, 300),
        max_resolution: tuple[int, int] = (2048, 2048),
    ):
        self.search_term = search_term
        self.max_images = max_images
        self.headless = headless
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution

        options = Options()
        if headless:
            options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_window_size(1400, 1050)
        google_home = "https://www.google.com"
        self.driver.get(google_home)
        LOGGER.info(
            f"Chrome web driver initialized. Page title for {google_home}: {self.driver.title}"
        )

        self.save_dir = save_dir / search_term
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Read metadata info if present
        self.metadata_file = self.save_dir / "metadata.jsonl"
        self.saved_files = []
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.saved_files = [json.loads(l) for l in f.readlines()]
        self.downloaded_urls = set(map(lambda x: x["url"], self.saved_files))

    def _refuse_rgpd(self):
        """
        Refuse cookie policy. Refuse button has id W0wltc.
        """
        try:
            WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "W0wltc"))
            ).click()
        except Exception as e:
            LOGGER.warning("Exception clicking on refuse cookie button", e)

    def _extract_image_url(self) -> tuple[str, str]:
        """
        Look for a valid image tag in the page.

        Returns
        -------
        tuple[str, str]
            the image's url and title
        """
        for class_name in ["n3VNCb", "iPVvYb", "r48jcc", "pT0Scc"]:
            for image in self.driver.find_elements(By.CLASS_NAME, class_name):
                url = image.get_attribute("src")
                if url and "http" in url and "encrypted" not in url:
                    return url, image.get_attribute("alt")
        return None, None

    def _filter_image(self, url: str, title: str) -> bool:
        """
        Filter image based on unwanted keywords

        Parameters
        ----------
        url : str
            the image's url
        title : str
            the image's title

        Returns
        -------
        bool
            False if image should be discarded, True otherwise
        """
        for kw in self.KEYWORDS:
            if kw.lower() in url.lower() or kw.lower() in title.lower():
                return False

        return True

    def get_image_urls(self) -> dict[str, str]:
        """
        Get image urls for a given search term

        Returns
        -------
        dict[str, str]
            a dict of url -> image title
        """
        LOGGER.info(f"Seaching images for {self.search_term}")
        # First, do a regular search for the term, and refuse cookie policy popup
        self.driver.get(f"https://www.google.com/search?q={self.search_term}")
        self._refuse_rgpd()

        # Click on images search button
        LOGGER.info("Clicking Images search button")
        image_search = self.driver.find_element(By.LINK_TEXT, "Images")
        image_search.click()

        image_urls = {}
        visited_thumbnails = set()
        while len(image_urls) < self.max_images:
            # Scroll to end of page to fetch more results
            LOGGER.info("Scrolling page.")
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            # Fetch thumbnails
            LOGGER.info("Fetching thumbnails.")
            thumbnails = set(self.driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd"))
            LOGGER.info(
                f"Found {len(thumbnails)} thumbnails ({len(thumbnails - visited_thumbnails)} new)."
            )
            # try to click every thumbnail to get the real image behind it
            for img in tqdm(thumbnails - visited_thumbnails):
                try:
                    WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable(img)
                    ).click()
                    # img.click()
                    time.sleep(3)
                except Exception as e:
                    LOGGER.debug(f"Exception clicking thumbnail {img}", exc_info=True)
                    continue

                url, title = self._extract_image_url()
                if (
                    url is not None
                    and self._filter_image(url, title)
                    and url not in self.downloaded_urls
                ):
                    image_urls[url] = title
                    LOGGER.info(f"{len(image_urls)}\t{title}\t{url}")

                if len(image_urls) >= self.max_images:
                    break

            # Keep track of thumbnails already seen
            visited_thumbnails |= thumbnails
        return image_urls

    def save_images(self, image_urls: dict[str, str]) -> None:
        LOGGER.info("Saving images to disk.")

        for url, title in tqdm(image_urls.items()):
            if url in self.downloaded_urls:
                LOGGER.info(f"Not downloading {url} as it already exists.")
                continue

            r = requests.get(url)
            if r.status_code != 200:
                LOGGER.warning(f"Unable to download image {url}.")
                continue

            with Image.open(requests.get(url, stream=True).raw) as image:
                id = uuid4()
                filename = f"{id}.{image.format}"
                if image.size is None or (
                    self.min_resolution[0] <= image.size[0] <= self.max_resolution[0]
                    and self.min_resolution[1]
                    <= image.size[1]
                    <= self.max_resolution[1]
                ):
                    try:
                        image.save(self.save_dir / filename)
                    except OSError:
                        image = image.convert("RGB")
                        image.save(self.save_dir / filename)

                    self.saved_files.append(
                        {"id": str(id), "url": url, "title": title, "size": image.size}
                    )
                else:
                    LOGGER.info(
                        f"Not saving image {url} because of invalid dimension ({image.size})"
                    )

        LOGGER.info("Writing metadata file.")
        # Write metadata
        with open(self.metadata_file, "w") as f:
            for image in self.saved_files:
                f.write(json.dumps(image) + "\n")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    save_dir = Path("adomvi/google")
    scraper = GoogleImageScraper(
        save_dir,
        "ebg vulcain",
        max_images=5,
        min_resolution=(640, 300),
        max_resolution=(2048, 2048),
    )
    images = scraper.get_image_urls()
    scraper.save_images(images)
