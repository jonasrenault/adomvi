import json
import logging
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from uuid import uuid4

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
URL_OPEN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding": "none",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}


class GoogleImageScraper:
    KEYWORDS = ["art", "model", "3D", "toy", "jouet", "jeu", "miniature", "maquette"]

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

        # First, do a regular search for the term, and refuse cookie policy popup.
        LOGGER.info(f"Seaching images for {self.search_term}")
        self.driver.get(f"https://www.google.com/search?q={self.search_term}")
        self._refuse_rgpd()

        # Click on images search button.
        LOGGER.info("Clicking Images search button")
        image_search = self.driver.find_element(By.LINK_TEXT, "Images")
        image_search.click()

        # Loop through all the thumbnails, stopping when we found enough images or
        # when results are exhausted.
        image_urls = {}
        visited_thumbnails = []
        new_results = True
        while len(image_urls) < self.max_images and new_results:
            # Fetch thumbnails
            LOGGER.info("Fetching thumbnails.")
            thumbnails = self.driver.find_elements(By.CSS_SELECTOR, "#islrg img.Q4LuWd")

            # Check that we have new results
            new_results = len(thumbnails) - len(visited_thumbnails) > 0
            LOGGER.info(
                f"Found {len(thumbnails)} thumbnails ({len(thumbnails) - len(visited_thumbnails)} new)."
            )

            # try to click on every new thumbnail to get the real image behind it
            for img in tqdm(thumbnails[len(visited_thumbnails) :]):
                try:
                    # Using EC.element_to_be_clickable will scroll down to the element
                    # (the element needs to be in the viewport to be clickable).
                    # This is important as scrolling down will load more results on the page.
                    WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable(img)
                    ).click()
                    time.sleep(0.5)
                except Exception as e:
                    LOGGER.warning(f"Exception clicking thumbnail {img}", exc_info=True)
                    continue

                # After clicking on a thumbnail, get the image url and title from the
                # side panel.
                url, title = self._extract_image_url()
                if (
                    url is not None
                    and self._filter_image(url, title)
                    and url not in self.downloaded_urls
                ):
                    image_urls[url] = title
                    LOGGER.debug(f"{len(image_urls)}\t{title}\t{url}")

                if len(image_urls) >= self.max_images:
                    break

            # Keep track of thumbnails already seen
            visited_thumbnails = thumbnails
        return image_urls

    def save_images(self, image_urls: dict[str, str]) -> None:
        LOGGER.info("Saving images to disk.")

        for url, title in tqdm(image_urls.items()):
            if url in self.downloaded_urls:
                LOGGER.info(f"Not downloading {url} as it already exists.")
                continue

            try:
                with Image.open(
                    urlopen(Request(url, headers=URL_OPEN_HEADERS))
                ) as image:
                    id = uuid4()
                    filename = f"{id}.{image.format}"
                    if image.size is None or (
                        self.min_resolution[0]
                        <= image.size[0]
                        <= self.max_resolution[0]
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
                            {
                                "id": str(id),
                                "url": url,
                                "title": title,
                                "size": image.size,
                            }
                        )
                    else:
                        LOGGER.debug(
                            f"Not saving image {url} because of invalid dimension ({image.size})"
                        )
            except Exception as e:
                LOGGER.warning(f"Exception saving image {url}", exc_info=True)
                continue

        LOGGER.info("Writing metadata file.")
        # Write metadata
        with open(self.metadata_file, "w") as f:
            for image in self.saved_files:
                f.write(json.dumps(image) + "\n")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    save_dir = Path("adomvi/google/AFV")
    scraper = GoogleImageScraper(
        save_dir,
        "leclerc tank",
        max_images=50,
        min_resolution=(400, 300),
        max_resolution=(2048, 2048),
    )
    images = scraper.get_image_urls()
    scraper.save_images(images)
