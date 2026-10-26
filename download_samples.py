"""Download freely licensed sample thali photos from Wikimedia Commons into data/samples/.

    python download_samples.py

These are for smoke-testing the pipeline only. Check each file's Commons page for its licence
before using it in a report or paper.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

TITLES = [
    "File:'5' A vegetarian thali, traditional style of serving a meal in India.jpg",
    "File:'3' A thali, traditional style of serving food in India, in a New York restaurant.jpg",
    "File:Boghi Thali.JPG",
    "File:Chitpavan Brahmin Thali.jpg",
    "File:Chutney's Deluxe Thali.jpg",
    "File:Egg Curry Thali.jpg",
    "File:Festival favourite thali.jpg",
    "File:Full Meal (Thali).jpg",
    "File:Gordhan Thal.jpg",
    "File:Kerala Style Lunch at Gundlupet.jpg",
    "File:Aloo Palak, Baigan, Raita & Chapatti (5105346232).jpg",
    "File:Kandhani Rajdhani.jpg",
]
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "thali-pipeline-samples/0.1 (student research project)"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()


def main(out_dir: str = "data/samples", width: int = 1024):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = {"action": "query", "titles": "|".join(TITLES), "prop": "imageinfo",
              "iiprop": "url", "iiurlwidth": width, "format": "json"}
    pages = json.loads(get(f"{API}?{urllib.parse.urlencode(params)}"))["query"]["pages"].values()

    for n, page in enumerate(sorted(pages, key=lambda p: p["title"]), 1):
        if "imageinfo" not in page:
            print(f"missing: {page['title']}")
            continue
        info = page["imageinfo"][0]
        dest = out / f"thali_{n:02d}.jpg"
        dest.write_bytes(get(info.get("thumburl") or info["url"]))
        print(f"{dest.name}  <-  {info['descriptionurl']}")


if __name__ == "__main__":
    main()
