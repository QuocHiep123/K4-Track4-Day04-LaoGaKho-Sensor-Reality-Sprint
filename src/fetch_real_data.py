"""Download the small, attributed real-world image set used by the demo."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


SOURCES = [
    (
        "kitti_um_000032.png",
        "https://csundergrad.science.uoit.ca/courses/csci3240u/latest/labs/data/"
        "kitti-samples/image_2/um_000032.png",
    ),
    (
        "kitti_umm_000005.png",
        "https://csundergrad.science.uoit.ca/courses/csci3240u/latest/labs/data/"
        "kitti-samples/image_2/umm_000005.png",
    ),
    (
        "kitti_uu_000010.png",
        "https://csundergrad.science.uoit.ca/courses/csci3240u/latest/labs/data/"
        "kitti-samples/image_2/uu_000010.png",
    ),
    (
        "night_wikimedia_cc0.jpg",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
        "Night_view_of_road.jpg?width=1280",
    ),
]


def download_real_images(output_dir: Path, overwrite: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    headers = {"User-Agent": "camera-health-minilab/1.0 (educational benchmark)"}
    for filename, url in SOURCES:
        destination = output_dir / filename
        if destination.exists() and not overwrite:
            downloaded.append(destination)
            continue
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        if len(content) < 10_000:
            raise RuntimeError(f"Downloaded file is unexpectedly small: {url}")
        destination.write_bytes(content)
        downloaded.append(destination)
        print(f"Downloaded {filename} ({len(content) / 1024:.1f} KiB)")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download attributed real-world road images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = download_real_images(args.output_dir, args.overwrite)
    print(f"Real-world images ready: {len(paths)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
