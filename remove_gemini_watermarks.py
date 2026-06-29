#!/usr/bin/env python3
"""Remove Gemini watermarks from podcast sharing images."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lib.config import BLOG_IMAGES_CLEAN_DIR, BLOG_IMAGES_DIR, PROJECT_ROOT

GWR_CLI = (
    PROJECT_ROOT
    / "node_modules"
    / "@pilio"
    / "gemini-watermark-remover"
    / "bin"
    / "gwr.mjs"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_gwr_cli() -> Path:
    """Return the local gwr CLI entrypoint, raising if npm deps are missing."""
    if not GWR_CLI.is_file():
        raise FileNotFoundError(
            "gemini-watermark-remover is not installed. "
            "Run: npm install"
        )
    return GWR_CLI


def iter_images(directory: Path) -> list[Path]:
    """Return supported image files in a directory, sorted by name."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def remove_watermark(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> subprocess.CompletedProcess[str]:
    """Run the gemini-watermark-remover CLI for one image."""
    cmd = [
        "node",
        str(resolve_gwr_cli()),
        "remove",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if overwrite:
        cmd.append("--overwrite")

    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def process_images(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> int:
    """Remove watermarks from all images in input_dir and write to output_dir."""
    images = iter_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(images)} image(s)")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    failures = 0
    skipped = 0

    for index, input_path in enumerate(images, start=1):
        output_path = output_dir / input_path.name

        if output_path.exists() and not overwrite:
            print(f"{index}/{len(images)} Skipping {input_path.name} - already exists")
            skipped += 1
            continue

        print(f"{index}/{len(images)} Processing {input_path.name}...")
        result = remove_watermark(input_path, output_path, overwrite=overwrite)
        if result.returncode != 0:
            failures += 1
            message = (result.stderr or result.stdout or "Unknown error").strip()
            print(f"  Error: {message}")
            continue

        print(f"  Saved: {output_path}")

    print()
    print(f"Complete: {len(images) - failures - skipped} saved, {skipped} skipped, {failures} failed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove Gemini watermarks from images using "
            "@pilio/gemini-watermark-remover"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BLOG_IMAGES_DIR,
        help=f"Directory of watermarked images (default: {BLOG_IMAGES_DIR.name}/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BLOG_IMAGES_CLEAN_DIR,
        help=f"Directory for cleaned images (default: {BLOG_IMAGES_CLEAN_DIR.name}/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cleaned images",
    )
    args = parser.parse_args()

    try:
        return process_images(
            args.input_dir.resolve(),
            args.output_dir.resolve(),
            overwrite=args.overwrite,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
