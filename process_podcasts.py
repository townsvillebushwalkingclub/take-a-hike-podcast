#!/usr/bin/env python3
"""
DEPRECATED: Use the four-script pipeline instead.

Run these scripts in order:
  python transcribe.py
  python generate_blog.py
  python create_videos.py
  python upload_youtube.py

See README.md for setup and usage.
"""

import sys


def main() -> int:
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
