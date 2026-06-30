#!/usr/bin/env python3
"""
DEPRECATED: Use the pipeline scripts instead.

Run these scripts in order:
  python transcribe.py
  python clean_transcripts.py
  python generate_blog.py
  python upload_spotify.py
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
