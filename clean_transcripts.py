#!/usr/bin/env python3
"""Step 2: Apply name and term fixes to raw transcripts, writing cleaned copies."""

import argparse
import sys

from lib.config import cleaned_transcript_path_for_episode, ensure_directories
from lib.state import get_episode, list_audio_episodes, load_state, save_state
from lib.text import clean_text
from lib.transcripts import read_raw_transcript


def process_episode(episode_filename: str, state: dict, force: bool = False) -> None:
    """Clean one raw transcript and write the fixed copy."""
    episode = get_episode(state, episode_filename)
    cleaned_path = cleaned_transcript_path_for_episode(episode_filename)

    if cleaned_path.exists() and not force:
        episode["transcript_file"] = f"transcripts/{cleaned_path.name}"
        episode["transcript_done"] = True
        print(f"Skipping {episode_filename} - cleaned transcript already exists")
        return

    raw_text = read_raw_transcript(episode, episode_filename)
    cleaned_text = clean_text(raw_text)
    cleaned_path.write_text(cleaned_text, encoding="utf-8")

    episode["transcript_file"] = f"transcripts/{cleaned_path.name}"
    episode["transcript_done"] = True
    print(f"Saved cleaned transcript: {cleaned_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply name and term fixes to raw Whisper transcripts"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-clean even if cleaned transcript file already exists",
    )
    args = parser.parse_args()

    ensure_directories()
    episodes = list_audio_episodes()
    if not episodes:
        print("No MP3 files found in audio/")
        return 0

    state = load_state()
    print(f"Found {len(episodes)} podcast episodes\n")

    changed = 0
    for index, episode_filename in enumerate(episodes, start=1):
        print(f"{index}/{len(episodes)} {episode_filename}")
        try:
            before = cleaned_transcript_path_for_episode(episode_filename).exists()
            process_episode(episode_filename, state, force=args.force)
            save_state(state)
            if args.force or not before:
                changed += 1
        except FileNotFoundError as exc:
            print(f"Skipping {episode_filename} - {exc}")
        except Exception as exc:
            print(f"Error cleaning {episode_filename}: {exc}")
            continue
        print()

    print(f"Cleaned {changed} transcript(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
