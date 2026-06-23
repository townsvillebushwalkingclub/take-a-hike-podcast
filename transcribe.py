#!/usr/bin/env python3
"""Step 1: Transcribe podcast audio to raw plain text using openai-whisper."""

import argparse
import sys

import whisper

from lib.config import AUDIO_DIR, WHISPER_MODEL, ensure_directories, raw_transcript_path_for_episode
from lib.state import get_episode, list_audio_episodes, load_state, save_state


def transcribe_file(model, audio_path) -> str:
    """Transcribe a single audio file and return plain text."""
    print(f"Transcribing {audio_path.name}...")
    result = model.transcribe(str(audio_path), language="en", verbose=False)
    return result.get("text", "").strip()


def process_episode(model, model_name: str, episode_filename: str, state: dict, force: bool = False) -> None:
    """Transcribe one episode to a raw transcript file if needed."""
    audio_path = AUDIO_DIR / episode_filename
    raw_path = raw_transcript_path_for_episode(episode_filename)
    episode = get_episode(state, episode_filename)

    if raw_path.exists() and not force:
        episode["raw_transcript_file"] = f"transcripts/raw/{raw_path.name}"
        episode["transcript_raw_done"] = True
        if not episode.get("whisper_model"):
            episode["whisper_model"] = model_name
        print(f"Skipping {episode_filename} - raw transcript already exists")
        return

    if not audio_path.exists():
        print(f"Skipping {episode_filename} - audio file not found")
        return

    transcript = transcribe_file(model, audio_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(transcript, encoding="utf-8")

    episode["raw_transcript_file"] = f"transcripts/raw/{raw_path.name}"
    episode["transcript_raw_done"] = True
    episode["whisper_model"] = model_name
    print(f"Saved raw transcript: {raw_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe Take A Hike podcast episodes (raw output, no text fixes)"
    )
    parser.add_argument(
        "--model",
        default=WHISPER_MODEL,
        help=f"Whisper model size (default: {WHISPER_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if raw transcript file already exists",
    )
    args = parser.parse_args()

    ensure_directories()
    episodes = list_audio_episodes()
    if not episodes:
        print(f"No MP3 files found in {AUDIO_DIR}/")
        return 0

    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model, device="cpu")

    state = load_state()
    print(f"Found {len(episodes)} podcast episodes\n")

    for index, episode_filename in enumerate(episodes, start=1):
        print(f"{index}/{len(episodes)} {episode_filename}")
        try:
            process_episode(model, args.model, episode_filename, state, force=args.force)
            save_state(state)
        except Exception as exc:
            print(f"Error transcribing {episode_filename}: {exc}")
            continue
        print()

    print("Raw transcription complete. Run clean_transcripts.py to apply name and term fixes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
