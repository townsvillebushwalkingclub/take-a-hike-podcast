"""Load raw and cleaned transcript files for an episode."""

from pathlib import Path

from lib.config import (
    cleaned_transcript_path_for_episode,
    raw_transcript_path_for_episode,
)


def read_raw_transcript(episode: dict, episode_filename: str) -> str:
    """Load raw Whisper transcript text (before name/term fixes)."""
    raw_file = episode.get("raw_transcript_file", "")
    if raw_file:
        path = Path(raw_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    fallback = raw_transcript_path_for_episode(episode_filename)
    if fallback.exists():
        return fallback.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        f"No raw transcript for {episode_filename}. Run transcribe.py first."
    )


def read_clean_transcript(episode: dict, episode_filename: str) -> str:
    """Load cleaned transcript text (after clean_transcripts.py)."""
    transcript_file = episode.get("transcript_file", "")
    if transcript_file:
        path = Path(transcript_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    fallback = cleaned_transcript_path_for_episode(episode_filename)
    if fallback.exists():
        return fallback.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        f"No cleaned transcript for {episode_filename}. Run clean_transcripts.py first."
    )
