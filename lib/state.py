"""Load and save episode processing state in podcasts_data.json."""

import json
from typing import Any

from lib.config import JSON_FILE


def load_state() -> dict[str, Any]:
    """Load episode metadata from JSON, or return an empty dict."""
    if JSON_FILE.exists():
        with JSON_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_state(data: dict[str, Any]) -> None:
    """Persist episode metadata to JSON."""
    with JSON_FILE.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def get_episode(data: dict[str, Any], episode_filename: str) -> dict[str, Any]:
    """Return episode metadata, creating a default record if missing."""
    if episode_filename not in data:
        data[episode_filename] = {
            "episode_file": episode_filename,
            "transcript_file": "",
            "transcript_done": False,
            "blog_slug": "",
            "blog_file": "",
            "blog_url": "",
            "youtube_id": "",
            "youtube_url": "",
            "youtube_title": "",
        }
    return data[episode_filename]


def list_audio_episodes() -> list[str]:
    """Return sorted MP3 basenames from the audio directory."""
    from lib.config import AUDIO_DIR

    if not AUDIO_DIR.exists():
        return []
    return sorted(path.name for path in AUDIO_DIR.glob("*.mp3"))
