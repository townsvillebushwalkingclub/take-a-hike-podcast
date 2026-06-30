#!/usr/bin/env python3
"""Strip legacy inline text from podcasts_data.json (transcripts live in files only)."""

import sys

from lib.config import JSON_FILE, cleaned_transcript_path_for_episode, raw_transcript_path_for_episode
from lib.state import get_episode, list_audio_episodes, load_state, save_state

LEGACY_FIELDS = (
    "transcript",
    "blog_title",
    "blog_excerpt",
    "blog_content",
)

CANONICAL_FIELDS = (
    "episode_file",
    "raw_transcript_file",
    "transcript_raw_done",
    "whisper_model",
    "transcript_file",
    "transcript_done",
    "blog_slug",
    "blog_file",
    "blog_url",
    "youtube_id",
    "youtube_url",
    "youtube_title",
    "youtube_description",
    "spotify_url",
    "spotify_title",
    "spotify_description",
    "sharing_image_file",
)


def default_episode(episode_filename: str) -> dict:
    return {
        "episode_file": episode_filename,
        "raw_transcript_file": "",
        "transcript_raw_done": False,
        "whisper_model": "",
        "transcript_file": "",
        "transcript_done": False,
        "blog_slug": "",
        "blog_file": "",
        "blog_url": "",
        "youtube_id": "",
        "youtube_url": "",
        "youtube_title": "",
        "youtube_description": "",
        "spotify_url": "",
        "spotify_title": "",
        "spotify_description": "",
        "sharing_image_file": "",
    }


def migrate_episode(episode_filename: str, episode: dict) -> bool:
    """Normalize one episode record; return True if changed."""
    defaults = default_episode(episode_filename)
    changed = False

    for field in LEGACY_FIELDS:
        if field in episode:
            del episode[field]
            changed = True

    for field in CANONICAL_FIELDS:
        if field not in episode:
            episode[field] = defaults[field]
            changed = True

    raw_path = raw_transcript_path_for_episode(episode_filename)
    if raw_path.exists():
        raw_ref = f"transcripts/raw/{raw_path.name}"
        if episode.get("raw_transcript_file") != raw_ref:
            episode["raw_transcript_file"] = raw_ref
            episode["transcript_raw_done"] = True
            changed = True

    cleaned_path = cleaned_transcript_path_for_episode(episode_filename)
    if cleaned_path.exists():
        cleaned_ref = f"transcripts/{cleaned_path.name}"
        if episode.get("transcript_file") != cleaned_ref:
            episode["transcript_file"] = cleaned_ref
            episode["transcript_done"] = True
            changed = True

    extra = set(episode) - set(CANONICAL_FIELDS)
    if extra:
        for field in extra:
            del episode[field]
        changed = True

    return changed


def main() -> int:
    if not JSON_FILE.exists():
        print(f"File not found: {JSON_FILE}")
        return 1

    state = load_state()
    episodes = list_audio_episodes() or list(state.keys())
    changed_episodes = 0

    for episode_filename in episodes:
        if episode_filename not in state:
            state[episode_filename] = get_episode(state, episode_filename)
            changed_episodes += 1
            print(f"Added: {episode_filename}")
            continue
        if migrate_episode(episode_filename, state[episode_filename]):
            changed_episodes += 1
            print(f"Migrated: {episode_filename}")

    orphan_keys = [key for key in state if key not in episodes and key.endswith(".mp3")]
    for key in orphan_keys:
        del state[key]
        changed_episodes += 1
        print(f"Removed orphan: {key}")

    save_state(state)
    print(f"\nMigrated episodes: {changed_episodes}")
    print(f"Saved: {JSON_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
