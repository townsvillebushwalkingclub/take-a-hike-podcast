#!/usr/bin/env python3
"""Generate Spotify metadata and upload podcast audio via Playwright."""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from lib.blog import read_blog_frontmatter, update_blog_spotify_url
from lib.config import AUDIO_DIR, GRAPHIC_FILE, ensure_directories
from lib.gemini_client import generate_json_sync
from lib.names import NAME_PROMPT_NOTE
from lib.spotify import publish_episode_to_spotify
from lib.state import get_episode, list_audio_episodes, load_state, save_state
from lib.text import TEXT_PROMPT_NOTE, clean_text
from lib.transcripts import read_clean_transcript


class SpotifyMetadata(BaseModel):
    title: str = Field(description="Engaging Spotify episode title under 120 characters")
    summary: str = Field(
        description="2-3 paragraph episode description under 3500 characters (no full transcript)"
    )


SPOTIFY_PROMPT = """You are a content creator for the "Take A Hike" podcast, a show about bushwalking and hiking adventures in and around Townsville, Australia. The podcast is hosted by Blair Woodcock with regular guests Luen Warneke and Cherry Judge.

Based on the following podcast transcript, create Spotify episode metadata.

Episode filename: {episode_filename}
Blog post URL: {blog_url}

Transcript:
{transcript}

Requirements:
- title: engaging episode title under 120 characters with hiking/bushwalking/Townsville keywords
- summary: 2-3 paragraph intro that hooks listeners and covers key topics (under 3500 characters)
- Do not include the full transcript in the summary
- {name_note}
- {text_note}

Respond with JSON only in this exact shape:
{{"title": "Example Title", "summary": "Intro paragraphs..."}}
"""


def build_description(summary: str, blog_url: str) -> str:
    """Assemble the Spotify episode description."""
    return f"{summary.strip()}\n\nRead the full blog post: {blog_url}"


def process_episode(episode_filename: str, state: dict, *, headless: bool = True, force: bool = False) -> None:
    """Upload one episode to Spotify if not already uploaded."""
    episode = get_episode(state, episode_filename)

    if episode.get("spotify_url") and not force:
        print(f"Skipping {episode_filename} - already on Spotify ({episode['spotify_url']})")
        return

    audio_path = AUDIO_DIR / episode_filename
    if not audio_path.exists():
        print(f"Skipping {episode_filename} - audio file not found")
        return

    blog_file = episode.get("blog_file", "")
    if not blog_file:
        print(f"Skipping {episode_filename} - no blog file. Run generate_blog.py first.")
        return

    blog_path = Path(blog_file)
    if not blog_path.exists():
        print(f"Skipping {episode_filename} - blog file missing: {blog_file}")
        return

    if not GRAPHIC_FILE.exists():
        raise FileNotFoundError(f"Episode art not found: {GRAPHIC_FILE}")

    frontmatter = read_blog_frontmatter(blog_path)
    blog_url = episode.get("blog_url") or frontmatter.get("blog_url", "")
    if not blog_url:
        raise ValueError(f"No blog_url for {episode_filename}")

    transcript = read_clean_transcript(episode, episode_filename)
    print(f"Generating Spotify metadata for {episode_filename}...")

    prompt = SPOTIFY_PROMPT.format(
        episode_filename=episode_filename,
        blog_url=blog_url,
        transcript=transcript,
        name_note=NAME_PROMPT_NOTE,
        text_note=TEXT_PROMPT_NOTE,
    )
    metadata = generate_json_sync(prompt, SpotifyMetadata)

    title = clean_text(metadata.title.strip())
    summary = clean_text(metadata.summary)
    if len(title) > 120:
        title = title[:117] + "..."

    description = clean_text(build_description(summary, blog_url))
    if len(description) > 4000:
        description = description[:3997] + "..."

    print(f"Uploading to Spotify: {episode_filename}...")
    spotify_url = publish_episode_to_spotify(
        audio_path,
        GRAPHIC_FILE,
        title=title,
        description=description,
        headless=headless,
    )

    episode["spotify_url"] = spotify_url
    episode["spotify_title"] = title
    update_blog_spotify_url(blog_path, spotify_url)
    print(f"Updated blog spotify_url: {blog_path}")
    print(f"Spotify episode: {spotify_url}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Upload podcast episodes to Spotify for Creators")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload even if spotify_url is already recorded",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show the browser window (useful for debugging cookie/session issues)",
    )
    args = parser.parse_args()

    ensure_directories()
    episodes = list_audio_episodes()
    if not episodes:
        print("No MP3 files found in audio/")
        return 0

    state = load_state()
    print(f"Found {len(episodes)} podcast episodes\n")

    for index, episode_filename in enumerate(episodes, start=1):
        print(f"{index}/{len(episodes)} {episode_filename}")
        try:
            process_episode(
                episode_filename,
                state,
                headless=not args.no_headless,
                force=args.force,
            )
            save_state(state)
        except Exception as exc:
            print(f"Error uploading to Spotify {episode_filename}: {exc}")
            continue
        print()

    print("Spotify upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
