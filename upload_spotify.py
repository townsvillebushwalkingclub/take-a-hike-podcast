#!/usr/bin/env python3
"""Generate Spotify metadata and upload podcast audio via Playwright."""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from lib.blog import read_blog_frontmatter, update_blog_spotify_url
from lib.config import AUDIO_DIR, BLOG_IMAGES_CLEAN_DIR, PROJECT_ROOT, ensure_directories
from lib.gemini_client import generate_json_sync
from lib.names import NAME_PROMPT_NOTE
from lib.spotify import (
    find_published_episode_url,
    list_published_episodes,
    publish_episode_to_spotify,
)
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
- Do not include episode numbers in the title (no "Episode 5", "Ep. 12", etc.)
- Do not include the full transcript in the summary
- {name_note}
- {text_note}

Respond with JSON only in this exact shape:
{{"title": "Example Title", "summary": "Intro paragraphs..."}}
"""


def build_description(summary: str, blog_url: str) -> str:
    """Assemble the Spotify episode description."""
    return f"{summary.strip()}\n\nRead the full blog post: {blog_url}"


def resolve_spotify_episode_art(episode: dict) -> Path:
    """Return watermark-free episode art from images_clean, or the original sharing image."""
    sharing = episode.get("sharing_image_file", "").strip()
    if not sharing:
        slug = episode.get("blog_slug", "").strip()
        if slug:
            inferred = f"images/{slug}-sharing.jpg"
            if (PROJECT_ROOT / inferred).is_file():
                sharing = inferred
                episode["sharing_image_file"] = sharing
    if not sharing:
        raise FileNotFoundError(
            "No sharing_image_file in state. Run generate_blog_image.py first."
        )

    clean_path = BLOG_IMAGES_CLEAN_DIR / Path(sharing).name
    if clean_path.is_file():
        return clean_path

    original_path = PROJECT_ROOT / sharing
    if original_path.is_file():
        print(
            f"Warning: {clean_path.name} not found in images_clean/, "
            f"using watermarked image from {sharing}"
        )
        return original_path

    raise FileNotFoundError(
        f"Episode art not found: {clean_path} (run remove_gemini_watermarks.py)"
    )


def resolve_spotify_metadata(
    episode: dict,
    episode_filename: str,
    blog_url: str,
    transcript: str,
) -> tuple[str, str, bool]:
    """
    Return (title, description) for Spotify, reusing cached values when available.

    Returns:
        title, description, generated (True if Gemini was called this run)
    """
    cached_title = episode.get("spotify_title", "").strip()
    cached_description = episode.get("spotify_description", "").strip()
    if cached_title and cached_description:
        print(f"Using cached Spotify metadata for {episode_filename}")
        return cached_title, cached_description, False

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

    return title, description, True


def sync_spotify_urls(
    state: dict,
    *,
    headless: bool = True,
    episode_filenames: list[str] | None = None,
    published_episodes: list[dict] | None = None,
) -> int:
    """
    Match missing spotify_url values against the Spotify for Creators dashboard.

    Returns:
        Number of episodes backfilled.
    """
    targets = episode_filenames or list(state.keys())
    pending = [
        filename
        for filename in targets
        if filename in state and not state[filename].get("spotify_url")
    ]
    if not pending:
        return 0

    print(f"Syncing {len(pending)} episode(s) from Spotify dashboard...")
    published = published_episodes or list_published_episodes(headless=headless)
    print(f"Found {len(published)} published episode(s) on Spotify\n")

    updated = 0
    for episode_filename in pending:
        episode = state[episode_filename]
        title = episode.get("spotify_title", "").strip()
        if not title:
            continue

        spotify_url = find_published_episode_url(title, published)
        if not spotify_url:
            continue

        episode["spotify_url"] = spotify_url
        blog_file = episode.get("blog_file", "")
        if blog_file:
            blog_path = Path(blog_file)
            if not blog_path.is_file():
                blog_path = PROJECT_ROOT / "blogs" / Path(blog_file).name
            if blog_path.is_file():
                update_blog_spotify_url(blog_path, spotify_url)

        print(f"Linked {episode_filename}")
        print(f"  {title}")
        print(f"  {spotify_url}\n")
        updated += 1

    if updated:
        save_state(state)
    print(f"Synced {updated} Spotify URL(s).")
    return updated


def process_episode(
    episode_filename: str,
    state: dict,
    *,
    headless: bool = True,
    force: bool = False,
    published_episodes: list[dict] | None = None,
) -> None:
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

    episode_art = resolve_spotify_episode_art(episode)

    frontmatter = read_blog_frontmatter(blog_path)
    blog_url = episode.get("blog_url") or frontmatter.get("blog_url", "")
    if not blog_url:
        raise ValueError(f"No blog_url for {episode_filename}")

    transcript = read_clean_transcript(episode, episode_filename)
    title, description, _generated = resolve_spotify_metadata(
        episode, episode_filename, blog_url, transcript
    )

    episode["spotify_title"] = title
    episode["spotify_description"] = description
    save_state(state)

    if published_episodes is not None:
        existing_url = find_published_episode_url(title, published_episodes)
        if existing_url:
            episode["spotify_url"] = existing_url
            update_blog_spotify_url(blog_path, existing_url)
            save_state(state)
            print(f"Already on Spotify (matched by title): {existing_url}")
            return

    print(f"Uploading to Spotify: {episode_filename}...")
    print(f"  Episode art: {episode_art.name}")
    spotify_url = publish_episode_to_spotify(
        audio_path,
        episode_art,
        title=title,
        description=description,
        headless=headless,
    )

    episode["spotify_url"] = spotify_url
    update_blog_spotify_url(blog_path, spotify_url)
    save_state(state)
    print(f"Updated blog spotify_url: {blog_path}")
    print(f"Spotify episode: {spotify_url}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Upload podcast episodes to Spotify for Creators")
    parser.add_argument(
        "episode",
        nargs="?",
        help="Episode MP3 filename (default: process all episodes in audio/)",
    )
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
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only backfill spotify_url from the Spotify dashboard (no uploads)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip dashboard sync before uploading",
    )
    args = parser.parse_args()

    ensure_directories()
    BLOG_IMAGES_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if args.episode:
        episodes = [args.episode]
    else:
        episodes = list_audio_episodes()
    if not episodes:
        print("No MP3 files found in audio/")
        return 0

    state = load_state()
    headless = not args.no_headless

    if args.sync_only:
        if args.episode:
            sync_targets = [args.episode]
        else:
            sync_targets = list_audio_episodes()
        sync_spotify_urls(state, headless=headless, episode_filenames=sync_targets)
        return 0

    published_episodes = None
    if not args.no_sync:
        published_episodes = list_published_episodes(headless=headless)
        sync_spotify_urls(
            state,
            headless=headless,
            episode_filenames=episodes,
            published_episodes=published_episodes,
        )

    print(f"Processing {len(episodes)} podcast episode(s)\n")

    for episode_filename in episodes:
        print(episode_filename)
        try:
            process_episode(
                episode_filename,
                state,
                headless=headless,
                force=args.force,
                published_episodes=published_episodes,
            )
        except Exception as exc:
            print(f"Error uploading to Spotify {episode_filename}: {exc}")
            save_state(state)
            continue
        print()

    print("Spotify upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
