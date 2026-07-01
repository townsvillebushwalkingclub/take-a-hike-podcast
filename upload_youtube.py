#!/usr/bin/env python3
"""Generate YouTube descriptions and upload podcast videos."""

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from lib.blog import read_blog_frontmatter, update_blog_youtube_url
from lib.config import YOUTUBE_PLAYLIST_ID, ensure_directories, video_path_for_episode
from lib.gemini_client import generate_json_sync
from lib.names import NAME_PROMPT_NOTE
from lib.text import TEXT_PROMPT_NOTE, clean_text
from lib.state import get_episode, list_audio_episodes, load_state, save_state
from lib.transcripts import read_clean_transcript
from lib.youtube import add_video_to_playlist, extract_hashtags_from_description, upload_video


class YouTubeMetadata(BaseModel):
    title: str = Field(description="Engaging YouTube title under 100 characters")
    summary: str = Field(description="2-4 paragraph intro/hook for the video description")
    hashtags: str = Field(description="10-15 hashtags, space-separated, each starting with #")


YOUTUBE_PROMPT = """You are a content creator for the "Take A Hike" podcast, a show about bushwalking and hiking adventures in and around Townsville, Australia. The podcast is hosted by Blair Woodcock with regular guests Luen Warneke and Cherry Judge.

Based on the following podcast transcript, create YouTube metadata.

Episode filename: {episode_filename}
Blog post URL: {blog_url}

Transcript:
{transcript}

Requirements:
- title: engaging YouTube title under 100 characters with hiking/bushwalking/Townsville keywords
- summary: 2-4 paragraph intro that hooks viewers and covers key topics (do not paste the transcript)
- hashtags: 10-15 relevant hashtags, space-separated, each starting with #
- Do not include episode numbers in the title (no "Episode 5", "Ep. 12", etc.)
- Return valid JSON only: escape double quotes inside strings, and use \\n for paragraph breaks in summary (no literal line breaks inside JSON strings)
- {name_note}
- {text_note}

Respond with JSON only in this exact shape:
{{"title": "Example Title", "summary": "Paragraph one.\\n\\nParagraph two.", "hashtags": "#TakeAHike #Hiking ..."}}
"""


def build_description(summary: str, blog_url: str, hashtags: str) -> str:
    """Assemble the YouTube description (summary, blog link, and hashtags only)."""
    return (
        f"{summary.strip()}\n\n"
        f"Read the full blog post: {blog_url}\n\n"
        f"{hashtags.strip()}"
    )


def resolve_youtube_metadata(
    episode: dict,
    episode_filename: str,
    blog_url: str,
    transcript: str,
) -> tuple[str, str, str, bool]:
    """
    Return (title, description, hashtags) for YouTube, reusing cached values when available.

    Returns:
        title, description, hashtags, generated (True if Gemini was called this run)
    """
    cached_title = episode.get("youtube_title", "").strip()
    cached_description = episode.get("youtube_description", "").strip()
    if cached_title and cached_description:
        print(f"Using cached YouTube metadata for {episode_filename}")
        hashtags = extract_hashtags_from_description(cached_description)
        return cached_title, cached_description, hashtags, False

    print(f"Generating YouTube metadata for {episode_filename}...")

    prompt = YOUTUBE_PROMPT.format(
        episode_filename=episode_filename,
        blog_url=blog_url,
        transcript=transcript,
        name_note=NAME_PROMPT_NOTE,
        text_note=TEXT_PROMPT_NOTE,
    )
    metadata = generate_json_sync(prompt, YouTubeMetadata)

    title = clean_text(metadata.title.strip())
    summary = clean_text(metadata.summary)
    if len(title) > 100:
        title = title[:97] + "..."

    description = clean_text(build_description(summary, blog_url, metadata.hashtags))
    if len(description) > 5000:
        description = description[:4997] + "..."

    return title, description, metadata.hashtags, True


def process_episode(
    episode_filename: str,
    state: dict,
    *,
    credentials_path: Path | None = None,
    playlist_id: str = "",
    force: bool = False,
) -> None:
    """Upload one episode to YouTube if not already uploaded."""
    episode = get_episode(state, episode_filename)

    if episode.get("youtube_id") and not force:
        print(f"Skipping {episode_filename} - already uploaded ({episode['youtube_id']})")
        return

    video_path = video_path_for_episode(episode_filename)
    if not video_path.exists():
        print(f"Skipping {episode_filename} - video not found. Run create_videos.py first.")
        return

    blog_file = episode.get("blog_file", "")
    if not blog_file:
        print(f"Skipping {episode_filename} - no blog file. Run generate_blog.py first.")
        return

    blog_path = Path(blog_file)
    if not blog_path.exists():
        print(f"Skipping {episode_filename} - blog file missing: {blog_file}")
        return

    frontmatter = read_blog_frontmatter(blog_path)
    blog_url = episode.get("blog_url") or frontmatter.get("blog_url", "")
    if not blog_url:
        raise ValueError(f"No blog_url for {episode_filename}")

    transcript = read_clean_transcript(episode, episode_filename)
    title, description, hashtags, _generated = resolve_youtube_metadata(
        episode, episode_filename, blog_url, transcript
    )

    episode["youtube_title"] = title
    episode["youtube_description"] = description
    save_state(state)

    video_id = upload_video(
        video_path,
        title=title,
        description=description,
        hashtags=hashtags,
        credentials_path=credentials_path,
    )
    if not video_id:
        return

    if playlist_id:
        add_video_to_playlist(
            video_id,
            playlist_id,
            credentials_path=credentials_path,
        )

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    episode["youtube_id"] = video_id
    episode["youtube_url"] = youtube_url

    update_blog_youtube_url(blog_path, youtube_url)
    save_state(state)
    print(f"Updated blog youtube_url: {blog_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload podcast videos to YouTube")
    parser.add_argument(
        "episode",
        nargs="?",
        help="Episode MP3 filename (default: process all episodes in audio/)",
    )
    parser.add_argument(
        "--youtube-credentials",
        type=Path,
        default=None,
        help="Path to YouTube OAuth credentials JSON (default: client_secret.json)",
    )
    parser.add_argument(
        "--playlist-id",
        default=YOUTUBE_PLAYLIST_ID,
        help="YouTube playlist ID to add each upload to (default: YOUTUBE_PLAYLIST_ID env)",
    )
    parser.add_argument(
        "--no-playlist",
        action="store_true",
        help="Skip adding uploads to a playlist",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload even if youtube_id is already recorded",
    )
    args = parser.parse_args()

    ensure_directories()
    if args.episode:
        episodes = [args.episode]
    else:
        episodes = list_audio_episodes()
    if not episodes:
        print("No MP3 files found in audio/")
        return 0

    playlist_id = "" if args.no_playlist else args.playlist_id.strip()
    if playlist_id:
        print(f"YouTube playlist: {playlist_id}\n")

    state = load_state()
    print(f"Found {len(episodes)} podcast episodes\n")

    for index, episode_filename in enumerate(episodes, start=1):
        print(f"{index}/{len(episodes)} {episode_filename}")
        try:
            process_episode(
                episode_filename,
                state,
                credentials_path=args.youtube_credentials,
                playlist_id=playlist_id,
                force=args.force,
            )
        except KeyboardInterrupt:
            save_state(state)
            raise
        except Exception as exc:
            print(f"Error uploading {episode_filename}: {exc}")
            save_state(state)
            continue
        print()

    print("YouTube upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
