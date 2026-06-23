#!/usr/bin/env python3
"""Generate Ghost-compatible blog posts from transcripts using Gemini 2.5 Pro."""

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from lib.blog import write_blog_post
from lib.config import BLOGS_DIR, build_blog_url, ensure_directories
from lib.gemini_client import generate_json_sync
from lib.names import NAME_PROMPT_NOTE
from lib.state import get_episode, list_audio_episodes, load_state, save_state
from lib.text import TEXT_PROMPT_NOTE, clean_text
from lib.transcripts import read_clean_transcript


class BlogPost(BaseModel):
    slug: str = Field(description="URL slug: lowercase, hyphenated, include year when relevant")
    title: str = Field(description="Engaging SEO title under 60 characters")
    body: str = Field(description="Full blog post in Markdown with clear paragraphs")


BLOG_PROMPT = """You are a content creator for the "Take A Hike" podcast, a show about bushwalking and hiking adventures in and around Townsville, Australia. The podcast is hosted by Blair Woodcock with regular guests Luen Warneke and Cherry Judge, along with occasional special guests.

Based on the following podcast transcript, create a blog post for the Townsville Bushwalking Club Ghost CMS site.

Episode filename: {episode_filename}

Transcript:
{transcript}

Requirements:
- slug: lowercase, hyphenated, descriptive Ghost URL slug (include year when relevant, e.g. walkers-creek-2026)
- title: engaging SEO title under 60 characters
- body: full blog post in Markdown
  * Well-structured with clear paragraphs
  * Capture key points, stories, and advice from the episode
  * Conversational and informative tone for hiking enthusiasts
  * Include practical tips when relevant
- {name_note}
- {text_note}

Respond with JSON only in this exact shape:
{{"slug": "example-slug-2026", "title": "Example Title", "body": "Markdown content here"}}
"""


def process_episode(episode_filename: str, state: dict, force: bool = False) -> None:
    """Generate a blog post for one episode."""
    episode = get_episode(state, episode_filename)

    if episode.get("blog_slug") and episode.get("blog_file") and not force:
        blog_path = Path(episode["blog_file"])
        if blog_path.exists():
            print(f"Skipping {episode_filename} - blog already exists")
            return

    transcript = read_clean_transcript(episode, episode_filename)
    print(f"Generating blog post for {episode_filename}...")

    prompt = BLOG_PROMPT.format(
        episode_filename=episode_filename,
        transcript=transcript,
        name_note=NAME_PROMPT_NOTE,
        text_note=TEXT_PROMPT_NOTE,
    )
    blog_post = generate_json_sync(prompt, BlogPost)

    slug = blog_post.slug.strip().lower()
    title = clean_text(blog_post.title.strip())
    body = clean_text(blog_post.body)
    if len(title) > 60:
        title = title[:57] + "..."

    blog_url = build_blog_url(slug)
    blog_path = BLOGS_DIR / f"{slug}.md"

    write_blog_post(
        blog_path,
        slug=slug,
        title=title,
        body=body,
        episode_file=episode_filename,
        blog_url=blog_url,
    )

    episode["blog_slug"] = slug
    episode["blog_file"] = f"blogs/{blog_path.name}"
    episode["blog_url"] = blog_url
    print(f"Saved blog post: {blog_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate blog posts from podcast transcripts")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate blog posts even if they already exist",
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
            process_episode(episode_filename, state, force=args.force)
            save_state(state)
        except Exception as exc:
            print(f"Error generating blog for {episode_filename}: {exc}")
            continue
        print()

    print("Blog generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
