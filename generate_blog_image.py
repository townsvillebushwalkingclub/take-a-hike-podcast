#!/usr/bin/env python3
"""Generate episode-specific podcast cover images via Gemini Nano Banana."""

import argparse
import sys
from pathlib import Path

from lib.blog import (
    blog_body_preview,
    format_episode_title,
    read_blog_frontmatter,
)
from lib.config import (
    BLOG_IMAGES_DIR,
    BLOGS_DIR,
    CLUB_LOGO_FILE,
    COVER_IMAGE_TEMPLATE,
    GEMINI_IMAGE_MODEL,
    ensure_directories,
)
from lib.gemini_client import generate_image_edit_sync, run_cover_image_batch_sync
from lib.state import get_episode, load_state, save_state

import re

def remove_listnr(text: str) -> str:
    """Remove the word 'LiSTNR' (case-insensitive) from the input text."""
    return re.sub(r"\bLiSTNR\b", "", text, flags=re.IGNORECASE)

IMAGE_PROMPT = """Design a striking podcast cover / social sharing image for this episode of "Take A Hike", published on the Townsville Bushwalking Club website.

You have strong creative freedom. Use the attached podcast cover template as inspiration—not a rigid frame. Feel free to reinterpret the scene, composition, colour palette, lighting, and mood. Make each episode feel distinct and evocative while still recognisable as the same show.

Brand anchors:
- "Take A Hike" text is our identity — this must stay the same as the reference image
- Incorporate the Townsville Bushwalking Club logo from the reference images

Creative direction:
- Let the episode topics below drive atmosphere: wildlife, waterfalls, rainforest, gorge country, islands, peaks, camping, gear, seasons, or local Townsville/Paluma adventures (North Queensland Wet Tropics Rainforest or outback bushland)
- North Queensland hiking aesthetic: tropical light, escarpments, reef-and-rainforest energy, adventure and warmth
- Bold visual storytelling is welcome — dramatic skies, depth, texture, and a cover someone would want to click
- Landscape orientation suitable for link previews

Episode context:
Title: {title}
Episode: {episode_title}
Excerpt: {excerpt}

Key topics from the blog:
{body_preview}"""

def resolve_blog_path(
    *,
    slug: str | None,
    episode_filename: str | None,
    state: dict,
) -> tuple[Path, str]:
    """Return the blog file path and episode filename for one episode."""
    if slug:
        blog_path = BLOGS_DIR / f"{slug}.md"
        if not blog_path.is_file():
            raise FileNotFoundError(f"Blog not found for slug: {slug}")

        frontmatter = read_blog_frontmatter(blog_path)
        episode = frontmatter.get("episode_file") or ""
        if not episode:
            for episode_filename_key, episode_data in state.items():
                if episode_data.get("blog_slug") == slug:
                    episode = episode_filename_key
                    break
        if not episode:
            raise ValueError(f"Could not resolve episode for slug: {slug}")
        return blog_path, episode

    if episode_filename:
        episode = get_episode(state, episode_filename)
        blog_file = episode.get("blog_file")
        if not blog_file:
            raise ValueError(f"No blog_file in state for episode: {episode_filename}")
        blog_path = Path(blog_file)
        if not blog_path.is_file():
            blog_path = BLOGS_DIR / Path(blog_file).name
        if not blog_path.is_file():
            raise FileNotFoundError(f"Blog not found for episode: {episode_filename}")
        return blog_path, episode_filename

    raise ValueError("Provide --slug or an episode filename")


def build_prompt(blog_path: Path, episode_filename: str) -> str:
    """Build the Nano Banana image edit prompt from blog metadata."""
    frontmatter = read_blog_frontmatter(blog_path)
    title = frontmatter.get("title", "").strip()
    excerpt = frontmatter.get("excerpt", "").strip()
    episode_title = format_episode_title(episode_filename)
    body_preview = blog_body_preview(blog_path)

    return IMAGE_PROMPT.format(
        title=remove_listnr(title),
        episode_title=remove_listnr(episode_title),
        excerpt=remove_listnr(excerpt),
        body_preview=remove_listnr(body_preview),
    )


def process_blog(
    blog_path: Path,
    episode_filename: str,
    state: dict,
    *,
    template: Path,
    club_logo: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    """Generate a cover image for one blog post."""
    frontmatter = read_blog_frontmatter(blog_path)
    slug = frontmatter.get("slug") or blog_path.stem
    output_path = output_dir / f"{slug}-sharing.jpg"
    episode = get_episode(state, episode_filename)

    if episode.get("sharing_image_file") and output_path.exists() and not force:
        print(f"Skipping {slug} - sharing image already exists")
        return

    if not template.is_file():
        raise FileNotFoundError(f"Cover template not found: {template}")
    if not club_logo.is_file():
        raise FileNotFoundError(f"Club logo not found: {club_logo}")

    prompt = build_prompt(blog_path, episode_filename)
    print(f"Generating cover image for {slug}...")
    print(f"  Model: {GEMINI_IMAGE_MODEL}")
    print(f"  Template: {template.name}")
    print(f"  Club logo: {club_logo.name}")
    print(f"  Episode: {format_episode_title(episode_filename)}")

    _, saved_path = generate_image_edit_sync(
        prompt,
        template_image=template,
        reference_images=[club_logo],
        output_dir=output_dir,
        filename=f"{slug}-sharing.jpg",
    )

    relative_path = f"images/{saved_path.name}"
    episode["sharing_image_file"] = relative_path
    print(f"Saved cover image: {saved_path}")


def iter_episodes_with_blogs(state: dict) -> list[tuple[Path, str]]:
    """Return blog paths and episode filenames for episodes that have blogs."""
    items: list[tuple[Path, str]] = []
    seen: set[str] = set()

    for episode_filename, episode in state.items():
        blog_file = episode.get("blog_file")
        if not blog_file:
            continue
        blog_path = Path(blog_file)
        if not blog_path.is_file():
            blog_path = BLOGS_DIR / Path(blog_file).name
        if not blog_path.is_file():
            continue
        key = str(blog_path)
        if key in seen:
            continue
        seen.add(key)
        items.append((blog_path, episode_filename))

    return sorted(items, key=lambda item: item[0].name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate episode-specific podcast cover images with Gemini Nano Banana"
    )
    parser.add_argument(
        "episode",
        nargs="?",
        help="Episode MP3 filename (optional if --slug or --all is used)",
    )
    parser.add_argument("--slug", help="Blog slug instead of episode filename")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate cover images for every episode that has a blog",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the cover image already exists",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=COVER_IMAGE_TEMPLATE,
        help=f"Podcast cover template (default: {COVER_IMAGE_TEMPLATE.name})",
    )
    parser.add_argument(
        "--club-logo",
        type=Path,
        default=CLUB_LOGO_FILE,
        help=f"Townsville Bushwalking Club logo (default: {CLUB_LOGO_FILE.name})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BLOG_IMAGES_DIR,
        help=f"Directory for generated images (default: {BLOG_IMAGES_DIR.name}/)",
    )
    args = parser.parse_args()

    if not args.all and not args.slug and not args.episode:
        parser.error("Provide an episode filename, --slug, or --all")

    ensure_directories()
    state = load_state()

    if args.all:
        items = iter_episodes_with_blogs(state)
        if not items:
            print("No blog posts found in podcasts_data.json")
            return 0

        print(f"Found {len(items)} blog posts")
        print("Using one shared Gemini chat for consistent cover styling\n")

        pending_jobs: list[tuple[str, str, str]] = []
        existing_covers: list[Path] = []

        for index, (blog_path, episode_filename) in enumerate(items, start=1):
            frontmatter = read_blog_frontmatter(blog_path)
            slug = frontmatter.get("slug") or blog_path.stem
            output_path = args.output_dir / f"{slug}-sharing.jpg"
            episode = get_episode(state, episode_filename)

            if episode.get("sharing_image_file") and output_path.exists() and not args.force:
                print(f"{index}/{len(items)} {blog_path.name}")
                print(f"Skipping {slug} - sharing image already exists")
                existing_covers.append(output_path)
                print()
                continue

            prompt = build_prompt(blog_path, episode_filename)
            pending_jobs.append((slug, prompt, episode_filename))
            print(f"{index}/{len(items)} {blog_path.name} (queued)")

        if not pending_jobs:
            print("\nNo cover images to generate.")
            return 0

        print(f"\nGenerating {len(pending_jobs)} cover(s) in one conversation...")
        if existing_covers:
            print(f"  Seeding chat with {len(existing_covers)} existing cover(s) for style reference")
        print(f"  Model: {GEMINI_IMAGE_MODEL}")
        print(f"  Template: {args.template.name}")
        print(f"  Club logo: {args.club_logo.name}\n")

        results = run_cover_image_batch_sync(
            [(slug, prompt) for slug, prompt, _ in pending_jobs],
            template_image=args.template,
            reference_images=[args.club_logo],
            output_dir=args.output_dir,
            existing_covers=existing_covers or None,
        )

        results_by_slug = {slug: (rel_path, saved_path, error) for slug, rel_path, saved_path, error in results}
        for slug, _prompt, episode_filename in pending_jobs:
            rel_path, saved_path, error = results_by_slug[slug]
            if error:
                print(f"Error generating image for {slug}: {error}")
                continue
            episode = get_episode(state, episode_filename)
            episode["sharing_image_file"] = rel_path
            print(f"Saved cover image: {saved_path}")

        save_state(state)
    else:
        blog_path, episode_filename = resolve_blog_path(
            slug=args.slug,
            episode_filename=args.episode,
            state=state,
        )
        process_blog(
            blog_path,
            episode_filename,
            state,
            template=args.template,
            club_logo=args.club_logo,
            output_dir=args.output_dir,
            force=args.force,
        )
        save_state(state)

    print("Cover image generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
