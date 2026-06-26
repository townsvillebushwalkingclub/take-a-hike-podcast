#!/usr/bin/env python3
"""Generate episode-specific social sharing images via Gemini Nano Banana."""

import argparse
import sys
from pathlib import Path

from lib.blog import (
    blog_body_preview,
    format_episode_title,
    read_blog_frontmatter,
)
from lib.config import BLOG_IMAGES_DIR, BLOGS_DIR, GEMINI_IMAGE_MODEL, SHARING_IMAGE_TEMPLATE, ensure_directories
from lib.gemini_client import generate_image_edit_sync
from lib.state import get_episode, load_state, save_state

IMAGE_PROMPT = """Keep the overall layout and branding intact:
- LiSTNR logo at the top
- "TAKE A HIKE" hand-lettered title style and placement
- Hiker on rocky outlook over green valley with golden-hour light
- 1200x630 landscape aspect ratio suitable for link previews

Add subtle visual elements that reflect this specific episode/blog post. Integrate them into the landscape naturally—for example regional wildlife, waterfalls, rainforest, gorge country, camping gear, or track features mentioned below. Do not create a busy collage. Do not obscure the LiSTNR logo or main title.

Blog post:
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
        title=title,
        episode_title=episode_title,
        excerpt=excerpt,
        body_preview=body_preview,
    )


def process_blog(
    blog_path: Path,
    episode_filename: str,
    state: dict,
    *,
    template: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    """Generate a sharing image for one blog post."""
    frontmatter = read_blog_frontmatter(blog_path)
    slug = frontmatter.get("slug") or blog_path.stem
    output_path = output_dir / f"{slug}-sharing.jpg"
    episode = get_episode(state, episode_filename)

    if episode.get("sharing_image_file") and output_path.exists() and not force:
        print(f"Skipping {slug} - sharing image already exists")
        return

    if not template.is_file():
        raise FileNotFoundError(f"Template image not found: {template}")

    prompt = build_prompt(blog_path, episode_filename)
    print(f"Generating sharing image for {slug}...")
    print(f"  Model: {GEMINI_IMAGE_MODEL}")
    print(f"  Template: {template.name}")
    print(f"  Episode: {format_episode_title(episode_filename)}")

    _, saved_path = generate_image_edit_sync(
        prompt,
        template_image=template,
        output_dir=output_dir,
        filename=f"{slug}-sharing.jpg",
    )

    relative_path = f"images/{saved_path.name}"
    episode["sharing_image_file"] = relative_path
    print(f"Saved sharing image: {saved_path}")


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
        description="Generate episode-specific 1200x630 sharing images with Gemini Nano Banana"
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
        help="Generate sharing images for every episode that has a blog",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the sharing image already exists",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=SHARING_IMAGE_TEMPLATE,
        help=f"Base sharing image template (default: {SHARING_IMAGE_TEMPLATE.name})",
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

        print(f"Found {len(items)} blog posts\n")
        for index, (blog_path, episode_filename) in enumerate(items, start=1):
            print(f"{index}/{len(items)} {blog_path.name}")
            try:
                process_blog(
                    blog_path,
                    episode_filename,
                    state,
                    template=args.template,
                    output_dir=args.output_dir,
                    force=args.force,
                )
                save_state(state)
            except Exception as exc:
                print(f"Error generating image for {blog_path.name}: {exc}")
            print()
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
            output_dir=args.output_dir,
            force=args.force,
        )
        save_state(state)

    print("Sharing image generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
