"""Blog post file helpers with YAML frontmatter."""

from pathlib import Path
from typing import Any

import yaml

from lib.config import PLACEHOLDER_YOUTUBE_URL


def write_blog_post(
    path: Path,
    *,
    slug: str,
    title: str,
    body: str,
    episode_file: str,
    blog_url: str,
    youtube_url: str = PLACEHOLDER_YOUTUBE_URL,
) -> None:
    """Write a Ghost-compatible Markdown file with YAML frontmatter."""
    frontmatter = {
        "slug": slug,
        "title": title,
        "youtube_url": youtube_url,
        "episode_file": episode_file,
        "blog_url": blog_url,
    }
    yaml_block = yaml.safe_dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    content = f"---\n{yaml_block}\n---\n\n{body.strip()}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_blog_frontmatter(path: Path) -> dict[str, Any]:
    """Read YAML frontmatter from a blog Markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}

    return yaml.safe_load(parts[1]) or {}


def update_blog_youtube_url(path: Path, youtube_url: str) -> None:
    """Replace the youtube_url in frontmatter after YouTube upload."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"Blog file missing frontmatter: {path}")

    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid blog frontmatter: {path}")

    frontmatter = yaml.safe_load(parts[1]) or {}
    frontmatter["youtube_url"] = youtube_url
    yaml_block = yaml.safe_dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    updated = f"---\n{yaml_block}\n---{parts[2]}"
    path.write_text(updated, encoding="utf-8")
