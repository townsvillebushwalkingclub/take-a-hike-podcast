#!/usr/bin/env python3
"""One-off cleanup for podcasts_data.json: fix names and normalize punctuation."""

import json
import re
import sys
from pathlib import Path

from lib.config import JSON_FILE, TRANSCRIPTS_DIR
from lib.text import clean_text

TEXT_FIELDS = (
    "transcript",
    "blog_title",
    "blog_excerpt",
    "blog_content",
    "youtube_title",
    "youtube_description",
)

NAME_PATTERNS = (
    r"Lewyn\s+Warnakie",
    r"Lil\s+and\s+Warnecke",
    r"Lewin\s+Warnakie",
    r"Lewin\s+Warnocky",
    r"Lewin\s+Warnicky",
    r"Lil\s+and",
    r"\bLewyn\b",
    r"\bLewin\b",
    r"\bLoon\b",
    r"\bLohan\b",
    r"\bcherry\s+judge\b",
    r"\bCherry\s+judge\b",
)


def count_name_issues(text: str) -> int:
    if not text:
        return 0
    return sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in NAME_PATTERNS)


def main() -> int:
    if not JSON_FILE.exists():
        print(f"File not found: {JSON_FILE}")
        return 1

    data = json.loads(JSON_FILE.read_text(encoding="utf-8"))
    before_issues = sum(
        count_name_issues(episode.get(field, ""))
        for episode in data.values()
        for field in TEXT_FIELDS
    )

    changed_episodes = 0
    changed_fields = 0

    for episode_key, episode in data.items():
        episode_changed = False
        for field in TEXT_FIELDS:
            value = episode.get(field)
            if not isinstance(value, str) or not value:
                continue
            cleaned = clean_text(value)
            if cleaned != value:
                episode[field] = cleaned
                changed_fields += 1
                episode_changed = True
        if episode_changed:
            changed_episodes += 1
            print(f"Fixed: {episode_key}")

    after_issues = sum(
        count_name_issues(episode.get(field, ""))
        for episode in data.values()
        for field in TEXT_FIELDS
    )

    JSON_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    transcript_files_changed = 0
    if TRANSCRIPTS_DIR.exists():
        for transcript_path in TRANSCRIPTS_DIR.glob("*.txt"):
            original = transcript_path.read_text(encoding="utf-8")
            cleaned = clean_text(original)
            if cleaned != original:
                transcript_path.write_text(cleaned, encoding="utf-8")
                transcript_files_changed += 1
                print(f"Fixed transcript: {transcript_path.name}")

    print(f"\nEpisodes updated: {changed_episodes}")
    print(f"Fields updated: {changed_fields}")
    print(f"Transcript files updated: {transcript_files_changed}")
    print(f"Name issues before: {before_issues}")
    print(f"Name issues after: {after_issues}")
    print(f"Saved: {JSON_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
