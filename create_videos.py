#!/usr/bin/env python3
"""Create portrait videos from podcast audio and the podcast graphic."""

import argparse
import sys
from pathlib import Path

from moviepy import AudioFileClip, CompositeVideoClip, ImageClip
from PIL import Image

from lib.config import (
    AUDIO_DIR,
    GRAPHIC_FILE,
    VIDEOS_DIR,
    ensure_directories,
    video_path_for_episode,
)
from lib.state import list_audio_episodes


def create_video(audio_path: Path, output_path: Path) -> None:
    """Create a portrait video from audio and the podcast graphic."""
    print(f"Creating video: {output_path.name}...")

    if not GRAPHIC_FILE.exists():
        raise FileNotFoundError(f"Graphic file not found: {GRAPHIC_FILE}")

    audio_clip = AudioFileClip(str(audio_path))
    duration = audio_clip.duration

    target_width = 1080
    target_height = 1920

    img = Image.open(GRAPHIC_FILE)
    img_width, img_height = img.size

    scale_w = target_width / img_width
    scale_h = target_height / img_height
    scale = max(scale_w, scale_h)

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    cropped_img = resized_img.crop((left, top, left + target_width, top + target_height))

    temp_image_path = Path("temp_portrait.jpg")
    cropped_img.save(temp_image_path, quality=95)

    video_clip = ImageClip(str(temp_image_path), duration=duration)
    video_clip = video_clip.resized(new_size=(target_width, target_height))

    try:
        final_clip = CompositeVideoClip([video_clip], duration=duration)
        final_clip = final_clip.with_audio(audio_clip)
    except (AttributeError, TypeError):
        try:
            final_clip = video_clip.with_audio(audio_clip)
        except AttributeError:
            final_clip = CompositeVideoClip([video_clip])
            final_clip = final_clip.with_audio(audio_clip)

    final_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        preset="medium",
        bitrate="5000k",
        threads=4,
        logger="bar",
    )

    audio_clip.close()
    video_clip.close()
    final_clip.close()

    if temp_image_path.exists():
        temp_image_path.unlink()

    print(f"Video created: {output_path}")


def process_episode(episode_filename: str, force: bool = False) -> None:
    """Create a video for one episode if needed."""
    audio_path = AUDIO_DIR / episode_filename
    video_path = video_path_for_episode(episode_filename)

    if video_path.exists() and not force:
        print(f"Skipping {episode_filename} - video already exists")
        return

    if not audio_path.exists():
        print(f"Skipping {episode_filename} - audio file not found")
        return

    create_video(audio_path, video_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create portrait videos for podcast episodes")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate videos even if they already exist",
    )
    args = parser.parse_args()

    ensure_directories()
    episodes = list_audio_episodes()
    if not episodes:
        print(f"No MP3 files found in {AUDIO_DIR}/")
        return 0

    print(f"Found {len(episodes)} podcast episodes\n")

    for index, episode_filename in enumerate(episodes, start=1):
        print(f"{index}/{len(episodes)} {episode_filename}")
        try:
            process_episode(episode_filename, force=args.force)
        except Exception as exc:
            print(f"Error creating video for {episode_filename}: {exc}")
            continue
        print()

    print(f"Video creation complete. Videos saved to {VIDEOS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
