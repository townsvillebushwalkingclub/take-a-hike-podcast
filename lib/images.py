"""Episode cover art resolution."""

from pathlib import Path

from lib.config import BLOG_IMAGES_CLEAN_DIR, PROJECT_ROOT


def resolve_episode_art(episode: dict, *, prefer_clean: bool = True) -> Path:
    """
    Return episode sharing image, preferring watermark-free images_clean/ when available.

    Updates episode["sharing_image_file"] when inferred from blog_slug.
    """
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
    if prefer_clean and clean_path.is_file():
        return clean_path

    original_path = PROJECT_ROOT / sharing
    if original_path.is_file():
        if prefer_clean and not clean_path.is_file():
            print(
                f"Warning: {clean_path.name} not found in images_clean/, "
                f"using watermarked image from {sharing}"
            )
        return original_path

    if prefer_clean:
        raise FileNotFoundError(
            f"Episode art not found: {clean_path} (run remove_gemini_watermarks.py)"
        )
    raise FileNotFoundError(f"Episode art not found: {original_path}")
