"""Project paths and environment configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = PROJECT_ROOT / "audio"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
RAW_TRANSCRIPTS_DIR = TRANSCRIPTS_DIR / "raw"
BLOGS_DIR = PROJECT_ROOT / "blogs"
VIDEOS_DIR = PROJECT_ROOT / "videos"
GRAPHIC_FILE = PROJECT_ROOT / "TAH_Podcast_Graphics.jpg"
JSON_FILE = PROJECT_ROOT / "podcasts_data.json"

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_TOKEN_FILE = PROJECT_ROOT / "youtube_token.json"
YOUTUBE_CREDENTIALS_FILE = PROJECT_ROOT / "client_secret.json"

GEMINI_MODEL = "gemini-2.5-pro"
PLACEHOLDER_YOUTUBE_URL = "PLACEHOLDER_YOUTUBE_URL"
PLACEHOLDER_SPOTIFY_URL = "PLACEHOLDER_SPOTIFY_URL"

BLOG_BASE_URL = os.getenv("BLOG_BASE_URL", "https://townsvillebushwalkingclub.com").rstrip("/")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
GEMINI_LOG_LEVEL = os.getenv("GEMINI_WEBAPI_LOG_LEVEL", "INFO")
SPOTIFY_PODCAST_ID = os.getenv("SPOTIFY_PODCAST_ID", "")
SPOTIFY_COOKIES_PATH = os.getenv("SPOTIFY_COOKIES_PATH", "spotify-cookies.json")


def build_blog_url(slug: str) -> str:
    """Build a Ghost CMS post URL at the site root."""
    return f"{BLOG_BASE_URL}/{slug.strip('/')}/"


def spotify_cookies_file() -> Path:
    """Return the path to Spotify session cookies JSON."""
    path = Path(SPOTIFY_COOKIES_PATH)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def raw_transcript_path_for_episode(episode_filename: str) -> Path:
    """Return the raw Whisper transcript path for an episode MP3 filename."""
    stem = Path(episode_filename).stem
    return RAW_TRANSCRIPTS_DIR / f"{stem}.txt"


def cleaned_transcript_path_for_episode(episode_filename: str) -> Path:
    """Return the cleaned transcript path for an episode MP3 filename."""
    stem = Path(episode_filename).stem
    return TRANSCRIPTS_DIR / f"{stem}.txt"


def transcript_path_for_episode(episode_filename: str) -> Path:
    """Alias for the cleaned transcript path (used by downstream scripts)."""
    return cleaned_transcript_path_for_episode(episode_filename)


def video_path_for_episode(episode_filename: str) -> Path:
    """Return the video .mp4 path for an episode MP3 filename."""
    stem = Path(episode_filename).stem
    return VIDEOS_DIR / f"{stem}.mp4"


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for directory in (RAW_TRANSCRIPTS_DIR, TRANSCRIPTS_DIR, BLOGS_DIR, VIDEOS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
