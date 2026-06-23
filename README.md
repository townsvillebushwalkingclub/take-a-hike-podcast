# Take A Hike Podcast

A processing pipeline for the "Take A Hike" podcast, a LiSTNR production with Blair Woodcock, Luen Warneke, and Cherry Judge. This project automates transcription, blog post generation, video creation, and YouTube upload for podcast episodes.

Find the audio files from the podcast: <https://drive.google.com/drive/folders/1g2efA-Rw0RiuZEYKuO2ItKbOy30V2nMH?usp=drive_link>

## Overview

The pipeline is split into four independent scripts:

1. **`transcribe.py`** — Transcribe audio to plain text using [openai-whisper](https://github.com/openai/whisper)
2. **`generate_blog.py`** — Generate Ghost-compatible blog posts using Gemini 2.5 Pro via [gemini-webapi](https://github.com/HanaokaYuzu/Gemini-API)
3. **`create_videos.py`** — Create portrait videos for social media
4. **`upload_youtube.py`** — Generate YouTube descriptions and upload videos publicly

Each script is resumable via `podcasts_data.json` and skips work that is already done.

## Project Structure

```text
take-a-hike-podcast/
├── audio/                  # Input podcast audio files (.mp3)
├── transcripts/            # Plain-text transcripts (.txt)
├── blogs/                  # Ghost-compatible blog posts (.md)
├── videos/                 # Generated portrait videos (.mp4)
├── lib/                    # Shared config, state, Gemini, blog, YouTube helpers
├── podcasts_data.json      # Processing metadata and YouTube tracking
├── transcribe.py
├── generate_blog.py
├── create_videos.py
├── upload_youtube.py
├── requirements.txt
└── TAH_Podcast_Graphics.jpg  # Podcast graphic for video generation
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

You also need FFmpeg installed:

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### 2. Configure Gemini (gemini-webapi)

1. Log in to [gemini.google.com](https://gemini.google.com)
2. Open browser dev tools → Network tab → refresh the page
3. Copy cookie values for `__Secure-1PSID` and `__Secure-1PSIDTS`
4. Create a `.env` file from `.env.template`:

```text
GEMINI_SECURE_1PSID=your-cookie-value
GEMINI_SECURE_1PSIDTS=your-cookie-value
GEMINI_COOKIE_PATH=./gemini_cache
GEMINI_WEBAPI_LOG_LEVEL=INFO
BLOG_BASE_URL=https://townsvillebushwalkingclub.com
WHISPER_MODEL=base
```

`GEMINI_COOKIE_PATH` can be any writable directory. On Windows, forward slashes are recommended.

### 3. Configure YouTube Upload (Optional)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable YouTube Data API v3
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download and save as `client_secret.json` in the project root
5. On first upload, the script opens a browser for authentication

## Usage

Run the scripts in this order:

```bash
python transcribe.py
python generate_blog.py
python create_videos.py
python upload_youtube.py
```

`create_videos.py` can run in parallel with transcription/blog generation since it only needs the audio files.

Each script supports `--force` to redo its step. `upload_youtube.py` also accepts `--youtube-credentials PATH`.

## Output Formats

### Transcripts

Plain text files in `transcripts/` named after the episode, e.g. `Take a Hike - Topic.txt`.

### Blog Posts

Markdown files with YAML frontmatter in `blogs/`, compatible with [Ghost CMS](https://ghost.org/):

```markdown
---
slug: walkers-creek-2026
title: "Walkers Creek - Trip Report"
youtube_url: "PLACEHOLDER_YOUTUBE_URL"
episode_file: "Take a Hike - Walkers Creek.mp3"
blog_url: "https://townsvillebushwalkingclub.com/walkers-creek-2026/"
---

Blog body in Markdown...
```

Posts use Ghost's root-level URL pattern (`/{slug}/`, not `/blog/{slug}/`). After YouTube upload, `upload_youtube.py` replaces `PLACEHOLDER_YOUTUBE_URL` with the real video link.

### YouTube Descriptions

Built programmatically in this structure:

```text
{AI summary/intro}

Read the full blog post: {blog_url}

---

Full Transcript:
{full transcript}

{hashtags}
```

Videos are uploaded as **public**.

### Name correction

Whisper and AI sometimes mishear **Luen Warneke** as "Lewyn Warnakie" and similar variants. All pipeline scripts automatically correct these to **Luen Warneke** in transcripts, blog posts, and YouTube descriptions. **Cherry Judge** is always normalized to that exact capitalization.

Typographic punctuation (curly apostrophes like `'`, smart quotes, em dashes, ellipsis characters) is also normalized to plain ASCII (`'`, `"`, `-`, `...`).

## JSON Data Structure

`podcasts_data.json` tracks progress per episode:

```json
{
  "Take a Hike - Topic.mp3": {
    "episode_file": "Take a Hike - Topic.mp3",
    "transcript_file": "transcripts/Take a Hike - Topic.txt",
    "transcript_done": true,
    "blog_slug": "topic-slug-2026",
    "blog_file": "blogs/topic-slug-2026.md",
    "blog_url": "https://townsvillebushwalkingclub.com/topic-slug-2026/",
    "youtube_id": "",
    "youtube_url": "",
    "youtube_title": ""
  }
}
```

## Troubleshooting

- **"FFmpeg not found"** → Install FFmpeg and ensure it is in your PATH
- **"Gemini cookies required"** → Set `GEMINI_SECURE_1PSID` and `GEMINI_SECURE_1PSIDTS` in `.env`
- **"YouTube credentials not found"** → Download OAuth credentials and save as `client_secret.json`
- **Video creation fails** → Verify `TAH_Podcast_Graphics.jpg` exists and audio files are valid MP3
- **Upload fails** → Check daily YouTube limits, verify YouTube Data API v3 is enabled

## Credits

- **Podcast:** Take A Hike — A LiSTNR production
- **Host:** Blair Woodcock
- **Regular Guests:** Luen Warneke, Cherry Judge
