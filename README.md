# Take A Hike Podcast

A processing pipeline for the "Take A Hike" podcast, a LiSTNR production with Blair Woodcock, Luen Warneke, and Cherry Judge. This project automates transcription, blog post generation, Spotify and YouTube publishing, and video creation for podcast episodes.

Find the audio files from the podcast: <https://drive.google.com/drive/folders/1g2efA-Rw0RiuZEYKuO2ItKbOy30V2nMH?usp=drive_link>

## Overview

The pipeline is split into six independent scripts:

1. **`transcribe.py`** — Transcribe audio to **raw** plain text using [openai-whisper](https://github.com/openai/whisper) (default: `large-v3`)
2. **`clean_transcripts.py`** — Apply name and place-name fixes to raw transcripts, writing cleaned copies
3. **`generate_blog.py`** — Generate Ghost-compatible blog posts using Gemini 2.5 Pro via [gemini-webapi](https://github.com/HanaokaYuzu/Gemini-API)
4. **`upload_spotify.py`** — Generate Spotify descriptions and upload audio via [Playwright](https://playwright.dev/python/) (Spotify for Creators has no upload API)
5. **`create_videos.py`** — Create portrait videos for social media
6. **`upload_youtube.py`** — Generate YouTube descriptions and upload videos publicly

Each script is resumable via `podcasts_data.json` and skips work that is already done. Transcript text is stored in files only — not duplicated in the JSON file.

## Project Structure

```text
take-a-hike-podcast/
├── audio/                  # Input podcast audio files (.mp3)
├── transcripts/
│   ├── raw/                # Raw Whisper output (step 1)
│   └── *.txt               # Cleaned transcripts after fixes (step 2)
├── blogs/                  # Ghost-compatible blog posts (.md)
├── videos/                 # Generated portrait videos (.mp4)
├── lib/                    # Shared config, state, Gemini, blog, YouTube, Spotify helpers
├── podcasts_data.json      # Processing metadata only (paths and flags)
├── transcribe.py
├── clean_transcripts.py
├── generate_blog.py
├── upload_spotify.py
├── create_videos.py
├── upload_youtube.py
├── migrate_podcasts_data.py
├── requirements.txt
├── spotify-cookies.json    # Session cookies (gitignored; export from browser)
└── TAH_Podcast_Graphics.jpg  # Podcast graphic (video + Spotify episode art)
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
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
WHISPER_MODEL=large-v3
```

`GEMINI_COOKIE_PATH` can be any writable directory. On Windows, forward slashes are recommended.

### 3. Configure Spotify Upload (Optional)

Spotify for Creators has no public upload API. This project uses Playwright with exported browser cookies.

1. Log in to [creators.spotify.com](https://creators.spotify.com)
2. Open your show and copy the show ID from the URL: `https://creators.spotify.com/pod/show/<SPOTIFY_PODCAST_ID>/…`
3. Export cookies to `spotify-cookies.json` in the project root (Cookie-Editor extension or browser DevTools while logged in)
4. Add to `.env`:

```text
SPOTIFY_PODCAST_ID=033EbCntVyEDfTa2Dz5EgH
```

Episode art uses `TAH_Podcast_Graphics.jpg` for every upload.

If uploads fail with cookie errors, re-export `spotify-cookies.json` or run with `--no-headless` to debug the browser session.

### 4. Configure YouTube Upload (Optional)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable YouTube Data API v3
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download and save as `client_secret.json` in the project root
5. On first upload, the script opens a browser for authentication

## Usage

Run the scripts in this order:

```bash
python transcribe.py
python clean_transcripts.py
python generate_blog.py
python upload_spotify.py
python create_videos.py
python upload_youtube.py
```

`create_videos.py` can run in parallel with earlier steps since it only needs the audio files.

Each script supports `--force` to redo its step. `upload_spotify.py` accepts `--no-headless` for debugging. `upload_youtube.py` accepts `--youtube-credentials PATH`.

`transcribe.py` writes untouched Whisper output to `transcripts/raw/`. Re-run with `--force` to overwrite raw files after changing the model.

`clean_transcripts.py` reads raw files and writes fixed text to `transcripts/`. Re-run with `--force` after updating correction rules in `lib/text.py` or `lib/names.py`.

If upgrading from an older version with inline transcript text in `podcasts_data.json`, run once:

```bash
python migrate_podcasts_data.py
```

## Output Formats

### Transcripts

- **Raw** (`transcripts/raw/`): direct Whisper output, named after the episode MP3
- **Cleaned** (`transcripts/`): after name and term fixes — used by blog, Spotify, and YouTube scripts

### Blog Posts

Markdown files with YAML frontmatter in `blogs/`, compatible with [Ghost CMS](https://ghost.org/):

```markdown
---
slug: walkers-creek-2026
title: "Walkers Creek - Trip Report"
youtube_url: "PLACEHOLDER_YOUTUBE_URL"
spotify_url: "PLACEHOLDER_SPOTIFY_URL"
episode_file: "Take a Hike - Walkers Creek.mp3"
blog_url: "https://townsvillebushwalkingclub.com/walkers-creek-2026/"
---

Blog body in Markdown...
```

Posts use Ghost's root-level URL pattern (`/{slug}/`, not `/blog/{slug}/`). After upload, `upload_spotify.py` and `upload_youtube.py` replace the placeholder URLs with real links.

### Spotify Descriptions

Built programmatically in this structure:

```text
{AI summary/intro}

Read the full blog post: {blog_url}
```

Gemini generates the title and summary; the blog slug URL is included in every description.

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

### Name and term correction

Whisper mishearings are corrected in **step 2** (`clean_transcripts.py`), not during transcription. This includes **Luen Warneke**, **Cherry Judge**, place names (e.g. `Casawary` → **Cassowary**, `Wallamann` → **Wallaman**), and typographic punctuation normalization.

Blog, Spotify, and YouTube scripts still apply `clean_text()` to AI-generated titles and summaries.

## JSON Data Structure

`podcasts_data.json` tracks progress per episode. It stores paths and flags only — not full transcript or blog text.

```json
{
  "Take a Hike - Topic.mp3": {
    "episode_file": "Take a Hike - Topic.mp3",
    "raw_transcript_file": "transcripts/raw/Take a Hike - Topic.txt",
    "transcript_raw_done": true,
    "whisper_model": "large-v3",
    "transcript_file": "transcripts/Take a Hike - Topic.txt",
    "transcript_done": true,
    "blog_slug": "topic-slug-2026",
    "blog_file": "blogs/topic-slug-2026.md",
    "blog_url": "https://townsvillebushwalkingclub.com/topic-slug-2026/",
    "spotify_url": "",
    "spotify_title": "",
    "youtube_id": "",
    "youtube_url": "",
    "youtube_title": ""
  }
}
```

## Troubleshooting

- **"FFmpeg not found"** → Install FFmpeg and ensure it is in your PATH
- **"Gemini cookies required"** → Set `GEMINI_SECURE_1PSID` and `GEMINI_SECURE_1PSIDTS` in `.env`
- **"SPOTIFY_PODCAST_ID must be set"** → Add your show ID from creators.spotify.com to `.env`
- **"spotify-cookies.json not found"** → Export cookies while logged in to creators.spotify.com
- **Spotify upload fails / times out** → Re-export cookies; try `python upload_spotify.py --no-headless`
- **"YouTube credentials not found"** → Download OAuth credentials and save as `client_secret.json`
- **Video creation fails** → Verify `TAH_Podcast_Graphics.jpg` exists and audio files are valid MP3
- **Upload fails** → Check daily YouTube limits, verify YouTube Data API v3 is enabled
- **Whisper MemoryError on large-v3** → Close other apps, ensure ~8GB+ free RAM, or use `WHISPER_MODEL=medium`

## Contributing

This project was created for the Townsville Bushwalking Club's "Take A Hike" podcast processing needs.

## Credits

- **Podcast:** Take A Hike — A LiSTNR production
- **Host:** Blair Woodcock
- **Regular Guests:** Luen Warneke, Cherry Judge
