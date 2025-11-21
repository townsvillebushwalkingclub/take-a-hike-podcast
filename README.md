# Take A Hike Podcast

A comprehensive processing pipeline for the "Take A Hike" podcast, a LiSTNR production with Blair Woodcock, Luen Warneke, and Cherry Judge. This project automates the transcription, blog post generation, video creation, and YouTube upload workflow for podcast episodes.

Find the audio files from the podcast: <https://drive.google.com/drive/folders/1g2efA-Rw0RiuZEYKuO2ItKbOy30V2nMH?usp=drive_link>

## Overview

This project processes podcast audio episodes by:

1. **Transcribing** audio using `whisper-timestamped` for accurate speech-to-text conversion
2. **Generating blog posts** using Google Gemini AI with title, excerpt, and full content
3. **Creating portrait videos** for social media by combining podcast audio with the podcast graphic
4. **Uploading to YouTube** with optimized titles, descriptions, and hashtags
5. **Tracking progress** via JSON to avoid re-processing or re-uploading content

## Project Structure

```text
take-a-hike-podcast/
├── audio/              # Input podcast audio files (.mp3)
├── videos/                # Generated portrait videos (.mp4)
├── podcasts_data.json     # Processing metadata and YouTube tracking
├── process_podcasts.py    # Main processing script
├── requirements.txt       # Python dependencies
└── TAH_Podcast_Graphics.jpg  # Podcast graphic for video generation
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** You'll also need FFmpeg installed:

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### 2. Configure Google API Key

1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file in the project root:

   ```text
   GOOGLE_API_KEY=your-api-key-here
   ```

### 3. Configure YouTube Upload (Optional)

If you want to upload videos to YouTube:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable YouTube Data API v3
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download and save as `client_secret.json` in the project root
5. On first upload, the script will open a browser for authentication

## Usage

### Process Episodes (Without YouTube Upload)

```bash
python process_podcasts.py
```

This will transcribe all podcasts, generate blog posts, and create portrait videos. Results are saved to `podcasts_data.json` and `videos/` directory.

### Process and Upload to YouTube

```bash
python process_podcasts.py --upload
```

**Note:** YouTube has daily upload limits (typically 6-15 videos/day). The script tracks uploaded videos by YouTube ID and automatically skips already-uploaded videos when you resume the next day.

### Command Line Options

- `--upload`: Upload videos to YouTube (default: False)
- `--api-key KEY`: Google API key (optional, can use GOOGLE_API_KEY in .env)
- `--youtube-credentials PATH`: Path to YouTube OAuth credentials (default: `client_secret.json`)

## JSON Data Structure

The `podcasts_data.json` file tracks all processed episodes with the following structure:

```json
{
  "episode_filename.mp3": {
    "episode_file": "episode_filename.mp3",
    "transcript": "Full transcription text...",
    "blog_title": "Generated blog post title",
    "blog_excerpt": "200-character excerpt...",
    "blog_content": "Full blog post content...",
    "youtube_title": "YouTube optimized title",
    "youtube_description": "YouTube description with hashtags...",
    "youtube_id": "YouTube video ID (if uploaded)"
  }
}
```

### Tracking Features

- **Avoids re-processing**: Episodes already in the JSON file are skipped
- **Resume capability**: Can stop and resume processing at any time
- **Upload tracking**: YouTube ID stored to prevent duplicate uploads
- **Daily limit handling**: Process some videos, resume the next day to upload remaining

## Workflow Details

### 1. Transcription

- Uses `whisper-timestamped` with the "base" model
- Extracts full transcript text from audio
- Handles English language podcast content

### 2. Blog Post Generation

- Uses Google Gemini AI to generate:
  - **Title**: SEO-friendly, under 60 characters
  - **Excerpt**: Exactly 200 characters
  - **Content**: Full blog post with key topics and practical tips

### 3. Video Creation

- Creates portrait videos (1080x1920) for social media
- Combines podcast audio with `TAH_Podcast_Graphics.jpg`
- Videos saved as `.mp4` files in the `videos/` directory

### 4. YouTube Upload

- Generates optimized titles and descriptions with hashtags
- Uploads videos with metadata
- Tracks YouTube IDs to prevent duplicates
- Respects daily upload limits (resume capability built-in)

## Rate Limiting & Daily Limits

YouTube has daily upload limits. The script:

- Tracks uploaded videos via YouTube ID in JSON
- Skips already-uploaded videos automatically
- Allows processing to be stopped and resumed
- Uploads can be done in batches across multiple days

**Example workflow:**

1. Day 1: Process all episodes (transcription + blog + video)
2. Day 1: Upload 10 videos (hits daily limit)
3. Day 2: Resume with `--upload` flag (skips already uploaded, continues with rest)

## Troubleshooting

- **"FFmpeg not found"** → Install FFmpeg and ensure it's in your PATH
- **"Google API key required"** → Set `GOOGLE_API_KEY` in `.env` file
- **"YouTube credentials not found"** → Download OAuth credentials from Google Cloud Console and save as `client_secret.json`
- **Video creation fails** → Verify `TAH_Podcast_Graphics.jpg` exists and audio files are valid MP3 format
- **Upload fails** → Check daily limit, verify YouTube Data API v3 is enabled, check internet connection

## Contributing

This project was created for the Townsville Bushwalking Club's "Take A Hike" podcast processing needs.

## License

[Add license information here]

## Credits

- **Podcast**: Take A Hike - A LiSTNR production
- **Host**: Blair Woodcock
- **Regular Guests**: Luen Warneke, Cherry Judge
- **Special Guests**: Various hiking enthusiasts from Townsville and beyond
