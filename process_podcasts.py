#!/usr/bin/env python3
"""
Take A Hike Podcast Processing Script

This script processes podcast audio files by:
1. Transcribing audio using whisper-timestamped
2. Generating blog posts using Google Gemini 2.5 Pro
3. Creating portrait videos for social media
4. Uploading videos to YouTube (with tracking to avoid duplicates)
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import whisper_timestamped as whisper
from google import genai
from moviepy import AudioFileClip, ImageClip, CompositeVideoClip

from PIL import Image
import google.auth.transport.requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
PODCASTS_DIR = "audio"
VIDEOS_DIR = "videos"
GRAPHIC_FILE = "TAH_Podcast_Graphics.jpg"
JSON_FILE = "podcasts_data.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# Google Gemini API setup
GEMINI_MODEL = "gemini-2.5-pro"


# Pydantic model for structured outputs - combines blog post and YouTube metadata
class EpisodeContent(BaseModel):
    """Combined blog post and YouTube metadata structure."""
    blog_title: str = Field(description="An engaging, SEO-friendly title under 60 characters")
    blog_excerpt: str = Field(description="A 200-character excerpt/summary that hooks readers")
    blog_content: str = Field(description="Full blog post content that captures key points, stories, and advice from the episode")
    youtube_title: str = Field(description="Engaging YouTube title under 100 characters, includes relevant keywords for hiking/bushwalking in Townsville")
    youtube_description: str = Field(description="Full YouTube description with first 2-3 lines hooking viewers, key topics covered, relevant hashtags (10-15), and includes 'Take A Hike Podcast - A LiSTNR production with Blair Woodcock, Luen Warneke, and Cherry Judge'")


class PodcastProcessor:
    def __init__(self, google_api_key: Optional[str] = None, youtube_credentials_path: Optional[str] = None):
        """
        Initialize the podcast processor.
        
        Args:
            google_api_key: Google API key for Gemini (can also be set via GOOGLE_API_KEY env var or .env file)
            youtube_credentials_path: Path to YouTube OAuth credentials JSON file (defaults to client_secret.json)
        """
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if self.google_api_key:
            # Initialize the client for structured outputs with google-genai
            self.client = genai.Client(api_key=self.google_api_key)
        else:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY in .env file, environment variable, or pass as argument.")
        
        # Check for client_secret.json first (common name)
        if os.path.exists("client_secret.json"):
            self.youtube_credentials_path = "client_secret.json"
        
        self.youtube_service = None
        
        # Load existing data
        self.podcasts_data = self.load_podcasts_data()
        
        # Create directories
        os.makedirs(VIDEOS_DIR, exist_ok=True)
    
    def load_podcasts_data(self) -> Dict:
        """Load existing podcasts data from JSON file."""
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_podcasts_data(self):
        """Save podcasts data to JSON file."""
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.podcasts_data, f, indent=2, ensure_ascii=False)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file using whisper-timestamped.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        print(f"Transcribing {audio_path}...")
        
        # Load audio and model
        audio = whisper.load_audio(audio_path)
        model = whisper.load_model("base", device="cpu")
        
        # Transcribe with timestamps
        result = whisper.transcribe(model, audio, language="en")
        
        # Extract text from segments
        transcript = ""
        for segment in result.get("segments", []):
            transcript += segment.get("text", "") + " "
        
        return transcript.strip()
    
    def generate_episode_content(self, transcript: str, episode_title: str) -> Dict[str, str]:
        """
        Generate blog post content and YouTube metadata in a single API call using Google Gemini 2.5 Pro.
        
        Args:
            transcript: The podcast transcript
            episode_title: The episode title from filename
            
        Returns:
            Dictionary with blog_title, blog_excerpt, blog_content, youtube_title, and youtube_description
        """
        print(f"Generating blog post and YouTube metadata for episode: {episode_title}...")
        
        prompt = f"""You are a content creator for the "Take A Hike" podcast, a show about bushwalking and hiking adventures in and around Townsville, Australia. The podcast is hosted by Blair Woodcock with regular guests Luen Warneke and Cherry Judge, along with occasional special guests.

Based on the following podcast transcript, create both:
1. A blog post (title, excerpt, and full content)
2. YouTube metadata (title and description)

Episode title from filename: {episode_title}

Transcript:
{transcript}

BLOG POST REQUIREMENTS:
- Blog title: An engaging, SEO-friendly title under 60 characters
- Blog excerpt: A 200-character excerpt/summary that hooks readers
- Blog content: Full blog post that captures key points, stories, and advice from the episode
  * Be well-structured with clear paragraphs
  * Include the key topics discussed
  * Maintain the conversational and informative tone
  * Be engaging for hiking and bushwalking enthusiasts
  * Include practical tips and information when relevant

YOUTUBE METADATA REQUIREMENTS:
- YouTube title: Engaging, under 100 characters, includes relevant keywords for hiking/bushwalking in Townsville, North Queensland
- YouTube description:
  * First 2-3 lines should hook viewers
  * Include key topics covered
  * Add relevant hashtags (10-15 hashtags related to hiking, bushwalking, Townsville, Australia, outdoor adventures)
  * Include: "Take A Hike Podcast - A LiSTNR production with Blair Woodcock, Luen Warneke, and Cherry Judge"
  * Optimized for YouTube algorithm with keywords
"""
        
        try:
            # Use structured outputs with combined Pydantic model
            # Add retry logic for rate limiting
            import time
            max_retries = 3
            retry_delay = 60  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_json_schema": EpisodeContent.model_json_schema(),
                        },
                    )
                    break  # Success, exit retry loop
                except Exception as api_error:
                    error_str = str(api_error)
                    # Check for rate limit or quota errors
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                            print(f"API rate limit hit. Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise Exception(f"API rate limit exceeded after {max_retries} attempts. Please try again later.")
                    else:
                        raise  # Re-raise if it's not a rate limit error
            
            # Parse response using Pydantic model
            content = EpisodeContent.model_validate_json(response.text)
            
            # Ensure excerpt is exactly 200 characters
            excerpt = content.blog_excerpt
            if len(excerpt) > 200:
                excerpt = excerpt[:197] + "..."
            
            # Ensure YouTube title is under 100 characters
            youtube_title = content.youtube_title
            if len(youtube_title) > 100:
                youtube_title = youtube_title[:97] + "..."
            
            return {
                "blog_title": content.blog_title,
                "blog_excerpt": excerpt,
                "blog_content": content.blog_content,
                "youtube_title": youtube_title,
                "youtube_description": content.youtube_description
            }
            
        except Exception as e:
            print(f"Error generating episode content: {e}")
            # Fallback to basic content
            hashtags = "#TakeAHike #Hiking #Bushwalking #Townsville #Australia #OutdoorAdventures #HikingTips #TrailGuides #QueenslandHiking #Nature #Adventure #ExploreTownsville"
            excerpt = transcript[:197] + "..." if len(transcript) > 200 else transcript
            return {
                "blog_title": episode_title,
                "blog_excerpt": excerpt,
                "blog_content": f"<p>{transcript}</p>",
                "youtube_title": episode_title[:100],
                "youtube_description": f"Take A Hike Podcast - A LiSTNR production with Blair Woodcock, Luen Warneke, and Cherry Judge.\n\n{transcript[:500]}...\n\n{hashtags}"
            }
    
    def create_video(self, audio_path: str, output_path: str):
        """
        Create a portrait video from audio and graphic image.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path where the video will be saved
        """
        print(f"Creating video: {output_path}...")
        
        # Load audio
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        # Resize image to portrait dimensions while maintaining aspect ratio
        target_width = 1080
        target_height = 1920
        
        # Get current image dimensions
        img = Image.open(GRAPHIC_FILE)
        img_width, img_height = img.size
        
        # Calculate scaling to fit portrait frame
        scale_w = target_width / img_width
        scale_h = target_height / img_height
        scale = max(scale_w, scale_h)  # Use max to ensure image covers the frame
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Crop to exact portrait size (center crop)
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        cropped_img = resized_img.crop((left, top, left + target_width, top + target_height))
        
        # Save temporary resized image
        temp_image_path = "temp_portrait.jpg"
        cropped_img.save(temp_image_path, quality=95)
        
        # Create video clip from image
        # MoviePy 2.x: methods are now with_* and return copies (resize -> resized)
        video_clip = ImageClip(temp_image_path, duration=duration)
        
        # Resize video clip to exact dimensions - MoviePy 2.x uses .resized() with new_size parameter
        video_clip = video_clip.resized(new_size=(target_width, target_height))
        
        # Combine video with audio - MoviePy 2.x: set_audio is now with_audio
        try:
            # MoviePy 2.x method: use with_audio instead of set_audio
            final_clip = CompositeVideoClip([video_clip], duration=duration)
            final_clip = final_clip.with_audio(audio_clip)
        except (AttributeError, TypeError) as e:
            # Fallback: try with_audio method
            try:
                final_clip = video_clip.with_audio(audio_clip)
            except AttributeError:
                # Last resort: create composite and use with_audio
                final_clip = CompositeVideoClip([video_clip])
                final_clip = final_clip.with_audio(audio_clip)
        
        # Write video file with progress bar
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            bitrate='5000k',
            threads=4,
            logger="bar"  # Show progress bar
        )
        
        # Cleanup
        audio_clip.close()
        video_clip.close()
        final_clip.close()
        
        # Remove temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        print(f"Video created: {output_path}")
    
    def authenticate_youtube(self):
        """Authenticate and build YouTube API service."""
        if self.youtube_service:
            return
        
        creds = None
        token_file = "youtube_token.json"
        
        # Load existing token
        if os.path.exists(token_file):
            creds = google.oauth2.credentials.Credentials.from_authorized_user_file(
                token_file, SCOPES
            )
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(google.auth.transport.requests.Request())
            else:
                if not os.path.exists(self.youtube_credentials_path):
                    raise FileNotFoundError(
                        f"YouTube credentials file not found: {self.youtube_credentials_path}\n"
                        "Please download OAuth 2.0 credentials from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.youtube_credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.youtube_service = build('youtube', 'v3', credentials=creds)
    
    def upload_to_youtube(self, video_path: str, title: str, description: str) -> Optional[str]:
        """
        Upload video to YouTube.
        
        Args:
            video_path: Path to the video file
            title: YouTube video title
            description: YouTube video description
            
        Returns:
            YouTube video ID if successful, None otherwise
        """
        if not self.youtube_service:
            self.authenticate_youtube()
        
        print(f"Uploading to YouTube: {title}...")
        
        try:
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'categoryId': '22',  # People & Blogs category
                    'tags': ['hiking', 'bushwalking', 'Townsville', 'Australia', 'outdoor adventures']
                },
                'status': {
                    'privacyStatus': 'public'  # Change to 'unlisted' or 'private' if needed
                }
            }
            
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
            
            insert_request = self.youtube_service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Execute upload
            response = None
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    print(f"Upload progress: {int(status.progress() * 100)}%")
            
            if 'id' in response:
                video_id = response['id']
                print(f"Video uploaded successfully! ID: {video_id}")
                return video_id
            else:
                print("Upload failed - no video ID in response")
                return None
                
        except Exception as e:
            print(f"Error uploading to YouTube: {e}")
            return None
    
    def process_episode(self, audio_path: str, upload_to_youtube: bool = False):
        """
        Process a single podcast episode.
        
        Args:
            audio_path: Path to the audio file
            upload_to_youtube: Whether to upload to YouTube (default: False)
        """
        episode_key = os.path.basename(audio_path)
        
        # Load existing episode data if available, otherwise create new
        if episode_key in self.podcasts_data:
            episode_data = self.podcasts_data[episode_key].copy()
            print(f"Resuming processing for {episode_key} - found existing data")
        else:
            # Initialize new episode data
            episode_data = {
                "episode_file": episode_key,
                "transcript": "",
                "blog_title": "",
                "blog_excerpt": "",
                "blog_content": "",
                "youtube_title": "",
                "youtube_description": "",
                "youtube_id": ""
            }
        
        # Check if already fully processed (has all required data)
        if episode_data.get("transcript") and episode_data.get("blog_title") and episode_data.get("youtube_title"):
            print(f"Skipping {episode_key} - already fully processed")
            
            # Upload if needed and not already uploaded
            if upload_to_youtube and not episode_data.get("youtube_id"):
                video_path = os.path.join(VIDEOS_DIR, episode_key.replace(".mp3", ".mp4"))
                if os.path.exists(video_path):
                    self.authenticate_youtube()
                    youtube_id = self.upload_to_youtube(
                        video_path,
                        episode_data.get("youtube_title", ""),
                        episode_data.get("youtube_description", "")
                    )
                    if youtube_id:
                        episode_data["youtube_id"] = youtube_id
                        self.podcasts_data[episode_key] = episode_data
                        self.save_podcasts_data()
            return
        
        # Step 1: Transcribe
        if not episode_data["transcript"]:
            episode_data["transcript"] = self.transcribe_audio(audio_path)
            # Save after transcription
            self.podcasts_data[episode_key] = episode_data
            self.save_podcasts_data()
        
        # Step 2: Generate blog post and YouTube metadata in one API call
        if not episode_data["blog_title"] or not episode_data["youtube_title"]:
            print(f"Generating blog post and YouTube metadata for episode: {episode_key}...")
            content_data = self.generate_episode_content(episode_data["transcript"], episode_key)
            episode_data["blog_title"] = content_data["blog_title"]
            episode_data["blog_excerpt"] = content_data["blog_excerpt"]
            episode_data["blog_content"] = content_data["blog_content"]
            episode_data["youtube_title"] = content_data["youtube_title"]
            episode_data["youtube_description"] = content_data["youtube_description"]
            # Save after blog/YouTube metadata generation
            self.podcasts_data[episode_key] = episode_data
            self.save_podcasts_data()
        
        # Step 3: Create video
        video_filename = episode_key.replace(".mp3", ".mp4")
        video_path = os.path.join(VIDEOS_DIR, video_filename)
        
        if not os.path.exists(video_path):
            self.create_video(audio_path, video_path)
            # Note: Video creation doesn't need to be saved to JSON, but we save anyway
            # to track progress (e.g., if script crashes before final save)
            self.podcasts_data[episode_key] = episode_data
            self.save_podcasts_data()
        
        # Step 4: Upload to YouTube (if requested and not already uploaded)
        if upload_to_youtube and not episode_data.get("youtube_id"):
            self.authenticate_youtube()
            youtube_id = self.upload_to_youtube(
                video_path,
                episode_data["youtube_title"],
                episode_data["youtube_description"]
            )
            if youtube_id:
                episode_data["youtube_id"] = youtube_id
                # Save after YouTube upload
                self.podcasts_data[episode_key] = episode_data
                self.save_podcasts_data()
        
        # Final save (in case episode was already fully processed and we're just checking)
        self.podcasts_data[episode_key] = episode_data
        self.save_podcasts_data()
        
        print(f"Completed processing: {episode_key}\n")
    
    def process_all(self, upload_to_youtube: bool = False):
        """
        Process all podcast episodes.
        
        Args:
            upload_to_youtube: Whether to upload videos to YouTube
        """
        audio_files = glob.glob(os.path.join(PODCASTS_DIR, "*.mp3"))
        audio_files.sort()
        
        print(f"Found {len(audio_files)} podcast episodes to process\n")
        
        for idx, audio_file in enumerate(audio_files, start=1):
            episode_name = os.path.basename(audio_file)
            print(f"{idx}/{len(audio_files)} Processing {episode_name}\n")
            try:
                self.process_episode(audio_file, upload_to_youtube=upload_to_youtube)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Take A Hike podcast episodes")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload videos to YouTube (default: False, only process locally)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Google API key (or set GOOGLE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--youtube-credentials",
        type=str,
        default=None,
        help="Path to YouTube OAuth credentials JSON file (defaults to client_secret.json)"
    )
    
    args = parser.parse_args()
    
    try:
        processor = PodcastProcessor(
            google_api_key=args.api_key,
            youtube_credentials_path=args.youtube_credentials
        )
        
        processor.process_all(upload_to_youtube=args.upload)
        
        print("\nProcessing complete!")
        print(f"Data saved to: {JSON_FILE}")
        print(f"Videos saved to: {VIDEOS_DIR}/")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

