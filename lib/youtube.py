"""YouTube OAuth authentication and video upload."""

from pathlib import Path
from typing import Optional

import google.auth.transport.requests
import google.oauth2.credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from lib.config import YOUTUBE_CREDENTIALS_FILE, YOUTUBE_SCOPES, YOUTUBE_TOKEN_FILE

_service = None


def authenticate_youtube(credentials_path: Optional[Path] = None):
    """Authenticate and return a YouTube API service."""
    global _service
    if _service is not None:
        return _service

    creds = None
    credentials_file = credentials_path or YOUTUBE_CREDENTIALS_FILE

    if YOUTUBE_TOKEN_FILE.exists():
        creds = google.oauth2.credentials.Credentials.from_authorized_user_file(
            str(YOUTUBE_TOKEN_FILE), YOUTUBE_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            if not credentials_file.exists():
                raise FileNotFoundError(
                    f"YouTube credentials file not found: {credentials_file}\n"
                    "Download OAuth 2.0 credentials from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), YOUTUBE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        with YOUTUBE_TOKEN_FILE.open("w", encoding="utf-8") as token:
            token.write(creds.to_json())

    _service = build("youtube", "v3", credentials=creds)
    return _service


def parse_hashtag_tags(hashtags: str) -> list[str]:
    """Convert hashtag string into YouTube tag list."""
    tags = []
    for token in hashtags.replace(",", " ").split():
        cleaned = token.strip().lstrip("#")
        if cleaned:
            tags.append(cleaned)
    defaults = ["hiking", "bushwalking", "Townsville", "Australia", "outdoor adventures"]
    for tag in defaults:
        if tag not in tags:
            tags.append(tag)
    return tags[:30]


def upload_video(
    video_path: Path,
    title: str,
    description: str,
    hashtags: str = "",
    credentials_path: Optional[Path] = None,
) -> Optional[str]:
    """Upload a video to YouTube as public."""
    service = authenticate_youtube(credentials_path)
    print(f"Uploading to YouTube: {title}...")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "22",
            "tags": parse_hashtag_tags(hashtags),
        },
        "status": {
            "privacyStatus": "public",
        },
    }

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    insert_request = service.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = insert_request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response.get("id")
    if video_id:
        print(f"Video uploaded successfully! ID: {video_id}")
        return video_id

    print("Upload failed - no video ID in response")
    return None
