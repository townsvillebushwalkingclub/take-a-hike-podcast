"""Gemini 2.5 Pro client wrapper using gemini-webapi."""

import asyncio
import os
import re
from typing import TypeVar

from pydantic import BaseModel

from lib.config import GEMINI_LOG_LEVEL, GEMINI_MODEL

T = TypeVar("T", bound=BaseModel)

_client = None


def _configure_cookie_path() -> None:
    cookie_path = os.getenv("GEMINI_COOKIE_PATH")
    if cookie_path:
        os.makedirs(cookie_path, exist_ok=True)
        os.environ["GEMINI_COOKIE_PATH"] = cookie_path


def _get_credentials() -> tuple[str, str]:
    secure_1psid = os.getenv("GEMINI_SECURE_1PSID", "").strip()
    secure_1psidts = os.getenv("GEMINI_SECURE_1PSIDTS", "").strip()
    if not secure_1psid:
        raise ValueError(
            "Gemini cookies required. Set GEMINI_SECURE_1PSID and GEMINI_SECURE_1PSIDTS in .env"
        )
    return secure_1psid, secure_1psidts


async def _get_client():
    global _client
    if _client is not None:
        return _client

    from gemini_webapi import GeminiClient, set_log_level

    set_log_level(GEMINI_LOG_LEVEL)
    _configure_cookie_path()

    secure_1psid, secure_1psidts = _get_credentials()
    client = GeminiClient(secure_1psid, secure_1psidts, proxy=None)
    await client.init(timeout=120, auto_close=False, auto_refresh=True)
    _client = client
    return _client


def extract_json_text(text: str) -> str:
    """Extract JSON from a model response, including fenced code blocks."""
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    if fence_match:
        return fence_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return stripped


def parse_model_json(text: str, model: type[T]) -> T:
    """Parse and validate JSON output from Gemini."""
    payload = extract_json_text(text)
    return model.model_validate_json(payload)


async def generate_json(prompt: str, model: type[T], retries: int = 3) -> T:
    """Generate structured JSON content with retry on transient failures."""
    client = await _get_client()
    last_error = None

    for attempt in range(retries):
        try:
            response = await client.generate_content(
                prompt,
                model=GEMINI_MODEL,
                temporary=True,
            )
            return parse_model_json(response.text, model)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"Gemini request failed ({exc}). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    raise RuntimeError(f"Gemini request failed after {retries} attempts: {last_error}")


def generate_json_sync(prompt: str, model: type[T], retries: int = 3) -> T:
    """Synchronous wrapper for generate_json."""
    return asyncio.run(generate_json(prompt, model, retries=retries))
