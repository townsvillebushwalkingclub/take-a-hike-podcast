"""Gemini 3 Pro client wrapper using gemini-webapi."""

import asyncio
import os
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from lib.config import GEMINI_IMAGE_MODEL, GEMINI_LOG_LEVEL, GEMINI_MODEL, resolve_gemini_model

T = TypeVar("T", bound=BaseModel)

_client = None
_client_loop: asyncio.AbstractEventLoop | None = None

_IMAGE_LIMIT_PHRASES = (
    "limit resets",
    "usage limit",
    "check your usage",
    "image generation limit",
)


def _is_image_limit_response(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _IMAGE_LIMIT_PHRASES)


async def _close_client() -> None:
    """Close the shared client and allow a fresh one on the next event loop."""
    global _client, _client_loop
    if _client is None:
        return
    try:
        await _client.close()
    except Exception:
        pass
    _client = None
    _client_loop = None


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
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is not None and _client_loop is not loop:
        await _close_client()

    if _client is not None:
        return _client

    from gemini_webapi import GeminiClient, set_log_level

    set_log_level(GEMINI_LOG_LEVEL)
    _configure_cookie_path()

    secure_1psid, secure_1psidts = _get_credentials()
    client = GeminiClient(secure_1psid, secure_1psidts, proxy=None)
    await client.init(timeout=120, auto_close=False, auto_refresh=True)
    _client = client
    _client_loop = loop
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

    async def _run() -> T:
        try:
            return await generate_json(prompt, model, retries=retries)
        finally:
            await _close_client()

    return asyncio.run(_run())


async def generate_image_edit(
    prompt: str,
    *,
    template_image: Path,
    output_dir: Path,
    filename: str,
    retries: int = 3,
) -> tuple[str, Path]:
    """Edit an image with Gemini Nano Banana and return response text plus saved path."""
    from gemini_webapi import GeneratedImage

    if not template_image.is_file():
        raise FileNotFoundError(f"Template image not found: {template_image}")

    output_dir.mkdir(parents=True, exist_ok=True)
    client = await _get_client()
    image_model = resolve_gemini_model(GEMINI_IMAGE_MODEL)
    last_error = None

    for attempt in range(retries):
        try:
            chat = client.start_chat(model=image_model)
            await chat.send_message(
                "Here is a 1200x630 social sharing image template for the Take A Hike podcast. "
                "Keep this branding in mind for the next request.",
                files=[template_image],
                temporary=False,
            )
            response = await chat.send_message(
                f"Use the image generation tool to GENERATE a new edited version of that template. {prompt}",
                temporary=False,
            )
            generated = [
                image for image in (response.images or []) if isinstance(image, GeneratedImage)
            ]
            if not generated:
                detail = (response.text or "").strip() or "No generated image in response"
                if _is_image_limit_response(detail):
                    raise RuntimeError(
                        "Gemini refused image generation (reported a limit). "
                        "This can be a separate image-gen quota from chat usage, or the model "
                        "may not have invoked Nano Banana. Try again in Gemini web UI first, "
                        f"or wait and retry later. Response: {detail[:500]}"
                    )
                raise RuntimeError(
                    "Gemini did not return a generated image. "
                    f"Response: {detail[:500]}"
                )

            image = generated[0]
            saved_path = Path(
                await image.save(
                    path=str(output_dir),
                    filename=filename,
                    full_size=True,
                    verbose=True,
                )
            )
            return response.text or "", saved_path
        except Exception as exc:
            last_error = exc
            if _is_image_limit_response(str(exc)) or "Unknown model name" in str(exc):
                break
            if attempt < retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"Gemini image request failed ({exc}). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    raise RuntimeError(f"Gemini image request failed after {retries} attempts: {last_error}")


def generate_image_edit_sync(
    prompt: str,
    *,
    template_image: Path,
    output_dir: Path,
    filename: str,
    retries: int = 3,
) -> tuple[str, Path]:
    """Synchronous wrapper for generate_image_edit."""

    async def _run() -> tuple[str, Path]:
        try:
            return await generate_image_edit(
                prompt,
                template_image=template_image,
                output_dir=output_dir,
                filename=filename,
                retries=retries,
            )
        finally:
            await _close_client()

    return asyncio.run(_run())
