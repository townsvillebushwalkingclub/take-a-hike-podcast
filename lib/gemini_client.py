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
                model=resolve_gemini_model(GEMINI_MODEL),
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


async def _save_first_generated_image(
    client,
    response,
    output_dir: Path,
    filename: str,
) -> tuple[str, Path]:
    """Extract and save the first GeneratedImage from a model response."""
    from gemini_webapi import GeneratedImage

    generated = [
        image for image in (response.images or []) if isinstance(image, GeneratedImage)
    ]
    if not generated:
        detail = (response.text or "").strip() or "No generated image in response"
        if _is_image_limit_response(detail):
            raise RuntimeError(f"IMAGE_LIMIT:{detail[:500]}")
        raise RuntimeError(f"NO_IMAGE:{detail[:500]}")

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


def _build_single_shot_prompt(
    template_image: Path,
    extra_images: list[Path],
    prompt: str,
) -> str:
    """Combine file context and creative brief into one image-generation request."""
    lines = [
        "Use the image generation tool to GENERATE one new podcast cover image.",
        "",
        "Attached reference images:",
        f"- {template_image.name}: Take A Hike podcast cover template (brand reference)",
    ]
    for ref in extra_images:
        lines.append(
            f"- {ref.name}: Townsville Bushwalking Club logo (incorporate into the design)"
        )
    lines.extend(["", prompt])
    return "\n".join(lines)


def _build_chat_setup_message(template_image: Path, extra_images: list[Path]) -> str:
    lines = [
        "We are generating a series of Take A Hike podcast episode covers for the Townsville Bushwalking Club website.",
        "Reference images:",
        f"1. Podcast cover template ({template_image.name}) — brand and title treatment",
    ]
    for index, ref in enumerate(extra_images, start=2):
        lines.append(f"{index}. Townsville Bushwalking Club logo ({ref.name})")
    lines.append(
        "I will ask you to GENERATE a new cover for each episode in this same conversation. "
        "Keep style, branding, and quality consistent across the series while making each episode visually distinct."
    )
    return "\n".join(lines)


def _build_existing_covers_message(count: int) -> str:
    return (
        f"These {count} podcast covers are already approved for this series. "
        "Match their visual style, branding, title treatment, and overall quality "
        "for every new cover you generate in this conversation."
    )


class CoverImageSession:
    """Persistent Gemini chat for generating a consistent cover image series."""

    def __init__(self) -> None:
        self.client = None
        self.chat = None
        self.image_model = None

    async def start(
        self,
        *,
        template_image: Path,
        reference_images: list[Path],
        existing_covers: list[Path] | None = None,
    ) -> None:
        self.client = await _get_client()
        self.image_model = resolve_gemini_model(GEMINI_IMAGE_MODEL)
        self.chat = self.client.start_chat(model=self.image_model)

        upload_files = [template_image, *reference_images]
        await self.chat.send_message(
            _build_chat_setup_message(template_image, reference_images),
            files=upload_files,
            temporary=False,
        )

        if existing_covers:
            await self.chat.send_message(
                _build_existing_covers_message(len(existing_covers)),
                files=existing_covers,
                temporary=False,
            )

    async def generate(
        self,
        prompt: str,
        *,
        output_dir: Path,
        filename: str,
        retries: int = 3,
    ) -> tuple[str, Path]:
        if self.chat is None:
            raise RuntimeError("CoverImageSession.start() must be called before generate()")

        chat_prompt = (
            "Use the image generation tool to GENERATE the next podcast cover image in this series. "
            "Match the style of previous covers in this conversation. "
            f"{prompt}"
        )
        last_error = None
        saw_limit = False

        for attempt in range(retries):
            try:
                response = await self.chat.send_message(chat_prompt, temporary=False)
                return await _save_first_generated_image(
                    self.client, response, output_dir, filename
                )
            except RuntimeError as exc:
                message = str(exc)
                if message.startswith("IMAGE_LIMIT:"):
                    saw_limit = True
                    last_error = RuntimeError(message.removeprefix("IMAGE_LIMIT:"))
                    break
                if message.startswith("NO_IMAGE:"):
                    last_error = RuntimeError(message.removeprefix("NO_IMAGE:"))
                else:
                    last_error = exc
            except Exception as exc:
                last_error = exc

            if attempt < retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"Gemini image request failed ({last_error}). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        if saw_limit:
            raise RuntimeError(
                "Gemini refused image generation (reported a limit). "
                "Image generation has its own quota separate from chat. "
                "Try generating a test image at gemini.google.com, wait for the limit to reset, "
                f"then retry. Last response: {last_error}"
            )
        raise RuntimeError(f"Gemini image request failed after {retries} attempts: {last_error}")


async def _generate_image_edit_standalone(
    prompt: str,
    *,
    template_image: Path,
    reference_images: list[Path] | None,
    output_dir: Path,
    filename: str,
    retries: int = 3,
) -> tuple[str, Path]:
    """Generate one cover in a fresh chat (single episode mode)."""
    extra_images = list(reference_images or [])
    client = await _get_client()
    image_model = resolve_gemini_model(GEMINI_IMAGE_MODEL)
    upload_files = [template_image, *extra_images]
    single_shot_prompt = _build_single_shot_prompt(template_image, extra_images, prompt)
    chat_setup = _build_chat_setup_message(template_image, extra_images)
    chat_prompt = (
        "Use the image generation tool to GENERATE a new podcast cover image. "
        f"{prompt}"
    )

    last_error = None
    saw_limit = False

    for attempt in range(retries):
        strategies = (
            ("single-shot", single_shot_prompt),
            ("chat", chat_prompt),
        )
        for strategy_name, request_prompt in strategies:
            try:
                if strategy_name == "single-shot":
                    response = await client.generate_content(
                        request_prompt,
                        files=upload_files,
                        model=image_model,
                        temporary=False,
                    )
                else:
                    chat = client.start_chat(model=image_model)
                    await chat.send_message(
                        chat_setup,
                        files=upload_files,
                        temporary=False,
                    )
                    response = await chat.send_message(request_prompt, temporary=False)

                return await _save_first_generated_image(
                    client, response, output_dir, filename
                )
            except RuntimeError as exc:
                message = str(exc)
                if message.startswith("IMAGE_LIMIT:"):
                    saw_limit = True
                    last_error = RuntimeError(message.removeprefix("IMAGE_LIMIT:"))
                    print(f"  {strategy_name}: image limit response, trying next approach...")
                    continue
                if message.startswith("NO_IMAGE:"):
                    last_error = RuntimeError(message.removeprefix("NO_IMAGE:"))
                    print(f"  {strategy_name}: no generated image, trying next approach...")
                    continue
                last_error = exc
            except Exception as exc:
                last_error = exc

        if saw_limit and attempt == retries - 1:
            break
        if attempt < retries - 1:
            wait_time = 30 * (attempt + 1)
            print(f"Gemini image request failed ({last_error}). Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    if saw_limit:
        raise RuntimeError(
            "Gemini refused image generation (reported a limit). "
            "Image generation has its own quota separate from chat. "
            "Try generating a test image at gemini.google.com, wait for the limit to reset, "
            f"then retry. Last response: {last_error}"
        )
    raise RuntimeError(f"Gemini image request failed after {retries} attempts: {last_error}")


async def generate_image_edit(
    prompt: str,
    *,
    template_image: Path,
    reference_images: list[Path] | None = None,
    output_dir: Path,
    filename: str,
    retries: int = 3,
) -> tuple[str, Path]:
    """Edit an image with Gemini Nano Banana and return response text plus saved path."""
    if not template_image.is_file():
        raise FileNotFoundError(f"Template image not found: {template_image}")

    extra_images = [path for path in (reference_images or []) if path.is_file()]
    missing_refs = [path for path in (reference_images or []) if not path.is_file()]
    if missing_refs:
        missing = ", ".join(str(path) for path in missing_refs)
        raise FileNotFoundError(f"Reference image(s) not found: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return await _generate_image_edit_standalone(
        prompt,
        template_image=template_image,
        reference_images=extra_images,
        output_dir=output_dir,
        filename=filename,
        retries=retries,
    )


async def run_cover_image_batch(
    jobs: list[tuple[str, str]],
    *,
    template_image: Path,
    reference_images: list[Path],
    output_dir: Path,
    existing_covers: list[Path] | None = None,
) -> list[tuple[str, str | None, Path | None, str | None]]:
    """Generate covers in one chat session. Each job is (slug, prompt)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str | None, Path | None, str | None]] = []

    session = CoverImageSession()
    try:
        await session.start(
            template_image=template_image,
            reference_images=reference_images,
            existing_covers=existing_covers,
        )
        for slug, prompt in jobs:
            try:
                _, saved_path = await session.generate(
                    prompt,
                    output_dir=output_dir,
                    filename=f"{slug}-sharing.jpg",
                )
                results.append((slug, f"images/{saved_path.name}", saved_path, None))
            except Exception as exc:
                results.append((slug, None, None, str(exc)))
    finally:
        await _close_client()

    return results


def run_cover_image_batch_sync(
    jobs: list[tuple[str, str]],
    *,
    template_image: Path,
    reference_images: list[Path],
    output_dir: Path,
    existing_covers: list[Path] | None = None,
) -> list[tuple[str, str | None, Path | None, str | None]]:
    """Synchronous wrapper for run_cover_image_batch."""

    async def _run() -> list[tuple[str, str | None, Path | None, str | None]]:
        return await run_cover_image_batch(
            jobs,
            template_image=template_image,
            reference_images=reference_images,
            output_dir=output_dir,
            existing_covers=existing_covers,
        )

    return asyncio.run(_run())


def generate_image_edit_sync(
    prompt: str,
    *,
    template_image: Path,
    reference_images: list[Path] | None = None,
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
                reference_images=reference_images,
                output_dir=output_dir,
                filename=filename,
                retries=retries,
            )
        finally:
            await _close_client()

    return asyncio.run(_run())
