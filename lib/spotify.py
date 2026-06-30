"""Playwright-based Spotify for Creators episode upload."""

import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from lib.config import SPOTIFY_PODCAST_ID, spotify_cookies_file

logger = logging.getLogger(__name__)


def _get_wizard_url() -> str:
    if not SPOTIFY_PODCAST_ID:
        raise ValueError(
            "SPOTIFY_PODCAST_ID must be set in .env (the show id in "
            "creators.spotify.com/pod/show/<SHOW_ID>/…)."
        )
    return f"https://creators.spotify.com/pod/show/{SPOTIFY_PODCAST_ID}/episode/wizard"


def _load_cookies() -> list[dict]:
    """Load session cookies from spotify-cookies.json."""
    path = spotify_cookies_file()
    if not path.exists():
        raise ValueError(
            f"Spotify cookies not found at {path}. "
            "Export cookies from your browser when logged in to creators.spotify.com."
        )
    try:
        cookies = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(cookies, list):
        raise ValueError(f"{path.name} must be a JSON array of cookie objects")

    result = []
    for cookie in cookies:
        entry = {
            "name": str(cookie.get("name", "")),
            "value": str(cookie.get("value", "")),
            "domain": cookie.get("domain", "creators.spotify.com"),
            "path": cookie.get("path", "/"),
        }
        expires = cookie.get("expires") or cookie.get("expirationDate")
        if expires is not None:
            entry["expires"] = int(expires)
        if cookie.get("httpOnly") is not None:
            entry["httpOnly"] = bool(cookie["httpOnly"])
        if cookie.get("secure") is not None:
            entry["secure"] = bool(cookie["secure"])
        same_site_raw = cookie.get("sameSite")
        if same_site_raw:
            value = str(same_site_raw).lower()
            if value == "strict":
                entry["sameSite"] = "Strict"
            elif value == "lax":
                entry["sameSite"] = "Lax"
            elif value in ("none", "no_restriction"):
                entry["sameSite"] = "None"
            else:
                entry["sameSite"] = "Lax"
        result.append(entry)
    return result


def publish_episode_to_spotify(
    audio_path: Path,
    thumbnail_path: Path,
    *,
    title: str,
    description: str,
    headless: bool = True,
) -> str:
    """
    Upload a podcast episode to Spotify for Creators via Playwright.

    Returns:
        Spotify episode URL on success.

    Raises:
        ValueError: If cookies, files, or required metadata are missing.
        RuntimeError: On upload/publish failure.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")
    if not thumbnail_path.exists():
        raise ValueError(f"Thumbnail file not found: {thumbnail_path}")
    if not title or not description:
        raise ValueError("Title and description are required")

    cookies = _load_cookies()
    wizard_url = _get_wizard_url()
    abs_audio = str(audio_path.resolve())
    abs_thumb = str(thumbnail_path.resolve())

    logger.info("Publishing to Spotify: %s", title[:50])

    with sync_playwright() as playwright:
        launch_kw: dict = {"headless": headless}
        if Path("/.dockerenv").exists():
            launch_kw["args"] = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ]
        browser = playwright.chromium.launch(**launch_kw)
        context = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
        context.add_cookies(cookies)
        page = context.new_page()

        try:
            page.goto(wizard_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_load_state("load", timeout=10000)

            try:
                cookie_dialog = page.get_by_role("dialog", name="Privacy")
                cookie_dialog.wait_for(state="visible", timeout=3000)
                cookie_dialog.get_by_role("button", name="Close").click()
                page.wait_for_timeout(500)
            except Exception:
                logger.debug("Privacy dialog not present or already closed", exc_info=True)

            upload_input = page.locator("#uploadAreaInput")
            upload_input.wait_for(state="attached", timeout=10000)
            upload_input.set_input_files(abs_audio)

            page.get_by_text("Generating preview").first.wait_for(state="visible", timeout=90000)
            details_form = page.locator("#details-form")
            details_form.wait_for(state="visible", timeout=60000)
            page.wait_for_timeout(1000)

            title_input = page.locator("#title-input")
            title_input.wait_for(state="visible", timeout=5000)
            title_input.fill(title)

            desc_editor = page.locator('[name="description"][contenteditable="true"]')
            if desc_editor.count() > 0:
                element = desc_editor.first
                element.wait_for(state="visible", timeout=5000)
                element.evaluate(
                    """(node, text) => {
                        node.focus();
                        node.innerText = text;
                        node.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertFromPaste' }));
                        node.dispatchEvent(new Event('change', { bubbles: true }));
                    }""",
                    description,
                )
            page.wait_for_timeout(500)

            change_btn = page.locator('button:has-text("Change")').first
            if change_btn.count() > 0:
                with page.expect_file_chooser() as file_chooser:
                    change_btn.click()
                file_chooser.value.set_files(abs_thumb)
            else:
                thumb_input = page.locator('input[accept*="image"][type="file"]').first
                if thumb_input.count() > 0:
                    thumb_input.set_input_files(abs_thumb)

            crop_modal = page.locator('[data-encore-id="dialogConfirmation"]')
            save_btn = crop_modal.get_by_role("button", name="Save")
            try:
                save_btn.first.wait_for(state="visible", timeout=10000)
                page.wait_for_timeout(2000)
                save_btn.first.click()
                crop_modal.wait_for(state="hidden", timeout=30000)
                page.wait_for_timeout(1000)
            except Exception as exc:
                logger.warning("Crop modal Save button not found or skipped: %s", exc)

            try:
                cookie_close = page.locator("#onetrust-close-btn-container button")
                cookie_close.first.click(timeout=2000)
                page.wait_for_timeout(500)
            except Exception:
                logger.debug("OneTrust close button not found, continuing", exc_info=True)

            next_btn = page.locator('button[form="details-form"]:has-text("Next")')
            next_btn.wait_for(state="visible", timeout=5000)
            next_btn.click()
            page.wait_for_timeout(3000)

            now_label = page.locator('label[for="publish-date-now"]')
            now_label.wait_for(state="visible", timeout=10000)
            now_label.click()
            page.wait_for_timeout(1500)

            publish_btn = page.locator('button[form="review-form"]:has-text("Publish")')
            publish_btn.wait_for(state="visible", timeout=5000)
            publish_btn.click()

            page.get_by_text("Share links are ready", exact=False).wait_for(
                state="visible", timeout=60000
            )
            page.wait_for_timeout(2000)

            episode_url = None
            try:
                copy_btn = page.get_by_role("button", name="Copy")
                copy_btn.first.click(timeout=5000)
                page.wait_for_timeout(500)
                episode_url = page.evaluate("() => navigator.clipboard.readText()")
            except Exception as exc:
                logger.warning("Could not get URL from clipboard: %s", exc)

            if not episode_url or "open.spotify.com/episode" not in episode_url:
                try:
                    twitter_btn = page.get_by_role("button", name="Twitter")
                    href = twitter_btn.first.get_attribute("href")
                    if href and "open.spotify.com" in href:
                        parsed = urlparse(href)
                        params = parse_qs(parsed.query)
                        raw = (params.get("url") or params.get("text") or [None])[0]
                        if raw:
                            episode_url = unquote(raw)
                except Exception:
                    logger.debug("Could not parse episode URL from Twitter button", exc_info=True)

            if episode_url and "open.spotify.com/episode" in episode_url:
                return episode_url
            return page.url

        except PlaywrightTimeout as exc:
            raise RuntimeError(f"Spotify upload timed out: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Spotify upload failed: {exc}") from exc
        finally:
            browser.close()
