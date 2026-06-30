"""Playwright-based Spotify for Creators episode upload."""

import json
import logging
import re
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

            _clear_season_episode_numbers(page)

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
            _save_episode_art_crop(page, crop_modal)

            _fill_description(page, description)

            try:
                cookie_close = page.locator("#onetrust-close-btn-container button")
                cookie_close.first.click(timeout=2000)
                page.wait_for_timeout(500)
            except Exception:
                logger.debug("OneTrust close button not found, continuing", exc_info=True)

            _advance_to_review_step(page, description)

            review_step = page.locator('label[for="publish-date-now"]')
            review_step.click()
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


def _clear_season_episode_numbers(page) -> None:
    """Leave Spotify season and episode number fields blank."""
    for selector in ("#season-number", "#episode-number"):
        field = page.locator(selector)
        if field.count() > 0:
            field.fill("")
            logger.debug("Cleared %s", selector)


def _save_episode_art_crop(page, crop_modal) -> None:
    """Confirm the episode art crop dialog when it appears."""
    save_btn = crop_modal.get_by_role("button", name="Save")
    try:
        save_btn.first.wait_for(state="visible", timeout=15000)
        page.wait_for_timeout(1500)
        save_btn.first.click()
        page.wait_for_timeout(1000)
    except Exception as exc:
        logger.warning("Crop modal Save button not found or skipped: %s", exc)
        return

    _dismiss_blocking_overlays(page)


def _dismiss_blocking_overlays(page) -> None:
    """Close crop/confirmation dialogs whose backdrop blocks the details form."""
    save_selectors = (
        '[data-encore-id="dialogConfirmation"] button:has-text("Save")',
        'button:has-text("Save")',
    )
    for attempt in range(4):
        visible_save = None
        for selector in save_selectors:
            save = page.locator(selector).first
            if save.count() > 0 and save.is_visible():
                visible_save = save
                break
        if visible_save is None:
            return
        try:
            visible_save.click(timeout=3000)
            page.wait_for_timeout(1500)
        except Exception:
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)


def _description_text_length(page) -> int:
    """Return the current character count in the description editor."""
    editor = page.locator('[name="description"][contenteditable="true"]')
    if editor.count() == 0:
        return 0
    return editor.first.evaluate(
        "(node) => (node.innerText || node.textContent || '').trim().length"
    )


def _description_counter_value(page) -> int | None:
    """Parse Spotify's description counter near the details form."""
    try:
        form_text = page.locator("#details-form").inner_text()
    except Exception:
        return None
    counts = [int(match) for match in re.findall(r"(\d+)\s*/\s*4000", form_text)]
    return max(counts) if counts else None


def _description_has_validation_error(page) -> bool:
    """True when Spotify still marks the description field as invalid."""
    editor = page.locator('[name="description"][contenteditable="true"]').first
    if editor.count() == 0:
        return True
    if editor.get_attribute("aria-invalid") == "true":
        return True
    section = page.locator('label:has-text("Description")').locator("xpath=ancestor::div[1]")
    if section.count() == 0:
        return False
    required = section.get_by_text("Required", exact=True)
    return required.count() > 0 and required.first.is_visible()


def _description_accepted_by_spotify(page, description: str) -> bool:
    """True when Spotify's form state accepts the description (not just DOM text)."""
    if _description_has_validation_error(page):
        return False

    expected = len(description.strip())
    if expected == 0:
        return False
    min_chars = max(100, int(expected * 0.85))

    counter = _description_counter_value(page)
    if counter is not None and counter >= min_chars:
        return True

    return _description_text_length(page) >= min_chars


def _fill_description(page, description: str) -> None:
    """Fill the Spotify contenteditable description field, verifying it stuck."""
    desc_editor = page.locator('[name="description"][contenteditable="true"]')
    if desc_editor.count() == 0:
        raise RuntimeError("Spotify description editor not found")

    element = desc_editor.first
    element.wait_for(state="visible", timeout=5000)

    fill_methods = (
        ("keyboard", _fill_description_via_keyboard),
        ("exec_command", _fill_description_via_exec_command),
        ("clipboard", _fill_description_via_clipboard),
    )

    for attempt, (method_name, fill_fn) in enumerate(fill_methods, start=1):
        fill_fn(page, element, description)
        if _description_accepted_by_spotify(page, description):
            logger.debug("Description filled via %s (%d chars)", method_name, _description_text_length(page))
            return

        logger.warning(
            "Description fill attempt %d (%s) rejected by Spotify (text=%d, counter=%s, required=%s)",
            attempt,
            method_name,
            _description_text_length(page),
            _description_counter_value(page),
            _description_has_validation_error(page),
        )

    raise RuntimeError(
        "Spotify description field not accepted "
        f"(text={_description_text_length(page)}, counter={_description_counter_value(page)}, "
        f"required_error={_description_has_validation_error(page)})"
    )


def _fill_description_via_keyboard(page, element, description: str) -> None:
    element.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    page.wait_for_timeout(200)
    page.keyboard.insert_text(description)
    page.wait_for_timeout(500)
    element.evaluate(
        """(node) => {
            node.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText' }));
            node.dispatchEvent(new Event('change', { bubbles: true }));
            node.blur();
        }"""
    )
    page.wait_for_timeout(800)


def _fill_description_via_exec_command(page, element, description: str) -> None:
    element.evaluate(
        """(node, text) => {
            node.focus();
            const range = document.createRange();
            range.selectNodeContents(node);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            document.execCommand('delete', false);
            document.execCommand('insertText', false, text);
            node.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText' }));
            node.dispatchEvent(new Event('change', { bubbles: true }));
            node.blur();
        }""",
        description,
    )
    page.wait_for_timeout(800)


def _fill_description_via_clipboard(page, element, description: str) -> None:
    page.evaluate("(text) => navigator.clipboard.writeText(text)", description)
    element.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Control+V")
    page.wait_for_timeout(800)
    element.evaluate(
        """(node) => {
            node.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertFromPaste' }));
            node.dispatchEvent(new Event('change', { bubbles: true }));
            node.blur();
        }"""
    )
    page.wait_for_timeout(500)


def _advance_to_review_step(page, description: str) -> None:
    """Click through the details step and wait for the review/publish screen."""
    next_btn = page.locator('button[form="details-form"]:has-text("Next")')
    review_step = page.locator('label[for="publish-date-now"]')

    for attempt in range(4):
        if review_step.is_visible():
            return

        _dismiss_blocking_overlays(page)

        if next_btn.count() == 0 or not next_btn.is_visible():
            review_step.wait_for(state="visible", timeout=15000)
            return

        if next_btn.is_disabled():
            alerts = page.locator('[role="alert"]').all_text_contents()
            raise RuntimeError(f"Spotify details form invalid: {alerts or 'Next button disabled'}")

        _fill_description(page, description)
        _dismiss_blocking_overlays(page)
        next_btn.click()
        try:
            review_step.wait_for(state="visible", timeout=20000)
            return
        except Exception:
            logger.warning("Review step not visible after Next (attempt %d)", attempt + 1)

    page.screenshot(path="spotify_upload_failure.png", full_page=True)
    raise RuntimeError(
        "Could not advance to Spotify review step. "
        "Saved screenshot to spotify_upload_failure.png"
    )
