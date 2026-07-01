"""Playwright-based Spotify for Creators episode upload."""

import html
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from lib.config import SPOTIFY_PODCAST_ID, spotify_cookies_file

logger = logging.getLogger(__name__)

EPISODE_ID_PATTERN = re.compile(r"/episode/([^/?#]+)")


def _get_wizard_url() -> str:
    if not SPOTIFY_PODCAST_ID:
        raise ValueError(
            "SPOTIFY_PODCAST_ID must be set in .env (the show id in "
            "creators.spotify.com/pod/show/<SHOW_ID>/…)."
        )
    return f"https://creators.spotify.com/pod/show/{SPOTIFY_PODCAST_ID}/episode/wizard"


def _get_episodes_list_url(*, page_size: int = 50, page: int = 1) -> str:
    if not SPOTIFY_PODCAST_ID:
        raise ValueError(
            "SPOTIFY_PODCAST_ID must be set in .env (the show id in "
            "creators.spotify.com/pod/show/<SHOW_ID>/…)."
        )
    return (
        f"https://creators.spotify.com/pod/show/{SPOTIFY_PODCAST_ID}/episodes"
        f"?pageSize={page_size}&page={page}"
    )


def open_spotify_episode_url(episode_id: str) -> str:
    """Build a public open.spotify.com episode URL from the episode id."""
    return f"https://open.spotify.com/episode/{episode_id}"


def normalize_spotify_title(title: str) -> str:
    """Normalize titles for fuzzy comparison."""
    text = html.unescape(title).lower()
    text = text.replace("|", " ")
    text = re.sub(r"[^\w\s&']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_match_score(expected: str, published: str) -> float:
    left = normalize_spotify_title(expected)
    right = normalize_spotify_title(published)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        shorter = min(len(left), len(right))
        longer = max(len(left), len(right))
        return 0.92 + (shorter / longer) * 0.07
    return SequenceMatcher(None, left, right).ratio()


def find_published_episode_url(title: str, published_episodes: list[dict]) -> str | None:
    """
    Return the open.spotify.com URL for a published episode matching title.

    When multiple dashboard rows share the same title (duplicate uploads), prefer
    the oldest publish (last row in Spotify's date-descending list).
    """
    matches: list[tuple[float, int, dict]] = []
    for index, episode in enumerate(published_episodes):
        score = _title_match_score(title, episode["title"])
        if score >= 0.88:
            matches.append((score, index, episode))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], -item[1]))
    best_score = matches[0][0]
    best = [episode for score, _, episode in matches if score >= best_score - 0.01]
    chosen = best[-1]
    return chosen["url"]


def _scrape_visible_episode_rows(page) -> list[dict]:
    """Return episode rows currently visible in the Spotify episodes table."""
    raw_episodes = page.evaluate(
        """() => {
            const links = [...document.querySelectorAll('a[href*="/episode/"][href$="/details"]')];
            return links.map((link) => ({
                title: (link.textContent || '').trim(),
                href: link.getAttribute('href') || '',
            }));
        }"""
    )

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for item in raw_episodes:
        href = item.get("href", "")
        match = EPISODE_ID_PATTERN.search(href)
        if not match:
            continue
        episode_id = match.group(1)
        if episode_id in seen_ids:
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        seen_ids.add(episode_id)
        rows.append(
            {
                "title": title,
                "episode_id": episode_id,
                "url": open_spotify_episode_url(episode_id),
            }
        )
    return rows


def _episode_list_has_next_page(page) -> bool:
    forward = page.locator("#page-nums-forward")
    if forward.count() == 0:
        return False
    return forward.first.is_enabled()


def _go_to_next_episode_page(page, *, page_size: int, page_num: int) -> None:
    """Open the next episodes table page via the forward control or URL."""
    forward = page.locator("#page-nums-forward")
    if forward.count() > 0 and _episode_list_has_next_page(page):
        forward.first.click()
        page.wait_for_timeout(1500)
    else:
        page.goto(
            _get_episodes_list_url(page_size=page_size, page=page_num),
            wait_until="domcontentloaded",
            timeout=60000,
        )
        page.wait_for_load_state("load", timeout=15000)
    page.locator('a[href*="/episode/"][href$="/details"]').first.wait_for(
        state="visible", timeout=15000
    )


def list_published_episodes(*, headless: bool = True, page_size: int = 50) -> list[dict]:
    """
    Scrape published episodes from the Spotify for Creators episode list.

    Returns:
        List of dicts with keys: title, episode_id, url (in dashboard order,
        newest first).
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    cookies = _load_cookies()
    episodes_url = _get_episodes_list_url(page_size=page_size)

    with sync_playwright() as playwright:
        launch_kw: dict = {"headless": headless}
        if Path("/.dockerenv").exists():
            launch_kw["args"] = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ]
        browser = playwright.chromium.launch(**launch_kw)
        context = browser.new_context()
        context.add_cookies(cookies)
        page = context.new_page()

        try:
            page.goto(episodes_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_load_state("load", timeout=15000)
            _dismiss_cookie_banners(page)
            page.locator('a[href*="/episode/"][href$="/details"]').first.wait_for(
                state="visible", timeout=30000
            )

            published: list[dict] = []
            seen_ids: set[str] = set()
            page_num = 1

            while True:
                for row in _scrape_visible_episode_rows(page):
                    if row["episode_id"] in seen_ids:
                        continue
                    seen_ids.add(row["episode_id"])
                    published.append(row)

                if not _episode_list_has_next_page(page):
                    break
                page_num += 1
                _go_to_next_episode_page(page, page_size=page_size, page_num=page_num)

            logger.info("Loaded %d published episode(s) from Spotify dashboard", len(published))
            return published

        except PlaywrightTimeout as exc:
            raise RuntimeError(f"Spotify episode list timed out: {exc}") from exc
        finally:
            browser.close()


def _dismiss_cookie_banners(page) -> None:
    try:
        cookie_dialog = page.get_by_role("dialog", name="Privacy")
        cookie_dialog.wait_for(state="visible", timeout=3000)
        cookie_dialog.get_by_role("button", name="Close").click()
        page.wait_for_timeout(500)
    except Exception:
        logger.debug("Privacy dialog not present or already closed", exc_info=True)

    try:
        cookie_close = page.locator("#onetrust-close-btn-container button")
        cookie_close.first.click(timeout=2000)
        page.wait_for_timeout(500)
    except Exception:
        logger.debug("OneTrust close button not found, continuing", exc_info=True)


def _extract_episode_url(page) -> str | None:
    """Try several UI locations for the public episode URL after publish."""
    try:
        copy_btn = page.get_by_role("button", name="Copy")
        copy_btn.first.click(timeout=3000)
        page.wait_for_timeout(500)
        episode_url = page.evaluate("() => navigator.clipboard.readText()")
        if episode_url and "open.spotify.com/episode" in episode_url:
            return episode_url.split("?")[0]
    except Exception:
        logger.debug("Could not get URL from clipboard", exc_info=True)

    try:
        twitter_btn = page.get_by_role("button", name="Twitter")
        href = twitter_btn.first.get_attribute("href")
        if href and "open.spotify.com" in href:
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            raw = (params.get("url") or params.get("text") or [None])[0]
            if raw:
                episode_url = unquote(raw)
                if "open.spotify.com/episode" in episode_url:
                    return episode_url.split("?")[0]
    except Exception:
        logger.debug("Could not parse episode URL from Twitter button", exc_info=True)

    match = EPISODE_ID_PATTERN.search(page.url)
    if match:
        return open_spotify_episode_url(match.group(1))

    return None


def _wait_for_publish_complete(page) -> bool:
    """Wait for Spotify to finish publishing and surface share links or redirect."""
    share_ready = page.get_by_text("Share links are ready", exact=False)
    for _ in range(120):
        if share_ready.count() > 0 and share_ready.first.is_visible():
            return True
        if EPISODE_ID_PATTERN.search(page.url) and "/details" in page.url:
            return True
        page.wait_for_timeout(1000)
    return False


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
            _dismiss_cookie_banners(page)

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
            _dismiss_cookie_banners(page)

            _advance_to_review_step(page, description)

            review_step = page.locator('label[for="publish-date-now"]')
            review_step.click()
            page.wait_for_timeout(1500)

            publish_btn = page.locator('button[form="review-form"]:has-text("Publish")')
            publish_btn.wait_for(state="visible", timeout=5000)
            publish_btn.click()

            if not _wait_for_publish_complete(page):
                episode_url = _extract_episode_url(page)
                if episode_url:
                    logger.warning(
                        "Publish share screen timed out; recovered episode URL from page"
                    )
                    return episode_url
                raise RuntimeError(
                    "Spotify publish timed out waiting for share links"
                )

            page.wait_for_timeout(2000)
            episode_url = _extract_episode_url(page)

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
