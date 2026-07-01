"""
SHL Product Catalog Scraper
----------------------------
Scrapes Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
Saves results to data/catalog.json

Usage:
    python -m scraper.scrape_catalog

Requires:
    pip install playwright beautifulsoup4 requests
    playwright install chromium
"""

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright, Page, TimeoutError as PWTimeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

# SHL test-type code → human label
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_url(href: str) -> str:
    """Return an absolute SHL URL regardless of whether href is relative."""
    if not href:
        return ""
    if href.startswith("http"):
        return href.rstrip("/") + "/"
    return BASE_URL + "/" + href.lstrip("/")


def extract_type_codes(raw: str) -> str:
    """
    Given a raw string like 'Ability & Aptitude, Simulations' or badge text 'A S',
    return comma-separated single-letter codes, e.g. 'A,S'.
    """
    codes = []
    for code, label in TEST_TYPE_MAP.items():
        if code in raw.upper().split() or label.lower() in raw.lower():
            if code not in codes:
                codes.append(code)
    return ",".join(sorted(set(codes))) if codes else raw.strip()


async def safe_text(locator, default: str = "") -> str:
    try:
        return (await locator.first.inner_text()).strip()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Per-product detail page scraping
# ---------------------------------------------------------------------------

async def scrape_product_page(page: Page, url: str) -> dict:
    """Visit an individual product page and extract description + metadata."""
    details = {"description": "", "remote_testing": False, "adaptive_irt": False}
    try:
        await page.goto(url, timeout=20_000, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)

        # Description — try several selectors commonly used for product copy
        for sel in [
            ".product-catalogue__description",
            ".product-detail__description",
            "[class*='description']",
            ".ss-product-detail p",
            "article p",
            ".product__body p",
        ]:
            els = page.locator(sel)
            count = await els.count()
            if count:
                texts = []
                for i in range(min(count, 5)):
                    t = (await els.nth(i).inner_text()).strip()
                    if t:
                        texts.append(t)
                if texts:
                    details["description"] = " ".join(texts)
                    break

        # Remote testing & adaptive IRT indicators
        page_text = await page.inner_text("body")
        details["remote_testing"] = bool(
            re.search(r"remote\s+testing", page_text, re.I)
        )
        details["adaptive_irt"] = bool(
            re.search(r"adaptive|irt", page_text, re.I)
        )
    except Exception as e:
        log.debug(f"Could not scrape detail page {url}: {e}")

    return details


# ---------------------------------------------------------------------------
# Catalog list scraping
# ---------------------------------------------------------------------------

async def extract_items_from_page(page: Page) -> list[dict]:
    """
    Pull product cards/rows from the currently-loaded catalog list page.
    Returns a list of partial assessment dicts.
    """
    items = []

    # SHL catalog renders a table with rows — try multiple selectors
    row_selectors = [
        "table.product-catalogue__table tbody tr",
        "[class*='product-catalogue'] tbody tr",
        "[class*='product-catalogue__row']",
        "[class*='product-list'] tr",
        "tr[data-course-id]",
        ".product-catalogue__item",
    ]

    rows = None
    for sel in row_selectors:
        locs = page.locator(sel)
        cnt = await locs.count()
        if cnt > 0:
            rows = locs
            log.info(f"Found {cnt} rows with selector: {sel}")
            break

    if rows is None:
        log.warning("No product rows found on this page.")
        return items

    count = await rows.count()
    for i in range(count):
        row = rows.nth(i)
        try:
            item = await extract_row(row)
            if item and item.get("name") and item.get("url"):
                items.append(item)
        except Exception as e:
            log.debug(f"Row {i} extraction failed: {e}")

    return items


async def extract_row(row) -> dict | None:
    """Extract a single catalog row into an assessment dict."""
    # Name + URL — look for an anchor tag
    link = row.locator("a").first
    try:
        name = (await link.inner_text()).strip()
        href = await link.get_attribute("href")
        url = normalize_url(href)
    except Exception:
        return None

    if not name or not url:
        return None

    # Test type codes — look for badge spans / td cells
    type_text = ""
    for sel in [
        "[class*='type'] span",
        "td:nth-child(2)",
        "[class*='test-type']",
        ".product-type",
    ]:
        try:
            t = await safe_text(row.locator(sel))
            if t:
                type_text = t
                break
        except Exception:
            pass

    # Duration
    duration = ""
    for sel in ["[class*='duration']", "td:nth-child(4)", ".duration"]:
        try:
            d = await safe_text(row.locator(sel))
            if d:
                duration = d
                break
        except Exception:
            pass

    # Languages
    languages = ""
    for sel in ["[class*='language']", "td:nth-child(5)", ".languages"]:
        try:
            l = await safe_text(row.locator(sel))
            if l:
                languages = l
                break
        except Exception:
            pass

    # Remote testing & adaptive IRT checkmark columns (often ✓ or icons)
    remote_testing = False
    adaptive_irt = False
    try:
        row_text = await row.inner_text()
        cols = [c.strip() for c in row_text.split("\t") if c.strip()]
        # Heuristic: the last 2 columns are often remote + adaptive indicators
        for col in cols[-3:]:
            if "●" in col or "✓" in col or col.lower() in ("yes", "true", "✔"):
                remote_testing = True
    except Exception:
        pass

    return {
        "name": name,
        "url": url,
        "test_type": extract_type_codes(type_text),
        "test_type_raw": type_text,
        "duration": duration,
        "languages": languages,
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt,
        "description": "",
    }


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

async def get_total_count(page: Page) -> int:
    """Try to read the total number of products from the page."""
    for sel in [
        "[class*='total']",
        "[class*='count']",
        ".product-catalogue__count",
        "[class*='results']",
    ]:
        try:
            text = await safe_text(page.locator(sel))
            nums = re.findall(r"\d+", text)
            if nums:
                return int(max(nums, key=int))
        except Exception:
            pass
    return 0


async def has_next_page(page: Page) -> bool:
    """Return True if there is a 'Next' pagination button that is not disabled."""
    for sel in [
        "[class*='pagination'] [class*='next']:not([disabled])",
        "a[rel='next']",
        "button[aria-label='Next']:not([disabled])",
        ".pagination__next:not(.disabled)",
        "[class*='next-page']:not([disabled])",
    ]:
        try:
            el = page.locator(sel).first
            if await el.is_visible():
                return True
        except Exception:
            pass
    return False


async def click_next_page(page: Page) -> bool:
    """Click the Next button and wait for new content to load."""
    for sel in [
        "[class*='pagination'] [class*='next']:not([disabled])",
        "a[rel='next']",
        "button[aria-label='Next']:not([disabled])",
        ".pagination__next:not(.disabled)",
    ]:
        try:
            el = page.locator(sel).first
            if await el.is_visible():
                await el.click()
                await page.wait_for_timeout(2000)
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Main scrape flow
# ---------------------------------------------------------------------------

async def scrape_all() -> list[dict]:
    """Full scrape: catalog list → individual detail pages."""
    all_items: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        list_page = await context.new_page()

        # ------------------------------------------------------------------
        # Step 1: Load catalog and apply "Individual Test Solutions" filter
        # ------------------------------------------------------------------
        log.info(f"Loading catalog: {CATALOG_URL}")
        await list_page.goto(CATALOG_URL, timeout=30_000, wait_until="networkidle")
        await list_page.wait_for_timeout(2000)

        # Try to click the "Individual Test Solutions" tab/filter
        for sel in [
            "text=Individual Test Solutions",
            "[data-filter='individual']",
            "button:has-text('Individual')",
            "a:has-text('Individual Test')",
            "[class*='tab']:has-text('Individual')",
        ]:
            try:
                el = list_page.locator(sel).first
                if await el.is_visible(timeout=3000):
                    await el.click()
                    await list_page.wait_for_timeout(2000)
                    log.info(f"Clicked 'Individual Test Solutions' filter via: {sel}")
                    break
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Step 2: Paginate through all list pages
        # ------------------------------------------------------------------
        page_num = 1
        seen_urls: set[str] = set()

        while True:
            log.info(f"Scraping list page {page_num}...")
            items = await extract_items_from_page(list_page)
            new_items = [it for it in items if it["url"] not in seen_urls]
            for it in new_items:
                seen_urls.add(it["url"])
            all_items.extend(new_items)
            log.info(f"  → {len(new_items)} new items (total so far: {len(all_items)})")

            if not await has_next_page(list_page):
                log.info("No more pages.")
                break

            if not await click_next_page(list_page):
                log.info("Could not click next page — stopping.")
                break

            page_num += 1
            if page_num > 50:  # safety cap
                log.warning("Hit 50-page safety cap.")
                break

        # ------------------------------------------------------------------
        # Step 3: Fetch individual product pages for descriptions
        # ------------------------------------------------------------------
        log.info(f"\nFetching detail pages for {len(all_items)} assessments...")
        detail_page = await context.new_page()

        for idx, item in enumerate(all_items):
            log.info(f"  [{idx+1}/{len(all_items)}] {item['name']}")
            details = await scrape_product_page(detail_page, item["url"])
            item.update(details)

        await detail_page.close()
        await list_page.close()
        await browser.close()

    return all_items


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("SHL Catalog Scraper — Individual Test Solutions")
    log.info("=" * 60)

    items = await scrape_all()

    if not items:
        log.error("No items scraped! Check selectors or page structure.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    log.info(f"\n✅ Saved {len(items)} assessments to {OUTPUT_PATH}")

    # Quick sanity check
    types = set()
    for it in items:
        for code in it.get("test_type", "").split(","):
            if code.strip():
                types.add(code.strip())
    log.info(f"Test types found: {sorted(types)}")


if __name__ == "__main__":
    asyncio.run(main())
