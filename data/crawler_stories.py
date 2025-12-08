# playwright_crawl.py
import os
import time
from datetime import datetime
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

DOWNLOAD_DIR = "data/freestories_children"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

BASE = "https://freestoriesforkids.com"
START = "https://freestoriesforkids.com/short-stories"
SLEEP_BETWEEN_REQUESTS = 1.0

def log(msg):
    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] {msg}")

def extract_story_text(page):
    # Try several fallback selectors and always collect <p> texts inside the article
    selectors = [
        "article .field--type-text-with-summary p",
        "article .field--name-body p",
        "article p",
        ".field__item p"
    ]
    parts = []
    for s in selectors:
        nodes = page.locator(s)
        if nodes.count() > 0:
            parts = nodes.all_text_contents()
            break
    # final fallback: all <p> on page
    if not parts:
        parts = page.locator("p").all_text_contents()
    # join paragraphs with blank line
    text = "\n\n".join([p.strip() for p in parts if p.strip()])
    return text

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (compatible; bot/1.0; +https://example.com/bot)",
            viewport={"width":1280, "height":800},
        )
        page = context.new_page()
        page.set_default_timeout(30000)

        log("starting crawl (playwright)")
        current = START
        story_urls = set()

        # Crawl listing pages
        while current:
            log(f"LIST: {current}")
            page.goto(current)
            # wait for main content
            page.wait_for_selector(".view-content, .views-row", timeout=15000)
            time.sleep(0.5)

            # Collect story links within the list area
            anchors = page.locator(".view-content a[href*='/children/'], .views-row a[href*='/children/']")
            for i in range(anchors.count()):
                href = anchors.nth(i).get_attribute("href") or ""
                if href and "/children/" in href:
                    full = urljoin(BASE, href)
                    story_urls.add(full)

            log(f"collected {len(story_urls)} unique story links so far")

            # Find next page link (try rel="next" first, then pager text)
            next_href = None
            try:
                el = page.query_selector('a[rel="next"]')
                if el:
                    next_href = el.get_attribute("href")
            except Exception:
                next_href = None

            if not next_href:
                # fallback: pager link with text Next or ›
                candidate = page.query_selector("a.pager__link:has-text('Next'), a.pager__link:has-text('›'), a:has-text('Next')")
                if candidate:
                    next_href = candidate.get_attribute("href")

            if not next_href:
                log("no next page found, ending listing crawl")
                break

            # resolve and continue
            current = urljoin(BASE, next_href)
            time.sleep(0.5)

        log(f"Total story links found: {len(story_urls)}")

        # Download each story
        for idx, url in enumerate(sorted(story_urls), 1):
            log(f"FETCH {idx}/{len(story_urls)}: {url}")
            try:
                page.goto(url)
                page.wait_for_load_state("networkidle")
                time.sleep(0.5)

                story_text = extract_story_text(page)
                if not story_text or len(story_text) < 200:
                    log(f"skipping (too short) {url}")
                    continue

                slug = url.rstrip("/").split("/")[-1] or f"story_{idx}"
                safe = "".join(c if c.isalnum() else "_" for c in slug)[:50]
                fname = os.path.join(DOWNLOAD_DIR, f"{safe}_{idx}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(story_text)
                log(f"saved -> {fname}")
                time.sleep(SLEEP_BETWEEN_REQUESTS)
            except Exception as e:
                log(f"error fetching {url}: {e}")

        browser.close()
        log("done")

if __name__ == "__main__":
    main()
