import os
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

DOWNLOAD_DIR = "gutenberg_children"
LOG_FILE = os.path.join(DOWNLOAD_DIR, "log.txt")
BASE_URL = "https://www.gutenberg.org"
CATEGORY_URL = "https://www.gutenberg.org/ebooks/bookshelf/26"  # Children and Young Adults

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def log(message):
    """Write message to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        log("Opening category page...")
        page.goto(CATEGORY_URL)
        time.sleep(3)

        links = page.query_selector_all("li.booklink a.link")
        book_urls = [BASE_URL + l.get_attribute("href") for l in links if l.get_attribute("href")]
        log(f"Found {len(book_urls)} book URLs.")

        for i, url in enumerate(book_urls, start=1):
            try:
                page.goto(url)
                time.sleep(2)

                # Optional: Skip non-English books
                lang_label = page.query_selector("tr:has(td:has-text('Language')) td a")
                if lang_label and lang_label.inner_text().strip().lower() != "english":
                    log(f"Skipped non-English book ({i}): {url}")
                    continue

                # Find UTF-8 Plain Text link
                link = page.query_selector('a[href*="utf8.txt"]') or page.query_selector('a:has-text("Plain Text UTF-8")')

                if link:
                    with page.expect_download() as download_info:
                        link.click()
                    download = download_info.value
                    filename = os.path.join(DOWNLOAD_DIR, f"book_{i}.txt")
                    download.save_as(filename)
                    log(f"Downloaded {filename}")
                else:
                    log(f"No UTF-8 text link found ({i}): {url}")

            except Exception as e:
                log(f"Error on book {i} ({url}): {e}")

        browser.close()
        log("Download process completed.")

if __name__ == "__main__":
    main()
