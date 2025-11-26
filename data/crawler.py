import os
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

DOWNLOAD_DIR = "data/gutenberg_children"
LOG_FILE = os.path.join(DOWNLOAD_DIR, "log.txt")
BASE_URL = "https://www.gutenberg.org"
CATEGORY_URL = "https://www.gutenberg.org/ebooks/bookshelf/636"  # Children and Young Adults

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def log(message):
    """Write message to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    # with open(LOG_FILE, "a", encoding="utf-8") as f:
    #     f.write(line + "\n")

def main():
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        log("Opening category page...")
        page.goto(CATEGORY_URL)
        time.sleep(3)

        all_book_urls = set()
        seen = set()

        start = 1              
        per_page = 25
        
        while len(all_book_urls)<3000:
            if start == 1:
                url = CATEGORY_URL
            else:
                url = f"{CATEGORY_URL}?start_index={start}"

            page.goto(url)
            page.wait_for_load_state("networkidle")
            time.sleep(0.4)  
            links = page.query_selector_all("a[href^='/ebooks/']")
            found_this_page = 0

            for l in links:
                href = l.get_attribute("href")
                if not href:
                    continue
                parts = href.split("/")
                if len(parts) == 3 and parts[-1].isdigit():
                    full = BASE_URL + href
                    if full not in seen:
                        seen.add(full)
                        all_book_urls.add(full)
                        found_this_page += 1

            log(f"Found {found_this_page} new book links on page start_index={start} (total {len(all_book_urls)})")

            # stop when this page had no new canonical book links (end of listing)
            if found_this_page == 0:
                log("No new books found on this page — assuming end of listing.")
                break

            # prepare next page
            start += per_page

        # all_book_urls now contains the deduplicated list across pages
        log(f"Collected {len(all_book_urls)} book URLs across pages.")
        # log(f"Collected URLs:\n" + "\n".join(book_urls))

        for i, url in enumerate(sorted(all_book_urls), start=1):
            try:
                page.goto(url)
                time.sleep(2)

                # Find UTF-8 Plain Text link
                link = page.query_selector('a[href*="utf8.txt"]') or page.query_selector('a:has-text("Plain Text UTF-8")')

                link = (
                    page.query_selector("a:has-text('Plain Text UTF-8')") or
                    page.query_selector("a:has-text('Plain Text')") or
                    page.query_selector("a[href$='.txt']") or
                    page.query_selector("a[href*='txt']")
                )

                if not link:
                    log(f"No text-format link found ({i}): {url}")
                    continue

                with page.expect_navigation():
                    link.click()

                new_url = page.url

                if new_url.endswith(".txt") or ".txt" in new_url:
                    text_content = page.content()   

                    try:
                        text_content = page.inner_text("body")
                    except:
                        text_content = page.content()

                    content_size_kb = len(text_content.encode("utf-8")) / 1024

                    if content_size_kb < 50:
                        print("Skipped (too small)")
                        continue


                    filename = os.path.join(DOWNLOAD_DIR, f"book_{i}.txt")
                    with open(filename, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(text_content)

                    log(f"Downloaded (direct text) {filename}")
                    continue

                try:
                    with page.expect_download(timeout=5000) as dl_info:
                        link.click()
                    dl = dl_info.value
                    filename = os.path.join(DOWNLOAD_DIR, f"book_{i}.txt")
                    dl.save_as(filename)
                    log(f"Downloaded (file download) {filename}")
                except:
                    log(f"Could not download book {i}, no direct .txt or download: {url}")

            except Exception as e:
                log(f"Error on book {i} ({url}): {e}")

        browser.close()
        log("Download process completed.")

if __name__ == "__main__":
    main()

