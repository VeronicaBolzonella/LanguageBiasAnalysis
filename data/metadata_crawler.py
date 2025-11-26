import os
import time
from datetime import datetime
from playwright.sync_api import sync_playwright
import pandas as pd

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
        browser = p.firefox.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        log("Opening category page...")
        page.goto(CATEGORY_URL)
        time.sleep(3)

        all_book_urls = set()
        seen = set()

        start = 1              
        per_page = 25
        
        while len(all_book_urls)<2:
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
        d = {'id': [], 'author': [], 'year': []}
        df = pd.DataFrame(data=d)

        ids = [i for i in range(len(all_book_urls))]

        for i, url in enumerate(sorted(all_book_urls), start=1):
            try:
                page.goto(url)
                time.sleep(2)

                # Gather metadata
                author_el = page.query_selector('tr:has(th:text("Author")) td')
                author = author_el.inner_text().strip() if author_el else None

                df.loc[len(df)] = [i, author, None]


                
            except Exception as e:
                log(f"Error on book {i} ({url}): {e}")

        browser.close()
        log("Complete")

if __name__ == "__main__":
    main()

