import os
import time
from datetime import datetime
from playwright.sync_api import sync_playwright
import pandas as pd
import re

METADATA_DIR = "data/gutenberg_children"
LOG_FILE = os.path.join(METADATA_DIR, "log.txt")
BASE_URL = "https://www.gutenberg.org"
CATEGORY_URL = "https://www.gutenberg.org/ebooks/bookshelf/636"  # Children and Young Adults

os.makedirs(METADATA_DIR, exist_ok=True)

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
        d = {'id': [], 'author': [], 'year': []}
        df = pd.DataFrame(data=d)

        ids = [i for i in range(len(all_book_urls))]

        for i, url in enumerate(sorted(all_book_urls), start=1):
            author = None
            year = None
            try:
                page.goto(url)
                time.sleep(2)

                # Gather metadata
                author_el = page.query_selector('tr:has(th:text("Author")) td')
                author = author_el.inner_text().strip() if author_el else None
                parts = author.split(",")[:2]   
                author = ", ".join(parts)  # join back if needed
                print(author)

                wiki_el = page.query_selector('a[href*="wikipedia.org"]')
                year = None
                if wiki_el:
                    wiki_href = wiki_el.get_attribute("href")
                    if wiki_href:
                        # Go to Wikipedia page
                        page.goto(wiki_href)
                        page.wait_for_load_state("networkidle")
                        time.sleep(1)

                        # Try to extract publication date from infobox
                        pub_el = page.query_selector('table.infobox th:has-text("Publication date") + td')
                        if pub_el:
                            pub_text = pub_el.inner_text().strip()
                            # Extract year if possible
                            match = re.search(r"\b(\d{4})\b", pub_text)
                            if match:
                                year = int(match.group(1))
                            print(year)

                df.loc[len(df)] = [i, author, year]


                
            except Exception as e:
                log(f"Error on book {i} ({url}): {e}")
            finally:
                # Always add a row, even if author/year is None
                df.loc[len(df)] = [i, author, year]

        browser.close()
        log("Complete")

        print(df)
        json_file = os.path.join(METADATA_DIR, "books_metadata.json")
        df.to_json(json_file, orient="records", force_ascii=False, indent=4)
        log(f"Data saved to {json_file}")

if __name__ == "__main__":
    main()

