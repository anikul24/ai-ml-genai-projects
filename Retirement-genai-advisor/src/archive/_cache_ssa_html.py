# src/cache_ssa_html.py
from pathlib import Path
import requests

from ingest_ssa import (
    get_internal_handbook_links,
    cache_html_if_needed,
    deep_collect_handbook_urls,
)

CACHE_DIR = Path("data/raw/ssa/html_cache")

#CACHE_DIR = Path("data/raw/ssa/html_cache")
#CACHE_DIR.mkdir(parents=True, exist_ok=True)

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "Upgrade-Insecure-Requests": "1"
}

# Global Session for persistent cookies/connection
GLOBAL_SESSION = requests.Session()
GLOBAL_SESSION.headers.update(BROWSER_HEADERS) # Apply headers once

def url_to_cache_path(url: str, cache_dir: Path = CACHE_DIR) -> Path:
    key = url.split("//", 1)[-1].replace("/", "_")
    return cache_dir / f"{key}.html"

def cache_html_if_needed(url: str, cache_dir: Path = CACHE_DIR, throttle: float = 1.0):
    """
    If HTML for this URL is not cached, download and save it.
    Returns the cache path (or None on failure).
    """
    path = url_to_cache_path(url, cache_dir)
    if path.exists():
        return path

    print("Downloading:", url)
    try:
        #resp = requests.get(url, headers=BROWSER_HEADERS, timeout=15)
        resp = GLOBAL_SESSION.get(url, timeout=15)
        resp.raise_for_status()
        if resp.status_code == 200:
            path.write_text(resp.text, encoding="utf-8")
            import time; time.sleep(throttle)
            return path
        else:
            print("Non-200 status for", url, "status:", resp.status_code)
            return None
    except Exception as e:
        print("Error downloading", url, "->", e)
        return None

def deep_cache_handbook_pages(start_urls, max_depth: int = 2, throttle: float = 1.0):
    """
    Recursively:
    - cache HTML for start_urls (if not already),
    - parse internal handbook links from each page,
    - cache those as well, up to max_depth.

    Works only with /OP_Home/handbook/ links.
    """
    visited = set()
    frontier = list(start_urls)
    all_urls = set(start_urls)

    for depth in range(max_depth):
        print(f"[Depth {depth}] frontier size = {len(frontier)}")
        next_frontier = []
        for url in frontier:
            if url in visited:
                continue
            visited.add(url)

            # 1) ensure this page is cached
            cache_path = cache_html_if_needed(url, CACHE_DIR, throttle=throttle)
            if cache_path is None:
                continue

            # 2) read cached HTML
            html = cache_path.read_text(encoding="utf-8")

            # 3) extract internal handbook links
            child_urls = get_internal_handbook_links(html, base_url=url)
            for cu in child_urls:
                if cu not in all_urls:
                    all_urls.add(cu)
                    next_frontier.append(cu)

        frontier = next_frontier

    print(f"Deep cache finished. Total unique handbook URLs seen: {len(all_urls)}")
    return sorted(all_urls)


def seed_from_local_toc(local_toc_path: Path, base_url: str):
    html = local_toc_path.read_text(encoding="utf-8")
    return get_internal_handbook_links(html, base_url=base_url)




def main():
    # manually downloaded top-level TOC
    local_toc = CACHE_DIR / "SSA Handbook Table of Contents.html"
    base_url = "https://www.ssa.gov/OP_Home/handbook/handbook-toc.html"

    first_level_urls = seed_from_local_toc(local_toc, base_url)
    print("First-level URLs from local TOC:", len(first_level_urls))

    all_urls = deep_cache_handbook_pages(first_level_urls, max_depth=2, throttle=1.0)
    print("Total cached or attempted URLs:", len(all_urls))

if __name__ == "__main__":
    main()
