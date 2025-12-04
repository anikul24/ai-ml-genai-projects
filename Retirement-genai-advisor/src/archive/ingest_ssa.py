"""
Ingest SSA Handbook using LangChain's UnstructuredURLLoader.
Saves HTML cache, extracts text, chunks, embeds and upserts to Pinecone.

Environment variables :
    OPENAI_API_KEY
    PINECONE_API_KEY
    VECTOR_INDEX

Notes:
- This script uses UnstructuredURLLoader for convenience; it still respects polite crawling (throttling).

"""

import os
import time, re
import json
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma ##for chromadb
from pinecone import Pinecone
from pinecone import ServerlessSpec



# CONFIG
BASE_TOC = "https://www.ssa.gov/OP_Home/handbook/handbook-toc.html"
DATA_DIR = Path("data/raw/ssa")
CACHE_DIR = DATA_DIR / "html_cache"
MANIFEST = DATA_DIR / "manifest.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Local uploaded notebook path (from your earlier upload)
LOCAL_NOTEBOOK_PATH = Path("/mnt/data/research_agent_langgraph.ipynb")

load_dotenv(dotenv_path="./cred.env")


# Pinecone config from env
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

VECTOR_DB_TYPE = 'CHROMA' # 'PINECONE'
PERSIST_DIRECTORY = "./chroma_data"
VECTOR_INDEX = os.environ.get("VECTOR_INDEX", "retirement-ssa-index")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Embedding model name
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
# Throttle between requests
THROTTLE_SECONDS = float(os.environ.get("CRAWL_THROTTLE", "1.0"))


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com/"
}

# place near top of src/ingest_ssa.py
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests, time, re

SSA_ROBOTS_URL = "https://www.ssa.gov/robots.txt"
USER_AGENT = "RetireAI-Ingest/0.1"

## This is giving Error - requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://www.ssa.gov/OP_Home/handbook/handbook-toc.html
# def fetch_toc_links(toc_url=BASE_TOC):
#     """Scrape the SSA Handbook TOC for internal links to handbook pages."""
#     print("Fetching TOC:", toc_url)
#     resp = requests.get(toc_url, headers={"User-Agent": "RetireAI-Ingest/0.1"})
#     resp.raise_for_status()
#     soup = BeautifulSoup(resp.text, "html.parser")

#     links = []
#     # Heuristic: select links that contain '/OP_Home/handbook/' in href
#     for a in soup.find_all("a", href=True):
#         print(f'Found link: {a["href"]}')
#         href = a["href"]
#         if "/OP_Home/handbook/" in href:
#             full = urljoin(toc_url, href)
#             title = a.get_text(strip=True)
#             links.append({"title": title, "url": full})

#     # Deduplicate preserving order
#     seen = set()
#     out = []
#     for item in links:
#         if item["url"] not in seen:
#             seen.add(item["url"])
#             out.append(item)
#     print(f"Found {len(out)} candidate links in TOC (after dedup)")
#     return out


def get_internal_handbook_links(html: str, base_url: str):
    """
    Generic helper: given ANY SSA handbook-related HTML page,
    return a list of absolute URLs that point to internal handbook pages.

    We only keep links that contain '/OP_Home/handbook/' and are on ssa.gov.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Only keep internal SSA handbook paths
        if "/OP_Home/handbook/" in href:
            full = urljoin(base_url, href)
            urls.append(full)

    # de-duplicate preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


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
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=15)
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

def deep_collect_handbook_urls(start_urls, max_depth=2, cache_dir=CACHE_DIR):
    """
    Breadth-first crawl within local cached handbook pages.

    - start_urls: list of absolute URLs (from the top-level TOC)
    - max_depth: how many link-levels to follow
    - Only uses local cached HTML (no requests).
    - Only returns internal SSA handbook URLs (/OP_Home/handbook/).
    """
    visited = set()
    frontier = list(start_urls)
    all_urls = set(start_urls)

    for depth in range(max_depth):
        print(f"Depth {depth}: frontier size = {len(frontier)}")
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

            # get internal links from this page
            child_urls = get_internal_handbook_links(html, base_url=url)
            for cu in child_urls:
                if cu not in all_urls:
                    all_urls.add(cu)
                    next_frontier.append(cu)

        frontier = next_frontier

    print(f"Deep collect finished. Total unique handbook URLs: {len(all_urls)}")
    return sorted(all_urls)


##New code to scrape link with referer 
def fetch_toc_links_robust(toc_url=BASE_TOC, cache_dir=CACHE_DIR, throttle=THROTTLE_SECONDS):
    # 0) If a local cached TOC file exists, use it first (manual fallback)
    """
    Robustly fetch the SSA Handbook Table of Contents HTML page and return a list of dictionaries containing the title and URL of each handbook page.

    First, it checks if a local cached TOC file exists. If so, it uses the cached file.
    Otherwise, it tries to fetch the TOC with robust headers + retries, and optionally caches it.
    If the TOC fetch fails or returns 403, it falls back to parsing the sitemap.xml file for handbook links.
    If the sitemap fallback fails, it returns an empty list.

    :param toc_url: The base URL of the SSA Handbook Table of Contents HTML page.
    :param cache_dir: The directory to cache the local TOC file in.
    :param throttle: The throttle (in seconds) to wait between retries if the TOC fetch fails.
    :return: A list of dictionaries containing the title and URL of each handbook page.
    """

    ## because of 403 error for web scrapping requests, we are using downloaded Html TOC page and referring from there 
    local_cache = cache_dir / "SSA Handbook Table of Contents.html"

    print("fetch_toc_links_robust")

    print("local_cache:", local_cache)
    if local_cache.exists():
        print("Inside if local_cache.exists():")
        print("Using local cached TOC:", local_cache)
        html = local_cache.read_text(encoding="utf-8")
        return _parse_toc_html(html, toc_url)

    # 1) Try fetch with robust headers + retries
    # attempts = 3
    # for attempt in range(1, attempts + 1):
    #     try:
    #         print(f"GET {toc_url} (attempt {attempt})")
    #         resp = requests.get(toc_url, headers=BROWSER_HEADERS, timeout=15)
    #         if resp.status_code == 200:
    #             html = resp.text
    #             # optionally cache it
    #             try:
    #                 cache_dir.mkdir(parents=True, exist_ok=True)
    #                 local_cache.write_text(html, encoding="utf-8")
    #             except Exception:
    #                 pass
    #             return _parse_toc_html(html, toc_url)
    #         else:
    #             print("Non-200 response:", resp.status_code)
    #             if resp.status_code == 403:
    #                 # quick sleep then retry with slightly different headers (mimic browser variability)
    #                 time.sleep(throttle * 1.2)
    #                 BROWSER_HEADERS["Referer"] = toc_url
    #                 continue
    #             else:
    #                 resp.raise_for_status()
    #     except requests.exceptions.RequestException as e:
    #         print("Request error:", e)
    #         time.sleep(throttle)
    #         continue

    # 2) If TOC blocked, fallback to sitemap.xml -> pick handbook links
    # print("TOC fetch failed or returned 403. Falling back to sitemap parsing.")
    # sitemap_url = "https://www.ssa.gov/sitemap.xml"
    # try:
    #     r = requests.get(sitemap_url, headers=BROWSER_HEADERS, timeout=15)
    #     r.raise_for_status()
    #     # parse sitemap for urls containing /OP_Home/handbook/
    #     urls = re.findall(r"<loc>(.*?)</loc>", r.text)
    #     handbook_urls = [u for u in urls if "/OP_Home/handbook/" in u]
    #     print("Found", len(handbook_urls), "handbook URLs in sitemap (fallback).")
    #     items = [{"title": None, "url": u} for u in handbook_urls]
    #     return items
    # except Exception as e:
    #     print("Sitemap fallback failed:", e)

    # 3) final fallback: return empty list (caller should handle)
    return []

def _parse_toc_html(html, base_url):
    """
    Parse the SSA Handbook Table of Contents HTML page and return a list of dictionaries containing the title and URL of each handbook page.

    :param html: The HTML content of the TOC page.
    :param base_url: The base URL to join with relative URLs in the TOC page.
    :return: A list of dictionaries containing the title and URL of each handbook page.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/OP_Home/handbook/" in href:
            full = urljoin(base_url, href)
            title = a.get_text(strip=True)
            links.append({"title": title, "url": full})
    # deduplicate preserving order
    seen = set()
    out = []
    for it in links:
        if it["url"] not in seen:
            seen.add(it["url"])
            out.append(it)
    print(f"Parsed {len(out)} handbook links from TOC HTML")
    return out


def fetch_and_parse_robots(robots_url=SSA_ROBOTS_URL, user_agent=USER_AGENT):
    """Fetch robots.txt and return a RobotFileParser plus crawl_delay (seconds or None)."""
    rp = RobotFileParser()
    try:
        r = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=10)
        r.raise_for_status()
        txt = r.text
    except Exception as e:
        print("Warning: could not fetch robots.txt:", e)
        # If we cannot fetch robots, be conservative: allow only base toc by default
        rp.parse("")   # empty parser -> defaults to allow everything; but we'll be conservative downstream
        return rp, None

    # feed parser
    rp.parse(txt.splitlines())

    # Parse Crawl-delay manually (robotparser doesn't expose it reliably in older Python versions)
    crawl_delay = None
    # look for lines like: Crawl-delay: 1
    for line in txt.splitlines():
        m = re.match(r"(?i)Crawl-delay:\s*([0-9]+(?:\.[0-9]+)?)", line.strip())
        if m:
            crawl_delay = float(m.group(1))
            break

    return rp, crawl_delay

def filter_allowed_urls(urls, rp=None, user_agent=USER_AGENT):
    """
    Given a list of URLs, return only those that are allowed by robots.txt.
    rp = RobotFileParser instance (if None, will fetch automatically).
    """
    if rp is None:
        print("Fetching robots.txt AGAIN if earlier call returned NONE")
        rp, _ = fetch_and_parse_robots()

    allowed = []
    for u in urls:
        # local file: skip robots check (we treat local files as always allowed)
        if u.startswith("file://") or u.startswith("/mnt/") or u.startswith("C:\\"):
            allowed.append(u)
            continue

        # robotparser expectation: path only or full URL is OK
        try:
            if rp.can_fetch(user_agent, u):
                print("robots.txt allows:", u)
                allowed.append(u)
            else:
                print("robots.txt disallows:", u)
        except Exception as e:
            # if parser fails for specific url, be conservative and skip
            print("robots check failed for", u, ":", e)

    return allowed

# Example usage integrated with your TOC fetch function:
def safe_fetch_toc_links(toc_url=BASE_TOC):
    # 1) get list of candidate links 
    raw_links = fetch_toc_links_robust(toc_url)   # returns list of dicts with 'url'
    candidate_urls = [it["url"] for it in raw_links]

    # 2) get robots parser + crawl delay
    rp, crawl_delay = fetch_and_parse_robots()

    # 3) filter allowed urls
    allowed_urls = filter_allowed_urls(candidate_urls, rp)

    print(f"{len(allowed_urls)} URLs allowed by robots.txt out of {len(candidate_urls)} candidates")

    # 4) Respect crawl delay (use crawl_delay if provided; else fallback to THROTTLE_SECONDS)
    effective_throttle = crawl_delay if crawl_delay is not None else THROTTLE_SECONDS
    print("Using throttle (seconds) =", effective_throttle)

    # return list of dicts (title,url) for allowed items, preserving original order
    allowed_items = [it for it in raw_links if it["url"] in set(allowed_urls)]
    return allowed_items, effective_throttle



def cache_html(url, cache_dir=CACHE_DIR):
    """Download & cache HTML for a URL; return local path."""
    key = url.split("//", 1)[-1].replace("/", "_")
    fname = cache_dir / (key + ".html")
    if fname.exists():
        return fname
    print("Downloading:", url)
    r = requests.get(url, headers={"User-Agent": "RetireAI-Ingest/0.1"})
    r.raise_for_status()
    fname.write_text(r.text, encoding="utf-8")
    time.sleep(THROTTLE_SECONDS)
    return fname


def load_with_unstructured(urls):
    """Use LangChain's UnstructuredURLLoader to fetch & parse a list of URLs.
    Returns a list of Document objects.
    """
    print(f"Loading {len(urls)} URLs via UnstructuredURLLoader (may request remote content)")
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    # Ensure metadata includes origin URL
    for i, d in enumerate(docs):
        md = dict(d.metadata or {})
        md.setdefault("origin_url", urls[i])
        md.setdefault("source", "ssa_handbook")
        docs[i] = Document(page_content=d.page_content, metadata=md)
    return docs




def chunk_docs ( docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Splitting {len(docs)} docs into {chunk_size} chunks")
    chunks = splitter.split_documents(docs)
    print(f"Got {len(chunks)} chunks")
    return chunks


def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("Initializing Pinecone (may take a minute or two)")
    test = pc.list_indexes().names()
    print("pc.list_indexes: ",pc.list_indexes().names())
    print(f'VECTOR_INDEX: {VECTOR_INDEX}')

    if VECTOR_INDEX not in pc.list_indexes().names():
        print("Creating Pinecone index:", VECTOR_INDEX)
        pc.create_index(VECTOR_INDEX, dimension=1536, metric="cosine",
         spec =ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        ))
    else:
        print(f"Index {VECTOR_INDEX} already exists" )

    return pc

def init_chromadb():
    print("Initializing ChromaDB ")


    



def embed_and_upsert( chunks, index_name=None,batch_size=50):

    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    if VECTOR_DB_TYPE == "PINECONE":
        index = index_name.Index(VECTOR_INDEX)
        print(f'fetched index {index}')
    else:
        print('create vector_stor for chromadb')
        vector_store = Chroma(
            collection_name=VECTOR_INDEX,
            embedding_function=emb,
            persist_directory=PERSIST_DIRECTORY
        )


    
    records = []

    

    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting Batches"):

        batch = chunks[i:i+batch_size]
        id = [f"ssa-{i+j}" for j in range(len(batch))]
        texts = [doc.page_content for doc in batch]
        vectors = emb.embed_documents(texts)

        batch_record = [(id[j],vectors[j],batch[j].metadata) for j in range(len(batch))]

        records.extend(batch_record)

        if index_name is not None:
            if VECTOR_DB_TYPE == "PINECONE":
                print("Pinecone upsert")
                try:
                    index_name.upsert(vectors=batch_record)
                except Exception as e:
                    print("Pinecone upsert failed:", e)
            else:
                print('chroma upsert')
                try:
                   vector_store.add_documents(documents=batch_record)
                except Exception as e:
                    print("chroma upsert failed:", e)

    ## return record ( multiple chunk tuples) to serialize
    if VECTOR_DB_TYPE == "CHROMA":
        vector_store.persist()
    return records


def save_manifest_json(records, manifest_path=MANIFEST):
    ##here records are multiple tuples
    manifest = [{"id": r[0], "meta": r[2]} for r in records]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote manifest with", len(records), "records to", manifest_path)               
        

def main():
    print("Ingesting SSA Handbook Main function:", BASE_TOC)

    ##testing if cacheing works
    ## NOT working 403 error
    #cache_html(BASE_TOC)
    

    # 1) fetch TOC and links
    toc_items, effective_throttle = safe_fetch_toc_links(BASE_TOC)
    first_level_urls  = [it["url"] for it in toc_items]
    # set the global throttle to effective_throttle or use per-request sleeps
    THROTTLE_SECONDS = effective_throttle

    # 2) recursively collect deeper handbook URLs based on cached HTML
    #    adjust max_depth depending on how deep your cached TOC/section pages go
    all_handbook_urls = deep_collect_handbook_urls(first_level_urls, max_depth=2, cache_dir=CACHE_DIR)

    #  keep the top-level URLs
    urls = all_handbook_urls

    # Optionally limit for dev
    limit = int(os.environ.get("URL_LIMIT", "50"))
    urls = urls[:limit]
    print(f"Using {len(urls)} URLs after deep crawl + limit")

    # 2) Load pages with UnstructuredURLLoader (LangChain)
    docs = load_with_unstructured(urls)

    # # 3) Split into chunks
    chunks = chunk_docs(docs)

    # 4) Init Pinecone and upsert
    idx = None
    if VECTOR_DB_TYPE == "PINECONE":
        print("Initializing Pinecone ")

        if PINECONE_API_KEY:
            try:
                idx = init_pinecone()
            except Exception as e:
                print("Pinecone init failed (will continue without upsert):", e)
                idx = None
        else:
            print("Pinecone API key not set ..exit")
            return
        
    else: 
        print("Initializing Chroma")


    records = embed_and_upsert(chunks, idx)
    save_manifest_json(records)
    print("Done. Total records:", len(records))

    

if __name__ == "__main__":
    main()