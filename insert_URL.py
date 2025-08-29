"""
insert_URL.py (verbose + fast dedup)
------------------------------------
Read a .txt file containing URLs (1 per line), fetch EXACTLY those URLs,
extract text/markdown, chunk the content, de-duplicate, and insert into ChromaDB.

Speed & reliability:
- Parallel downloads for PDFs/DOCs/TXTs (--max-parallel, default 8)
- Browser-like headers; optional --cookies cookies.txt for gated files
- Optional OCR for image-only PDFs (--enable-ocr)  [pdf2image + pytesseract]
- **FAST DEDUP**: batched collection.get(ids=[...]) + big batch inserts
- Immediate, flushed progress logs

Examples:
  python insert_URL.py URLs.txt --collection docs --db-dir ./chroma_db --log
  python insert_URL.py URLs.txt --cookies cookies.txt --max-parallel 12 --timeout 15 --enable-ocr
"""

import argparse
import sys
import os
import re
import asyncio
import shutil
import tempfile
import mimetypes
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
import http.cookiejar as cookiejar
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Crawl4AI for HTML pages
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from utils import (
    get_chroma_client,
    get_or_create_collection,
    add_documents_to_collection,
    compute_hash,
    document_exists,            # still available, but not used in the fast path
    log_collection_contents,
    chunk_text
)

try:
    from agents.modules.loaders.local_loader import load_supported_file
except Exception:
    load_supported_file = None  # handled at runtime


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://community.vantiq.com/",
    "Connection": "keep-alive",
}


# ---------------- Utility logging ----------------

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- Core helpers ----------------

def read_url_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def guess_ext_from_url(u: str) -> str:
    parsed = urlparse(u)
    path = parsed.path.lower()
    if "." in path:
        return os.path.splitext(path)[1]
    return ""


def smart_chunk_markdown(markdown: str, max_len: int = 1000) -> List[str]:
    """Split Markdown by header hierarchy and max_len."""
    def split_by_header(md, header_pattern):
        indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
        indices.append(len(md))
        return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]

    chunks = []
    for h1 in split_by_header(markdown, r'^# .+$'):
        if len(h1) > max_len:
            for h2 in split_by_header(h1, r'^## .+$'):
                if len(h2) > max_len:
                    for h3 in split_by_header(h2, r'^### .+$'):
                        if len(h3) > max_len:
                            for i in range(0, len(h3), max_len):
                                chunks.append(h3[i:i+max_len].strip())
                        else:
                            chunks.append(h3)
                else:
                    chunks.append(h2)
        else:
            chunks.append(h1)

    final_chunks = []
    for c in chunks:
        if len(c) > max_len:
            final_chunks.extend([c[i:i+max_len].strip() for i in range(0, len(c), max_len)])
        else:
            final_chunks.append(c)

    return [c for c in final_chunks if c]


async def crawl_plain_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Use Crawl4AI for HTML/markdown pages only (no recursion)."""
    if not urls:
        return []
    log(f"Crawling {len(urls)} HTML pages with headless browser...")
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig()
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=urls, config=crawl_config)
    out = []
    for r in results:
        if getattr(r, "success", False) and getattr(r, "markdown", None):
            out.append({"url": r.url, "text": r.markdown, "format": "markdown"})
        else:
            log(f"[warn] HTML crawl failed or empty: {getattr(r, 'url', 'unknown')}")
    return out


def load_cookies(cookies_path: str | None) -> cookiejar.MozillaCookieJar | None:
    if not cookies_path:
        return None
    cj = cookiejar.MozillaCookieJar()
    try:
        cj.load(cookies_path, ignore_discard=True, ignore_expires=True)
        return cj
    except Exception as e:
        log(f"[warn] Failed to load cookies from {cookies_path}: {e}")
        return None


def download_to_temp(url: str, timeout: Tuple[int, int], cookies: cookiejar.MozillaCookieJar | None) -> Tuple[str, str]:
    """Download a remote file to a temp path. Returns (filepath, content_type)."""
    session = requests.Session()
    if cookies:
        session.cookies = cookies
    resp = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True, allow_redirects=True)
    # quick-fail codes
    if resp.status_code in (403, 404):
        resp.close()
        raise requests.HTTPError(f"{resp.status_code} for {url}")
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    ext = guess_ext_from_url(url)
    if not ext:
        ext = mimetypes.guess_extension((ctype or "").split(";")[0].strip()) or ""
    fd, tmp_path = tempfile.mkstemp(suffix=ext or ".bin")
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp_path, ctype


def _ocr_pdf_local(tmp_pdf_path: str) -> str:
    """Try OCR on a local PDF using pdf2image + pytesseract. Returns text or ''."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return ""
    text_parts: List[str] = []
    try:
        images = convert_from_path(tmp_pdf_path, dpi=200)  # requires poppler
        for img in images:
            try:
                txt = pytesseract.image_to_string(img)  # requires tesseract binary
                if txt and txt.strip():
                    text_parts.append(txt)
            except Exception:
                continue
    except Exception:
        return ""
    return "\n".join(text_parts).strip()


def extract_text_from_remote_file(url: str, enable_ocr: bool, timeout: Tuple[int, int], cookies: cookiejar.MozillaCookieJar | None) -> Tuple[str, str]:
    """
    Download file and extract text using local_loader (PDF/DOCX/PPTX/etc).
    If loader returns empty for PDFs and enable_ocr=True, attempt OCR fallback.
    Returns (text, source_url). If extraction fails, returns ("", url).
    """
    if load_supported_file is None:
        return "", url
    try:
        tmp_path, ctype = download_to_temp(url, timeout=timeout, cookies=cookies)
    except Exception as e:
        log(f"[warn] Download failed: {url} :: {e}")
        return "", url

    text = ""
    try:
        text = load_supported_file(Path(tmp_path))
        if enable_ocr and (tmp_path.lower().endswith('.pdf') or ('pdf' in (ctype or '').lower())):
            if not text or not text.strip():
                ocr_text = _ocr_pdf_local(tmp_path)
                if ocr_text:
                    text = ocr_text
    except Exception as e:
        log(f"[warn] Parse failed: {url} :: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return (text or ""), url


def classify_url_for_strategy(url: str) -> str:
    ext = (guess_ext_from_url(url) or "").lower()
    if ext in (".pdf", ".docx", ".pptx", ".xlsx"):
        return "binary"
    if ext in (".txt", ".md"):
        return "textlike"
    return "html"


def process_extracted_text(text: str, source_url: str, chunk_size: int) -> List[Tuple[str, Dict[str, Any]]]:
    if not text or not text.strip():
        return []
    if re.search(r'(?m)^#{1,6}\s+\S', text):
        chunks = smart_chunk_markdown(text, max_len=chunk_size)
    else:
        chunks = chunk_text(text)
        wrapped = []
        for c in chunks:
            if len(c) > chunk_size:
                for i in range(0, len(c), chunk_size):
                    wrapped.append(c[i:i+chunk_size])
            else:
                wrapped.append(c)
        chunks = wrapped
    out = []
    for idx, c in enumerate(chunks):
        c = (c or "").strip()
        if not c:
            continue
        meta = {"chunk_index": idx, "source": source_url}
        out.append((c, meta))
    return out


def process_binary_urls_parallel(urls: List[str], cookies: cookiejar.MozillaCookieJar | None, enable_ocr: bool, max_parallel: int, timeout_connect: int, timeout_read: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not urls:
        return results

    log(f"Downloading {len(urls)} binary/text files with {max_parallel} workers...")

    def worker(u: str) -> Tuple[str, str, str]:
        txt, src = extract_text_from_remote_file(u, enable_ocr, timeout=(timeout_connect, timeout_read), cookies=cookies)
        return (src, txt, "text")

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(worker, u): u for u in urls}
        completed = 0
        total = len(futs)
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                src, txt, fmt = fut.result()
                if txt and txt.strip():
                    results.append({"url": src, "text": txt, "format": fmt})
                else:
                    log(f"[warn] No extractable text: {u}")
            except Exception as e:
                log(f"[warn] Binary/text task failed: {u} :: {e}")
            finally:
                completed += 1
                if completed % 5 == 0 or completed == total:
                    log(f"Progress: {completed}/{total} binary/text files done")

    return results


def batched(iterable, n):
    it = list(iterable)
    for i in range(0, len(it), n):
        yield it[i:i+n]


def main():
    parser = argparse.ArgumentParser(description="Insert EXACT URLs (from a .txt) into ChromaDB (verbose + fast dedup)")
    parser.add_argument("url_file", help="Path to .txt file containing URLs (1 per line)")
    parser.add_argument("--collection", default="docs", help="ChromaDB collection name")
    parser.add_argument("--db-dir", default="./chroma_db", help="ChromaDB directory")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--batch-size", type=int, default=100, help="ChromaDB insert batch size")
    parser.add_argument("--clear-db", action="store_true", help="Clear ChromaDB before inserting")
    parser.add_argument("--log", action="store_true", help="Log collection contents after insertion")
    parser.add_argument("--enable-ocr", action="store_true", help="Try OCR for image-only PDFs (requires pdf2image & pytesseract; plus poppler & tesseract binaries)")
    parser.add_argument("--cookies", type=str, default=None, help="Path to Netscape/Mozilla cookies.txt for authenticated downloads")
    parser.add_argument("--max-parallel", type=int, default=8, help="Max parallel workers for binary/text downloads")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout seconds (applies to connect and read)")
    args = parser.parse_args()

    log("=== insert_URL.py start ===")

    if not os.path.exists(args.url_file):
        log(f"Error: file not found: {args.url_file}")
        sys.exit(1)

    urls = read_url_file(args.url_file)
    if not urls:
        log("No URLs found in file.")
        sys.exit(0)

    # Partition URLs by type
    buckets = {"binary": [], "textlike": [], "html": []}
    for u in urls:
        buckets[classify_url_for_strategy(u)].append(u)
    log(f"Loaded {len(urls)} URLs  |  binary: {len(buckets['binary'])}  textlike: {len(buckets['textlike'])}  html: {len(buckets['html'])}")

    # DB prep
    if args.clear_db and os.path.exists(args.db_dir):
        log(f"[info] Clearing ChromaDB directory: {args.db_dir}")
        shutil.rmtree(args.db_dir)

    client = get_chroma_client(args.db_dir)
    collection = get_or_create_collection(client, args.collection, embedding_model_name=args.embedding_model)
    log(f"Connected to ChromaDB at '{args.db_dir}', collection '{args.collection}'")

    # Cookies
    cookies = load_cookies(args.cookies)
    if args.cookies:
        if cookies:
            log(f"Loaded cookies from {args.cookies}")
        else:
            log("[warn] Cookies file provided but could not be loaded; proceeding without cookies")

    # Extract
    bin_like = buckets["binary"] + buckets["textlike"]
    extracted: List[Dict[str, Any]] = process_binary_urls_parallel(
        bin_like,
        cookies=cookies,
        enable_ocr=args.enable_ocr,
        max_parallel=args.max_parallel,
        timeout_connect=args.timeout,
        timeout_read=args.timeout,
    )
    if buckets["html"]:
        html_results = asyncio.run(crawl_plain_urls(buckets["html"]))
        extracted.extend(html_results)

    log(f"Extraction complete. {len(extracted)} items produced. Building chunks & dedup...")

    # ---------- FAST DEDUP & BATCH INSERT ----------
    log("Indexing chunks and computing hashes...")
    all_rows = []   # (id_hash, chunk_text, metadata_dict)
    for item in extracted:
        src = item["url"]
        txt = item["text"]
        for (chunk, meta) in process_extracted_text(txt, src, args.chunk_size):
            h = compute_hash(chunk)     # ID = hash of chunk text
            meta["doc_hash"] = h
            all_rows.append((h, chunk, meta))

    if not all_rows:
        log("No chunks produced; nothing to insert.")
        sys.exit(0)

    log(f"Computed {len(all_rows)} chunk hashes. Checking for existing IDs in batches...")

    # Unique lookup to avoid DuplicateIDError
    all_ids = [h for (h, _, _) in all_rows]
    unique_ids = list(dict.fromkeys(all_ids))

    existing_ids = set()
    BATCH_LOOKUP = 1000
    for batch in batched(unique_ids, BATCH_LOOKUP):
        got = collection.get(ids=batch)
        existing_ids.update(got.get("ids", []))

    log(f"Existing IDs in DB: {len(existing_ids)}")

    # Keep only new rows; avoid inserting the same new ID twice
    to_add_ids, to_add_docs, to_add_metas = [], [], []
    seen_new = set()
    for h, chunk, meta in all_rows:
        if h in existing_ids or h in seen_new:
            continue
        seen_new.add(h)
        to_add_ids.append(h)
        to_add_docs.append(chunk)
        to_add_metas.append(meta)

    if not to_add_ids:
        log("All chunks already exist. No new documents to insert.")
        sys.exit(0)

    log(f"Preparing to insert {len(to_add_ids)} new chunks...")

    # Insert in big batches (single add() per batch)
    BATCH_INSERT = max(args.batch_size, 500)
    for ids_batch, docs_batch, metas_batch in zip(
            batched(to_add_ids, BATCH_INSERT),
            batched(to_add_docs, BATCH_INSERT),
            batched(to_add_metas, BATCH_INSERT)):
        add_documents_to_collection(
            collection,
            list(docs_batch),
            list(metas_batch),
            list(ids_batch),
            batch_size=0,  # single add() call for this prepared batch
        )

    log(f"Successfully added {len(to_add_ids)} chunks to ChromaDB collection '{args.collection}'.")

    if args.log:
        log(f"[log] Previewing content from ChromaDB collection '{args.collection}':")
        log_collection_contents(collection)

    log("=== insert_URL.py done ===")


if __name__ == "__main__":
    main()