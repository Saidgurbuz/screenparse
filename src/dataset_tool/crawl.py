"""Crawling orchestration with optimized multiprocessing support."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import csv
import math
from tqdm import tqdm
from .config import Config
from .utils import (
    ensure_dir,
    get_processed_url_stems,
    canonicalize_url,
    safe_stem_from_url,
)


def _load_urls(path: str) -> List[str]:
    """Load URLs from CSV file (one per line)."""
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() and not row[0].strip().startswith("#"):
                urls.append(row[0].strip())
    return urls


def _chunk_urls(urls: List[str], num_chunks: int) -> List[List[str]]:
    """Split URLs into roughly equal chunks for worker processes."""
    chunk_size = math.ceil(len(urls) / num_chunks)
    return [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]


def crawl(
    urls_file: str,
    out_dir: str = "data/raw",
    do_ocr: bool = False,
    headless: bool = True,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    Crawl URLs and collect annotations with optimized parallelization.

    Key improvements:
    - Each worker process maintains a persistent browser
    - URLs are batched to minimize inter-process communication
    - Scales efficiently to 32-256 CPU cores

    Args:
        urls_file: Path to CSV file with URLs
        out_dir: Output directory for raw data
        do_ocr: Enable OCR processing
        headless: Run browser in headless mode
        workers: Number of parallel worker processes

    Returns:
        List of collected records
    """
    ensure_dir(out_dir)
    urls = _load_urls(urls_file)

    if not urls:
        print(f"No URLs found in {urls_file}")
        return []

    print(f"Loaded {len(urls)} URLs")

    # Check for already-processed URLs using stem matching
    processed_stems = get_processed_url_stems(out_dir)

    if processed_stems:
        print(f"Found {len(processed_stems)} already-processed URLs in {out_dir}")

        # Filter out already-processed URLs
        urls_to_process = []
        skipped_count = 0

        for url in urls:
            url_canonical = canonicalize_url(url)
            url_stem = safe_stem_from_url(url_canonical)

            if url_stem in processed_stems:
                skipped_count += 1
            else:
                urls_to_process.append(url)

        print(f"Skipping {skipped_count} already-processed URLs")
        print(f"Remaining: {len(urls_to_process)} URLs to process")

        urls = urls_to_process

        if not urls:
            print("All URLs have already been processed!")
            return []

    cfg = Config(
        out_dir=out_dir,
        do_ocr=do_ocr,
        headless=headless,
    )

    results = []
    errors = []

    if workers == 1:
        # Sequential execution with single persistent browser
        print("Using sequential execution (single browser)")
        from .worker import BrowserWorker

        with BrowserWorker(worker_id=0, cfg=cfg) as worker:
            for url in tqdm(urls, desc="Crawling"):
                url_str, result, error = worker.process_url(url)
                if error:
                    print(f"\n✗ {url_str}: {error.split(chr(10))[0][:100]}")
                    errors.append((url_str, error))
                else:
                    results.append(result)

    else:
        # Parallel execution with worker pool
        print(f"Using {workers} worker processes (each with persistent browser)")

        # Split URLs into batches for workers
        url_batches = _chunk_urls(urls, workers)
        actual_workers = len(url_batches)
        print(f"URL batches: {[len(batch) for batch in url_batches]}")

        # Import worker function
        from .worker import worker_process_urls

        # Process in parallel
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            mp_context=None,  # Use default (usually 'spawn' on Linux/Mac, safest for Playwright)
        ) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(worker_process_urls, worker_id, batch, cfg): (
                    worker_id,
                    batch,
                )
                for worker_id, batch in tqdm(
                    enumerate(url_batches),
                    desc="Submitting batches",
                    total=len(url_batches),
                )
            }

            # Collect results with progress tracking
            with tqdm(total=len(urls), desc="Crawling") as pbar:
                for future in tqdm(
                    as_completed(future_to_batch),
                    desc="Processing",
                    total=len(future_to_batch),
                ):
                    worker_id, batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        for url_str, result, error in batch_results:
                            if error:
                                errors.append((url_str, error))
                            else:
                                results.append(result)
                            pbar.update(1)
                    except Exception as e:
                        # Worker process crashed
                        print(f"\n✗ Worker {worker_id} crashed: {e}")
                        for url in batch:
                            errors.append((url, f"Worker crashed: {e}"))
                        pbar.update(len(batch))

    print(f"\nCompleted: {len(results)} successful, {len(errors)} failed")

    if errors:
        print("\nFailed URLs:")
        for url, error in errors[:10]:
            error_lines = str(error).split("\n")
            print(f"  - {url}: {error_lines[0][:150]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return results
