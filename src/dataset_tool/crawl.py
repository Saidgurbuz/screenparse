"""Crawling orchestration with multithreading support."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import csv
import traceback
from tqdm import tqdm
from .collector import collect_one
from .config import Config
from .utils import ensure_dir


def _load_urls(path: str) -> List[str]:
    """Load URLs from CSV file (one per line)."""
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() and not row[0].strip().startswith("#"):
                urls.append(row[0].strip())
    return urls


def _collect_one_wrapper(
    url: str, cfg: Config
) -> tuple[str, Optional[Dict[str, Any]], Optional[Exception]]:
    """Wrapper for collect_one that catches exceptions."""
    try:
        result = collect_one(url, cfg)
        return (url, result, None)
    except Exception as e:
        # Capture full traceback for debugging
        tb = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {str(e)}\n{tb}"
        return (url, None, error_msg)


def crawl(
    urls_file: str,
    out_dir: str = "data/raw",
    do_ocr: bool = False,
    headless: bool = True,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    Crawl URLs and collect annotations.

    Args:
        urls_file: Path to CSV file with URLs
        out_dir: Output directory for raw data
        do_ocr: Enable OCR processing
        headless: Run browser in headless mode
        workers: Number of parallel workers (1 = sequential)

    Returns:
        List of collected records
    """
    ensure_dir(out_dir)
    urls = _load_urls(urls_file)

    if not urls:
        print(f"No URLs found in {urls_file}")
        return []

    print(f"Loaded {len(urls)} URLs")
    print(f"Using {workers} worker(s)")

    cfg = Config(
        out_dir=out_dir,
        do_ocr=do_ocr,
        headless=headless,
    )

    results = []
    errors = []

    if workers == 1:
        # Sequential execution
        for url in tqdm(urls, desc="Crawling"):
            url_str, result, error = _collect_one_wrapper(url, cfg)
            if error:
                print(f"\n✗ {url_str}: {error}")
                errors.append((url_str, error))
            else:
                print(f"✓ {url_str}")
                results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(_collect_one_wrapper, url, cfg): url for url in urls
            }

            # Process as they complete
            with tqdm(total=len(urls), desc="Crawling") as pbar:
                for future in as_completed(future_to_url):
                    url_str, result, error = future.result()
                    if error:
                        print(f"\n✗ {url_str}: {error}")
                        errors.append((url_str, error))
                    else:
                        results.append(result)
                    pbar.update(1)

    print(f"\nCompleted: {len(results)} successful, {len(errors)} failed")

    if errors:
        print("\nFailed URLs:")
        for url, error in errors[:10]:  # Show first 10
            print(f"  - {url}:")
            # Print first line of error
            error_lines = str(error).split("\n")
            print(f"    {error_lines[0][:150]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return results
