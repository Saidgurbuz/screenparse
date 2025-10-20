import csv, os
from typing import List
from tqdm import tqdm
from .collector import collect_one
from .config import Config
from .utils import ensure_dir


def read_urls(csv_path: str) -> List[str]:
    urls = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            u = row[0].strip()
            if u and not u.startswith("#"):
                urls.append(u)
    return urls


def crawl(csv_path: str, out_dir: str, do_ocr: bool = False, headless: bool = True):
    ensure_dir(out_dir)
    cfg = Config(out_dir=out_dir, do_ocr=do_ocr, headless=headless)
    urls = read_urls(csv_path)
    results = []
    for u in tqdm(urls, desc="Collecting"):
        try:
            rec = collect_one(u, cfg)
            results.append(rec)
        except Exception as e:
            print("FAIL:", u, e)
    return results
