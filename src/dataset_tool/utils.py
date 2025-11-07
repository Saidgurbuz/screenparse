import hashlib, re, json, os, orjson, unicodedata
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from slugify import slugify


def stable_hash(s: str, length: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def canonicalize_url(url: str) -> str:
    u = urlparse(url)
    q = sorted(
        (k, v)
        for k, v in parse_qsl(u.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    )
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q), ""))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj))


def load_json(path: str):
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def safe_stem_from_url(url: str) -> str:
    # human-ish but stable
    u = urlparse(url)
    base = f"{u.netloc}-{u.path}".strip("/")
    base = unicodedata.normalize("NFKC", base)
    base = slugify(base)[:80] or "page"
    return base


def rect_to_int(r):
    return {k: int(round(v)) for k, v in r.items()}


def get_processed_url_stems(out_dir: str) -> set:
    """
    Get set of URL stems for already-processed URLs.

    Scans output directory for .png files and extracts URL stems
    from filenames (format: stem-HASH.png).

    Args:
        out_dir: Directory containing processed files

    Returns:
        Set of URL stems (strings)
    """
    import os
    import glob

    if not os.path.exists(out_dir):
        return set()

    processed_stems = set()

    # Find all .png files (screenshot files)
    png_files = glob.glob(os.path.join(out_dir, "*.png"))

    for png_file in png_files:
        # Extract stem from filename: stem-HASH.png
        basename = os.path.basename(png_file)
        if basename.endswith(".png"):
            # Remove .png suffix
            name_without_ext = basename[:-4]  # len(".png") == 4
            # Stem is everything before the last hyphen
            parts = name_without_ext.rsplit("-", 1)
            if len(parts) == 2:
                url_stem = parts[0]
                processed_stems.add(url_stem)

    return processed_stems
