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
