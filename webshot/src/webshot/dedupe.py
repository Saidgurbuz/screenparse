# src/webshot/dedupe.py
import os, csv
from typing import Dict, List, Tuple, Optional
from PIL import Image
import imagehash
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from .utils import ensure_dir

# ---------------------- Utilities ----------------------


def _phash_one(path: str) -> Tuple[str, Optional[int]]:
    """
    Compute perceptual hash (pHash) for one image and return it as a 64-bit int.
    Returns (path, hash_int or None if invalid).
    """
    try:
        with Image.open(path) as im:
            h = imagehash.phash(im)  # 64-bit perceptual hash
            # Convert to integer via hex string to avoid numpy packing differences
            return path, int(str(h), 16)
    except (IOError, OSError, Image.DecompressionBombError):
        return path, None


def _hamming_int(a: int, b: int) -> int:
    """Hamming distance between two 64-bit ints."""
    return (a ^ b).bit_count()


# ---------------------- BK-tree for Hamming distance ----------------------


class _BKNode:
    __slots__ = ("key", "indices", "children")

    def __init__(self, key: int, first_index: int):
        self.key: int = key
        self.indices: List[int] = [first_index]  # handle exact-equal hashes
        self.children: Dict[int, "_BKNode"] = {}  # edge labeled by distance


class _BKTree:
    """
    BK-tree over 64-bit integer hashes with Hamming distance.
    Supports exact radius queries (no false negatives).
    """

    def __init__(self):
        self.root: Optional[_BKNode] = None

    def insert(self, key: int, idx: int):
        if self.root is None:
            self.root = _BKNode(key, idx)
            return
        node = self.root
        while True:
            d = _hamming_int(key, node.key)
            if d == 0:
                node.indices.append(idx)
                return
            child = node.children.get(d)
            if child is None:
                node.children[d] = _BKNode(key, idx)
                return
            node = child

    def query(self, key: int, radius: int) -> List[int]:
        """
        Return indices of items within Hamming distance <= radius from key.
        """
        out: List[int] = []
        node = self.root
        if node is None:
            return out
        stack = [node]
        while stack:
            node = stack.pop()
            d = _hamming_int(key, node.key)
            if d <= radius:
                out.extend(node.indices)
            # Only children with edge distance in [d - r, d + r] can contain matches
            lo = max(0, d - radius)
            hi = d + radius
            for edge_dist, child in node.children.items():
                if lo <= edge_dist <= hi:
                    stack.append(child)
        return out


# ---------------------- Public API ----------------------


def find_duplicates(
    image_dir: str,
    distance_threshold: int = 8,
    workers: Optional[int] = None,
    chunksize: int = 128,
) -> List[List[str]]:
    """
    Find groups of near-duplicate images using perceptual hash (pHash)
    and Hamming distance threshold. Exact, but much faster than O(n^2)
    thanks to a BK-tree index.

    Args:
        image_dir: directory containing images (non-recursive; same as original)
        distance_threshold: max Hamming distance to consider duplicates (default 8)
        workers: number of worker processes for hashing (default: all CPUs)
        chunksize: number of files per hashing task chunk

    Returns:
        List of groups; each group is a list of image paths (length >= 2).
    """
    # Collect images (top-level only, like original)
    try:
        entries = os.listdir(image_dir)
    except FileNotFoundError:
        return []

    imgs = [
        os.path.join(image_dir, f)
        for f in entries
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not imgs:
        return []

    if workers is None:
        workers = max(1, min(cpu_count() or 1, 128))

    # 1) Parallel pHash -> integer hashes
    valid_paths: List[str] = []
    hashes_int: List[int] = []

    with Pool(processes=workers) as pool:
        for path, hval in tqdm(
            pool.imap_unordered(_phash_one, imgs, chunksize=chunksize),
            total=len(imgs),
            desc="Computing hashes",
            unit="img",
        ):
            if hval is not None:
                valid_paths.append(path)
                hashes_int.append(hval)

    n = len(valid_paths)
    if n < 2:
        return []

    # Keep paths/hashes aligned; optionally sort for determinism
    # Sorting improves reproducibility of groups
    zipped = sorted(zip(valid_paths, hashes_int), key=lambda x: x[0])
    valid_paths = [p for p, _ in zipped]
    hashes_int = [h for _, h in zipped]

    # 2) Build duplicate groups using BK-tree + union-find (DSU)
    class DSU:
        def __init__(self, n: int):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, x: int) -> int:
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, a: int, b: int):
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if self.rank[ra] < self.rank[rb]:
                self.parent[ra] = rb
            elif self.rank[ra] > self.rank[rb]:
                self.parent[rb] = ra
            else:
                self.parent[rb] = ra
                self.rank[ra] += 1

    dsu = DSU(n)
    tree = _BKTree()

    # Insert progressively; for each hash, query neighbors among previously inserted ones
    # to avoid duplicate comparisons and to keep memory bounded.
    # This preserves exactness and the "one group per connected component" logic.
    for i in tqdm(range(n), desc="Indexing & matching", unit="img"):
        h = hashes_int[i]
        # query neighbors among already inserted items
        neighbors = tree.query(h, distance_threshold)
        for j in neighbors:
            # 'neighbors' can include duplicates with same hash (distance 0)
            dsu.union(i, j)
        # then insert current
        tree.insert(h, i)

    # 3) Extract connected components as groups (size >= 2)
    comps: Dict[int, List[str]] = {}
    for i in range(n):
        root = dsu.find(i)
        comps.setdefault(root, []).append(valid_paths[i])

    groups = [g for g in comps.values() if len(g) > 1]
    return groups


def write_groups_csv(groups: List[List[str]], out_csv: str):
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "image_path"])
        for gid, g in enumerate(groups):
            for p in g:
                w.writerow([gid, p])
