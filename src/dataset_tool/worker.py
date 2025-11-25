"""Persistent worker process with browser reuse for maximum parallelization."""

import os
import time
import traceback
from typing import Dict, Any, Optional, List
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
from .config import Config
from .utils import (
    stable_hash,
    canonicalize_url,
    ensure_dir,
    save_json,
    safe_stem_from_url,
    rect_to_int,
    build_element_tree,
    elements_to_screentag,
)
from .labels import guess_type
from tqdm import tqdm


def is_blank_image(img_path: str, white_threshold: float = 0.99) -> bool:
    """
    Check if an image is blank (mostly white/single color).
    
    Args:
        img_path: Path to the image file
        white_threshold: Fraction of pixels that must be white/near-white to consider blank
        
    Returns:
        True if image is blank, False otherwise
    """
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(img_path).convert('RGB')
        pixels = np.array(img)
        
        # Check if image is mostly white (RGB values > 250)
        white_pixels = np.all(pixels > 250, axis=2)
        white_ratio = np.mean(white_pixels)
        
        if white_ratio > white_threshold:
            return True
        
        # Also check for single-color images (very low variance)
        # This catches solid gray, black, or other solid color pages
        # Compute std per channel and take the max - if all pixels are identical,
        # std will be 0 for all channels regardless of the RGB values
        pixel_std = np.max([np.std(pixels[:, :, c]) for c in range(3)])
        if pixel_std < 5:  # Very low variance = likely single solid color
            return True
            
        return False
    except Exception:
        # If we can't read the image, consider it invalid
        return True


def navigate_resilient(page, url, nav_timeout_ms):
    """Navigate to URL with resilient retry logic.
    
    Tries progressively stricter wait conditions, with retries and backoffs.
    Handles various edge cases like client-side redirects and interstitials.
    """
    # Try the least strict first; escalate only if needed
    waits = ["commit", "domcontentloaded", "load"]
    last_exc = None
    for attempt in range(3):
        for w in waits:
            try:
                resp = page.goto(url, wait_until=w, timeout=nav_timeout_ms)
                # Small settle to let async content paint without hanging forever.
                page.wait_for_timeout(500)
                return resp
            except Exception as e:
                last_exc = e
        # Backoff & retry a reload of final URL (handles client-side redirects)
        try:
            page.reload(wait_until="domcontentloaded", timeout=nav_timeout_ms)
            page.wait_for_timeout(500)
            return None
        except Exception as e:
            last_exc = e
            page.wait_for_timeout(750 * (attempt + 1))
    # Give one last shot via JS redirect in case of weird interstitial
    try:
        page.evaluate(f"location.href = {repr(url)}")
        page.wait_for_load_state("domcontentloaded", timeout=nav_timeout_ms)
        page.wait_for_timeout(500)
        return None
    except Exception:
        if last_exc:
            raise last_exc
        raise


class BrowserWorker:
    """Worker that maintains a persistent browser for processing multiple URLs."""

    def __init__(self, worker_id: int, cfg: Config):
        self.worker_id = worker_id
        self.cfg = cfg
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.processed_count = 0
        self.error_count = 0

    def __enter__(self):
        """Initialize browser on worker start."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.cfg.headless,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-features=TranslateUI",
                "--no-sandbox",
                # Performance optimizations for parallel execution
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        print(f"[Worker {self.worker_id}] Browser initialized")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup browser on worker shutdown."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print(
            f"[Worker {self.worker_id}] Processed {self.processed_count} URLs ({self.error_count} errors)"
        )

    def process_url(
        self, url: str
    ) -> tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """Process a single URL using the persistent browser."""
        try:
            result = self._collect_one_internal(url)
            self.processed_count += 1
            return (url, result, None)
        except Exception as e:
            self.error_count += 1
            tb = traceback.format_exc()
            error_msg = f"{type(e).__name__}: {str(e)}\n{tb}"
            return (url, None, error_msg)

    def _collect_one_internal(self, url: str) -> Dict[str, Any]:
        """Internal collection logic - same as collector.py but reuses browser."""
        cfg = self.cfg
        ensure_dir(cfg.out_dir)
        url_c = canonicalize_url(url)
        key = stable_hash(url_c)
        stem = safe_stem_from_url(url_c)
        base = os.path.join(cfg.out_dir, f"{stem}-{key}")

        # Create NEW context for each URL (cheap, provides isolation)
        ctx = self.browser.new_context(
            viewport={"width": cfg.viewport.width, "height": cfg.viewport.height},
            device_scale_factor=cfg.viewport.device_scale_factor,
            locale=cfg.locale,
            color_scheme=cfg.color_scheme,
            ignore_https_errors=True,
            user_agent=getattr(cfg, "user_agent", None),
        )

        try:
            page = ctx.new_page()
            page.add_style_tag(
                content="""
                * { animation: none !important; transition: none !important; }
                html { scroll-behavior: auto !important; }
            """
            )
            nav_timeout_ms = getattr(
                getattr(cfg, "timeouts", object()), "navigation_ms", 30000
            )
            page.set_default_navigation_timeout(nav_timeout_ms)
            page.set_default_timeout(max(nav_timeout_ms, 30000))

            # Navigate
            response = navigate_resilient(page, url_c, nav_timeout_ms)
            final_url = page.url
            page.wait_for_timeout(cfg.network_idle_wait_ms)

            # Capture mode
            if cfg.capture_full_page:
                page.evaluate(
                    """() => new Promise(res => {
                    let y=0; const step=window.innerHeight*0.8;
                    function s(){ y+=step; window.scrollTo({top:y,behavior:'instant'});
                      if(y<document.body.scrollHeight-2*window.innerHeight) requestAnimationFrame(s); else res(); }
                    s();
                })"""
                )
                page.wait_for_timeout(400)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(200)
            else:
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(200)

            viewport_width = cfg.viewport.width
            viewport_height = cfg.viewport.height

            # Collect elements from all frames
            frames = [page.main_frame] + [
                fr for fr in page.frames if fr is not page.main_frame
            ]
            all_elements: List[Dict[str, Any]] = []
            all_texts: List[Dict[str, Any]] = []

            for fr_idx, fr in enumerate(frames):
                try:
                    off = fr.evaluate(
                        """() => {
                          try {
                            if (window.frameElement) {
                              const r = window.frameElement.getBoundingClientRect();
                              return {x: Math.round(r.x), y: Math.round(r.y)};
                            }
                          } catch(e) {}
                          return {x:0, y:0};
                        }"""
                    )
                except Exception:
                    off = {"x": 0, "y": 0}

                skip_reason = None
                # Collect elements with hierarchy information
                try:
                    els = fr.evaluate(
                        """(viewportHeight) => {
                        function isVisible(el){
                          if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
                          
                          // Check HTML hidden attribute
                          if (el.hidden) return false;
                          
                          // Check aria-hidden (but not on the element itself if it has visible children)
                          if (el.getAttribute('aria-hidden') === 'true') return false;
                          
                          const s = getComputedStyle(el);
                          
                          // Basic visibility checks
                          if (s.display === 'none') return false;
                          if (s.visibility === 'hidden' || s.visibility === 'collapse') return false;
                          if (parseFloat(s.opacity) === 0) return false;
                          
                          // Check for off-screen positioning (common dropdown hiding technique)
                          const left = parseFloat(s.left);
                          const top = parseFloat(s.top);
                          if (!isNaN(left) && left < -1000) return false;
                          if (!isNaN(top) && top < -1000) return false;
                          
                          // Check for clip/clip-path hiding
                          if (s.clip === 'rect(0px, 0px, 0px, 0px)' || 
                              s.clip === 'rect(0, 0, 0, 0)' ||
                              s.clipPath === 'inset(100%)' ||
                              s.clipPath === 'polygon(0 0, 0 0, 0 0, 0 0)') return false;
                          
                          // Check for zero-size with overflow hidden (collapsed elements)
                          const r = el.getBoundingClientRect();
                          if (r.width <= 0 || r.height <= 0) return false;
                          
                          // Check if element is within viewport bounds
                          if (r.right < 0 || r.bottom < 0) return false;
                          if (r.left > window.innerWidth || r.top > viewportHeight) return false;
                          
                          // Check if element is actually clipped by an ancestor with overflow:hidden
                          let parent = el.parentElement;
                          while (parent && parent !== document.body) {
                            const ps = getComputedStyle(parent);
                            if (ps.overflow === 'hidden' || ps.overflowX === 'hidden' || ps.overflowY === 'hidden') {
                              const pr = parent.getBoundingClientRect();
                              // If element is completely outside parent's visible area
                              if (r.right <= pr.left || r.left >= pr.right ||
                                  r.bottom <= pr.top || r.top >= pr.bottom) {
                                return false;
                              }
                            }
                            // Also check parent visibility
                            if (ps.display === 'none' || ps.visibility === 'hidden' || parseFloat(ps.opacity) === 0) {
                              return false;
                            }
                            parent = parent.parentElement;
                          }
                          
                          return true;
                        }
                        
                        // First pass: collect all visible elements and assign indices
                        const nodes = Array.from(document.querySelectorAll('*'));
                        const visibleNodes = [];
                        const nodeToIndex = new Map();
                        
                        for (const n of nodes) {
                          if (isVisible(n)) {
                            nodeToIndex.set(n, visibleNodes.length);
                            visibleNodes.push(n);
                          }
                        }
                        
                        // Second pass: build output with parent references
                        const out = [];
                        for (let i = 0; i < visibleNodes.length; i++) {
                          const n = visibleNodes[i];
                          const r = n.getBoundingClientRect();
                          const attrs = {};
                          for (const a of n.attributes) attrs[a.name] = a.value;
                          const role = n.getAttribute('role');
                          const text = (n.innerText || '').replace(/\\s+/g,' ').trim();
                          
                          // Find parent index - walk up until we find a visible parent
                          let parentIndex = null;
                          let parent = n.parentElement;
                          while (parent) {
                            if (nodeToIndex.has(parent)) {
                              parentIndex = nodeToIndex.get(parent);
                              break;
                            }
                            parent = parent.parentElement;
                          }
                          
                          // Collect direct children indices (only visible ones)
                          const childrenIndices = [];
                          for (const child of n.children) {
                            if (nodeToIndex.has(child)) {
                              childrenIndices.push(nodeToIndex.get(child));
                            }
                          }
                          
                          // Compute depth in DOM tree
                          let depth = 0;
                          let p = n.parentElement;
                          while (p) { depth++; p = p.parentElement; }
                          
                          out.push({
                            tag: n.tagName.toLowerCase(),
                            role: role || null,
                            id: n.id || null,
                            classes: n.className || null,
                            attrs,
                            rect: {x:r.x, y:r.y, w:r.width, h:r.height},
                            z: Number(getComputedStyle(n).zIndex) || 0,
                            aria_hidden: n.getAttribute('aria-hidden') || null,
                            inner_text: text,
                            // Hierarchy information
                            _dom_index: i,
                            _parent_dom_index: parentIndex,
                            _children_dom_indices: childrenIndices,
                            _depth: depth
                          });
                        }
                        return out;
                    }""",
                        viewport_height,
                    )
                    # Calculate global index offset for multi-frame support
                    global_offset = len(all_elements)
                    
                    for e in els:
                        e["frame_index"] = fr_idx
                        e["type"] = guess_type(
                            e.get("tag") or "",
                            e.get("role") or "",
                            e.get("attrs") or {},
                        )
                        e["rect"]["x"] = int(round(e["rect"]["x"])) + off["x"]
                        e["rect"]["y"] = int(round(e["rect"]["y"])) + off["y"]
                        e["rect"] = rect_to_int(e["rect"])
                        
                        # Adjust hierarchy indices for global offset (multi-frame support)
                        if global_offset > 0:
                            e["_dom_index"] = e["_dom_index"] + global_offset
                            if e["_parent_dom_index"] is not None:
                                e["_parent_dom_index"] = e["_parent_dom_index"] + global_offset
                            e["_children_dom_indices"] = [
                                idx + global_offset for idx in e["_children_dom_indices"]
                            ]
                    
                    all_elements.extend(els)
                except Exception as e:
                    # put the exception in the skip reason for debugging
                    skip_reason = str(e)
                    pass

                # Collect text spans
                try:
                    tboxes = fr.evaluate(
                        """(viewportHeight) => {
                      const res=[];
                      const walker=document.createTreeWalker(document, NodeFilter.SHOW_TEXT);
                      let n=walker.nextNode();
                      while(n){
                        const s = (n.nodeValue || '').replace(/\\s+/g,' ').trim();
                        if(s){
                          const range = document.createRange();
                          range.selectNodeContents(n);
                          const rects = Array.from(range.getClientRects());
                          for(const r of rects){
                            if(r.width>0 && r.height>0 && r.top < viewportHeight && r.bottom > 0){
                              const parent = range.startContainer.parentElement;
                              const cs = parent ? getComputedStyle(parent) : null;
                              res.push({
                                text: s,
                                rect:{x:r.x, y:r.y, w:r.width, h:r.height},
                                font_family: cs ? cs.fontFamily : null,
                                font_size: cs ? cs.fontSize : null,
                                font_weight: cs ? cs.fontWeight : null
                              });
                            }
                          }
                        }
                        n=walker.nextNode();
                      }
                      return res;
                    }""",
                        viewport_height,
                    )
                    for t in tboxes:
                        t["frame_index"] = fr_idx
                        t["rect"]["x"] = int(round(t["rect"]["x"])) + off["x"]
                        t["rect"]["y"] = int(round(t["rect"]["y"])) + off["y"]
                        t["rect"] = rect_to_int(t["rect"])
                    all_texts.extend(tboxes)
                except Exception as e:
                    skip_reason = str(e)
                    pass

            # Filter elements
            from .filtering import filter_elements, analyze_filtering_impact

            if cfg.filter_config.save_unfiltered:
                save_json(f"{base}.elements.unfiltered.json", all_elements)

            filtered_elements = filter_elements(
                all_elements,
                viewport_w=viewport_width,
                viewport_h=viewport_height,
                iou_threshold=cfg.filter_config.iou_threshold,
                containment_threshold=cfg.filter_config.containment_threshold,
                min_box_size=cfg.filter_config.min_box_size,
                max_box_size=cfg.filter_config.max_box_size,
            )

            all_elements = filtered_elements

            # Accessibility tree
            ax_tree = page.accessibility.snapshot(root=None, interesting_only=False)

            # Screenshot
            img_path = f"{base}.png"
            try:
                if cfg.capture_full_page:
                    page.screenshot(path=img_path, full_page=True, scale="css")
                else:
                    page.screenshot(path=img_path, scale="css")
                scale_used = "css"
            except TypeError:
                if cfg.capture_full_page:
                    page.screenshot(path=img_path, full_page=True)
                else:
                    page.screenshot(path=img_path)
                scale_used = "device"

            # ============================================================
            # VALIDATION: Skip samples with empty elements or blank images
            # ============================================================
            
            # Check 1: Empty elements list
            if not all_elements or len(all_elements) == 0:
                skip_reason = skip_reason + "empty_elements" if skip_reason else "empty_elements"
            
            # Check 2: Blank/white screenshot
            elif is_blank_image(img_path):
                skip_reason = skip_reason + "blank_image" if skip_reason else "blank_image"
            
            if skip_reason:
                # Clean up the screenshot file we just created
                try:
                    os.remove(img_path)
                except OSError:
                    pass
                # Also clean up unfiltered elements if saved
                unfiltered_path = f"{base}.elements.unfiltered.json"
                if os.path.exists(unfiltered_path):
                    try:
                        os.remove(unfiltered_path)
                    except OSError:
                        pass
                # Return None to indicate skipped sample
                return {"skipped": True, "reason": skip_reason, "url": url_c}

            # Metadata
            meta = {
                "url": url_c,
                "ts": int(time.time()),
                "viewport": {
                    "w": viewport_width,
                    "h": viewport_height,
                    "dpr": cfg.viewport.device_scale_factor,
                },
                "browser": ctx.browser.version,
                "frames": len(frames),
                "capture_type": (
                    "full_page" if cfg.capture_full_page else "viewport_only"
                ),
                "screenshot_scale": scale_used,
                "css_to_image_scale": (
                    1 if scale_used == "css" else cfg.viewport.device_scale_factor
                ),
                "requested_url": url_c,
                "final_url": final_url,
                "status": response.status if response else None,
                "redirected": final_url != url_c,
            }

            if cfg.capture_full_page:
                meta["scroll_height"] = page.evaluate(
                    "() => document.body.scrollHeight"
                )

            # OCR (optional)
            ocr_results: List[Dict[str, Any]] = []
            if cfg.do_ocr:
                import shutil

                if shutil.which("tesseract"):
                    try:
                        import cv2, pytesseract

                        img = cv2.imread(img_path)
                        _sf = (
                            1
                            if meta["screenshot_scale"] == "css"
                            else meta["css_to_image_scale"]
                        )
                        for idx, el in enumerate(all_elements):
                            x = int(round(el["rect"]["x"] * _sf))
                            y = int(round(el["rect"]["y"] * _sf))
                            w = int(round(el["rect"]["w"] * _sf))
                            h = int(round(el["rect"]["h"] * _sf))
                            x2 = max(0, x)
                            y2 = max(0, y)
                            crop = img[y2 : y2 + h, x2 : x2 + w]
                            if crop.size > 0:
                                txt = pytesseract.image_to_string(crop).strip()
                                if txt:
                                    ocr_results.append(
                                        {"element_index": idx, "text": txt}
                                    )
                    except Exception:
                        pass

            # Write artifacts
            record = {
                "id": key,
                "image_path": img_path,
                "meta_path": f"{base}.meta.json",
                "elements_path": f"{base}.elements.json",
                "texts_path": f"{base}.texts.json",
                "ax_path": f"{base}.ax.json",
            }
            save_json(record["meta_path"], meta)
            save_json(record["elements_path"], all_elements)
            save_json(record["texts_path"], all_texts)
            save_json(record["ax_path"], ax_tree)

            if ocr_results:
                record["ocr_path"] = f"{base}.ocr.json"
                save_json(record["ocr_path"], ocr_results)

            # Build and save element tree structure
            element_tree = build_element_tree(all_elements)
            record["tree_path"] = f"{base}.tree.json"
            save_json(record["tree_path"], element_tree)
            
            # Generate and save ScreenTag representation
            screentag_repr = elements_to_screentag(all_elements, element_tree["roots"])
            record["screentag_path"] = f"{base}.screentag.txt"
            with open(record["screentag_path"], "w", encoding="utf-8") as f:
                f.write(screentag_repr)

            # Triplets
            ocr_map = (
                {o["element_index"]: o["text"] for o in ocr_results}
                if ocr_results
                else {}
            )
            triplets = []
            for idx, el in enumerate(all_elements):
                lt_rb = [
                    int(el["rect"]["x"]),
                    int(el["rect"]["y"]),
                    int(el["rect"]["x"] + el["rect"]["w"]),
                    int(el["rect"]["y"] + el["rect"]["h"]),
                ]
                text = (el.get("inner_text") or "").strip()
                if not text:
                    text = ocr_map.get(idx, "").strip()
                triplets.append(
                    {
                        "element_index": idx,
                        "bbox_ltrb": lt_rb,
                        "type": el.get("type") or "unknown",
                        "text": text,
                        "tag": el.get("tag"),
                        "role": el.get("role"),
                    }
                )

            if triplets:
                record["triplets_path"] = f"{base}.triplets.jsonl"
                with open(record["triplets_path"], "w", encoding="utf-8") as f:
                    import json as _json

                    for row in triplets:
                        f.write(_json.dumps(row, ensure_ascii=False) + "\n")

            return record

        finally:
            # Always close context (cheap operation, ensures clean state)
            ctx.close()


def worker_process_urls(worker_id: int, urls: List[str], cfg: Config) -> List[tuple]:
    """Entry point for worker process - processes a batch of URLs."""
    results = []
    with BrowserWorker(worker_id, cfg) as worker:
        for url in tqdm(urls, desc=f"Worker {worker_id}", position=worker_id):
            result = worker.process_url(url)
            results.append(result)
    return results
