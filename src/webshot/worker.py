"""Persistent worker process with browser reuse for maximum parallelization."""

import os
import time
import traceback
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

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
    navigate_resilient,
)
from .labels import guess_type
from tqdm import tqdm
from docling_ibm_models.reading_order.reading_order_rb import Size

# ---------------------------------------------------------------------------
# Iframe classification
# ---------------------------------------------------------------------------
AD_IFRAME_DOMAINS = {
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "adnxs.com", "amazon-adsystem.com", "taboola.com", "outbrain.com",
    "criteo.com", "pubmatic.com", "openx.net", "rubiconproject.com",
    "moatads.com", "adsrvr.org", "facebook.com/tr",
}
TRACKING_IFRAME_DOMAINS = {
    "google-analytics.com", "googletagmanager.com",
    "hotjar.com", "segment.com", "mixpanel.com",
}


def classify_iframe(frame) -> str:
    """Returns 'content', 'ad', or 'tracking'."""
    domain = urlparse(frame.url or "").netloc.lower()
    for ad in AD_IFRAME_DOMAINS:
        if domain.endswith(ad):
            return "ad"
    for tr in TRACKING_IFRAME_DOMAINS:
        if domain.endswith(tr):
            return "tracking"
    # Size heuristic: 1x1 or 0x0 iframes are tracking pixels
    try:
        rect = frame.evaluate("""() => {
            if (window.frameElement) {
                const r = window.frameElement.getBoundingClientRect();
                return {w: r.width, h: r.height};
            }
            return {w: 0, h: 0};
        }""")
        if rect["w"] <= 1 or rect["h"] <= 1:
            return "tracking"
    except Exception:
        pass
    return "content"


# ---------------------------------------------------------------------------
# Ad-blocking route domains
# ---------------------------------------------------------------------------
AD_BLOCK_DOMAINS = [
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "adnxs.com", "amazon-adsystem.com", "taboola.com", "outbrain.com",
    "google-analytics.com", "googletagmanager.com", "hotjar.com",
    "criteo.com", "pubmatic.com", "openx.net", "rubiconproject.com",
    "moatads.com", "adsrvr.org",
]

# ---------------------------------------------------------------------------
# Cookie consent auto-dismiss JS
# ---------------------------------------------------------------------------
COOKIE_DISMISS_JS = """() => {
    const SELECTORS = [
        '#onetrust-accept-btn-handler',
        '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll',
        '.cc-btn.cc-dismiss', '.cc-accept', '.cc-allow',
        '#sp-cc-accept',
        '.cmpboxbtn.cmpboxbtnyes',
        'button[class*="accept"]', 'a[class*="accept-cookies"]',
        '[aria-label*="accept" i]', '[aria-label*="agree" i]',
    ];
    for (const sel of SELECTORS) {
        try {
            const btn = document.querySelector(sel);
            if (btn && btn.offsetParent !== null) { btn.click(); return true; }
        } catch(e) {}
    }
    // Fallback: find buttons with accept/agree text
    for (const btn of document.querySelectorAll('button, a[role="button"]')) {
        const text = (btn.innerText || '').trim();
        if (/^(accept|agree|allow|ok|got it|accept all|allow all)$/i.test(text)) {
            if (btn.offsetParent !== null) { btn.click(); return true; }
        }
    }
    return false;
}"""

# ---------------------------------------------------------------------------
# Overlay / chat widget dismissal JS
# ---------------------------------------------------------------------------
DISMISS_OVERLAYS_JS = """() => {
    let removed = 0;
    const vpW = window.innerWidth;
    const vpH = window.innerHeight;
    const vpArea = vpW * vpH;

    document.querySelectorAll('*').forEach(el => {
        const s = getComputedStyle(el);
        if (s.position !== 'fixed' && s.position !== 'sticky') return;
        const z = parseInt(s.zIndex, 10);
        if (isNaN(z) || z <= 100) return;
        const r = el.getBoundingClientRect();
        const elArea = r.width * r.height;
        if (elArea / vpArea > 0.3) {
            el.style.display = 'none';
            removed++;
        }
    });
    return removed;
}"""

HIDE_CHAT_WIDGETS_JS = """() => {
    const SELECTORS = [
        '#intercom-container', '#drift-widget',
        '#hubspot-messages-iframe-container',
        '[class*="zendesk"]', '[id*="intercom"]',
        '[id*="drift"]', '[class*="chat-widget"]',
    ];
    let hidden = 0;
    for (const sel of SELECTORS) {
        try {
            document.querySelectorAll(sel).forEach(el => {
                el.style.display = 'none';
                hidden++;
            });
        } catch(e) {}
    }
    return hidden;
}"""


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
                # Anti-bot-detection stealth
                "--disable-blink-features=AutomationControlled",
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
        """Internal collection logic for a single URL, reusing persistent browser."""
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
            # Optional request-level ad blocking
            if cfg.block_ad_requests:
                for domain in AD_BLOCK_DOMAINS:
                    ctx.route(f"**{domain}**", lambda route: route.abort())

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

            # Wait for network idle with ceiling, then brief stabilization
            try:
                page.wait_for_load_state("networkidle", timeout=cfg.network_idle_timeout_ms)
            except Exception:
                pass  # If networkidle times out (streaming/websocket sites), continue
            page.wait_for_timeout(cfg.post_idle_stabilization_ms)

            # Dismiss cookie consent banners
            if cfg.dismiss_cookie_consent:
                try:
                    dismissed = page.evaluate(COOKIE_DISMISS_JS)
                    if dismissed:
                        page.wait_for_timeout(500)  # Wait for banner animation
                except Exception:
                    pass

            # Hide chat widgets
            if cfg.hide_chat_widgets:
                try:
                    page.evaluate(HIDE_CHAT_WIDGETS_JS)
                except Exception:
                    pass

            # Dismiss large overlays (opt-in, aggressive)
            if cfg.dismiss_overlays:
                try:
                    page.evaluate(DISMISS_OVERLAYS_JS)
                except Exception:
                    pass

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
                # Force above-the-fold lazy images to load
                page.evaluate("""() => {
                    document.querySelectorAll('img[loading="lazy"]').forEach(img => {
                        if (img.getBoundingClientRect().top < window.innerHeight * 1.5) {
                            img.loading = 'eager';
                            if (img.dataset.src) img.src = img.dataset.src;
                            if (img.dataset.srcset) img.srcset = img.dataset.srcset;
                        }
                    });
                }""")
                # Wait for images to finish loading instead of hard timeout
                try:
                    page.wait_for_function("""() => {
                        const images = Array.from(document.images);
                        return images.every(img => img.complete);
                    }""", timeout=3000)
                except Exception:
                    page.wait_for_timeout(1000)

            viewport_width = cfg.viewport.width
            viewport_height = cfg.viewport.height

            # Collect elements from all frames
            frames = [page.main_frame] + [
                fr for fr in page.frames if fr is not page.main_frame
            ]
            all_elements: List[Dict[str, Any]] = []
            all_texts: List[Dict[str, Any]] = []

            for fr_idx, fr in enumerate(frames):
                # Skip ad/tracking iframes
                if cfg.filter_ad_iframes and fr is not page.main_frame:
                    iframe_type = classify_iframe(fr)
                    if iframe_type in ("ad", "tracking"):
                        continue

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
                    traverse_shadow = cfg.traverse_shadow_dom
                    els = fr.evaluate(
                        """([viewportHeight, traverseShadow]) => {
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

                          // CSS filter opacity
                          if (s.filter && s.filter.includes('opacity(0')) return false;

                          // CSS transform hiding (scale(0))
                          const transform = s.transform;
                          if (transform && transform !== 'none') {
                            if (/scale\\s*\\(\\s*0[\\s,)]/.test(transform)) return false;
                          }

                          // max-height: 0 with overflow: hidden (common collapse pattern)
                          if (s.maxHeight === '0px' && (s.overflow === 'hidden' || s.overflowY === 'hidden')) {
                            return false;
                          }

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

                        // Collect all DOM nodes, optionally entering shadow roots
                        function collectAllNodes(root, visited, depth) {
                          if (!visited) visited = new Set();
                          if (!depth) depth = 0;
                          if (depth > 20) return [];  // Max shadow DOM nesting depth
                          if (visited.has(root)) return [];  // Cycle detection
                          visited.add(root);
                          const result = [];
                          const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
                          let node = walker.nextNode();
                          while (node) {
                            result.push(node);
                            if (traverseShadow && node.shadowRoot && !visited.has(node.shadowRoot)) {
                              result.push(...collectAllNodes(node.shadowRoot, visited, depth + 1));
                            }
                            node = walker.nextNode();
                          }
                          return result;
                        }

                        // First pass: collect all visible elements and assign indices
                        const nodes = traverseShadow ? collectAllNodes(document) : Array.from(document.querySelectorAll('*'));
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
                          const s = getComputedStyle(n);
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
                            z: Number(s.zIndex) || 0,
                            position: s.position,
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
                        [viewport_height, traverse_shadow],
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
            from .filtering import filter_elements, analyze_filtering_impact, get_leaf_elements

            # Keep reference to unfiltered elements — needed later for leaf extraction
            # and mixed-content text recovery (both require the original DOM structure).
            unfiltered_elements = all_elements

            if cfg.filter_config.save_unfiltered:
                save_json(f"{base}.elements.unfiltered.json", unfiltered_elements)

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

            # Annotate elements with reading order position (in-place, no reordering).
            # Physically reordering the list would invalidate parent_index/children_indices
            # fields, which reference positions in the filtered order. By annotating
            # in-place, hierarchy indices remain valid for the saved elements.json.
            from .visualize import _build_reading_order
            try:
                from PIL import Image as PILImage
                im = PILImage.open(img_path)
                page_size = Size(width=im.width, height=im.height)
                reading_order_indices = _build_reading_order(all_elements, page_size)
                # Annotate each element with its reading order position
                for order_pos, elem_idx in enumerate(reading_order_indices):
                    all_elements[elem_idx]["reading_order_index"] = order_pos
            except Exception as e:
                print(f"[Worker {self.worker_id}] Failed to apply reading order: {e}")

            # Elements stay in their filtered order so hierarchy indices remain valid
            ordered_elements = all_elements

            # Compute leaf coverage set: true leaves + mixed-content text blocks.
            # reading_order_index must already be annotated above before this call.
            leaf_elements = get_leaf_elements(ordered_elements, unfiltered_elements)

            save_json(record["meta_path"], meta)
            save_json(record["elements_path"], ordered_elements)

            leaf_path = f"{base}.elements.leaf.json"
            save_json(leaf_path, leaf_elements)
            record["leaf_elements_path"] = leaf_path
            save_json(record["texts_path"], all_texts)
            save_json(record["ax_path"], ax_tree)

            if ocr_results:
                record["ocr_path"] = f"{base}.ocr.json"
                save_json(record["ocr_path"], ocr_results)

            # Build and save element tree structure (using ordered elements)
            try:
                element_tree = build_element_tree(ordered_elements)
                record["tree_path"] = f"{base}.tree.json"
                save_json(record["tree_path"], element_tree)
            except Exception as e:
                print(f"[Worker {self.worker_id}] Failed to build element tree: {e}")

            # Generate and save ScreenTag representation (using ordered elements)
            try:
                screentag_repr = elements_to_screentag(
                    elements=ordered_elements,
                    viewport=meta["viewport"],
                )
                record["screentag_path"] = f"{base}.screentag.txt"
                with open(record["screentag_path"], "w", encoding="utf-8") as f:
                    f.write(screentag_repr)
            except Exception as e:
                print(f"[Worker {self.worker_id}] Failed to generate ScreenTag: {e}")

            # Triplets (using ordered elements)
            try:
                ocr_map = (
                    {o["element_index"]: o["text"] for o in ocr_results}
                    if ocr_results
                    else {}
                )
                triplets = []
                for idx, el in enumerate(ordered_elements):
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
            except Exception as e:
                print(f"[Worker {self.worker_id}] Failed to generate triplets: {e}")

            # Reading order visualization (two files: leaf-only and all-elements)
            try:
                from .visualize import visualize_reading_order, visualize_reading_order_flat
                import os as _os

                ro_path = f"{base}_readingorder.png"
                visualize_reading_order(img_path, record["elements_path"], ro_path)
                record["reading_order_path"] = ro_path
                # The _all variant is written alongside by visualize_reading_order
                ro_all_path = ro_path.replace("_readingorder.png", "_readingorder_all.png")
                if _os.path.exists(ro_all_path):
                    record["reading_order_all_path"] = ro_all_path

                # Leaf reading order: flat list sorted by reading_order_index,
                # covering all visible UI without containers.
                ro_leaf_path = f"{base}_readingorder_leaf.png"
                visualize_reading_order_flat(img_path, record["leaf_elements_path"], ro_leaf_path)
                record["reading_order_leaf_path"] = ro_leaf_path
            except Exception as e:
                print(f"[Worker {self.worker_id}] Failed to generate reading order visualization: {e}")
                pass

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
