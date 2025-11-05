# src/dataset_tool/collector.py
from typing import List, Dict, Any
import time, os, shutil
from playwright.sync_api import sync_playwright
from .config import Config
from .utils import (
    stable_hash,
    canonicalize_url,
    ensure_dir,
    save_json,
    safe_stem_from_url,
    rect_to_int,
)
from .labels import guess_type


def _has_tesseract() -> bool:
    return shutil.which("tesseract") is not None


def _xywh_to_lt_rb(r: Dict[str, int]):
    # convert {x,y,w,h} -> [l,t,r,b]
    return [int(r["x"]), int(r["y"]), int(r["x"] + r["w"]), int(r["y"] + r["h"])]


def collect_one(url: str, cfg: Config) -> Dict[str, Any]:
    ensure_dir(cfg.out_dir)
    url_c = canonicalize_url(url)
    key = stable_hash(url_c)
    stem = safe_stem_from_url(url_c)
    base = os.path.join(cfg.out_dir, f"{stem}-{key}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=cfg.headless,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-features=TranslateUI",
                "--no-sandbox",
            ],
        )
        ctx = browser.new_context(
            viewport={"width": cfg.viewport.width, "height": cfg.viewport.height},
            device_scale_factor=cfg.viewport.device_scale_factor,
            locale=cfg.locale,
            color_scheme=cfg.color_scheme,
        )
        page = ctx.new_page()
        page.add_style_tag(
            content="""
            * { animation: none !important; transition: none !important; }
            html { scroll-behavior: auto !important; }
        """
        )

        page.goto(url_c, wait_until="networkidle", timeout=10000)
        page.wait_for_timeout(cfg.network_idle_wait_ms)

        # ============================================================
        # Capture mode: viewport-only OR full-page with scrolling
        # ============================================================
        if cfg.capture_full_page:
            # OLD BEHAVIOR: Scroll to trigger lazy loads
            page.evaluate(
                """() => new Promise(res => {
                let y=0; const step=window.innerHeight*0.8;
                function s(){ y+=step; window.scrollTo({top:y,behavior:'instant'});
                  if(y<document.body.scrollHeight-2*window.innerHeight) requestAnimationFrame(s); else res(); }
                s();
            })"""
            )
            page.wait_for_timeout(400)
            # Reset to top after scrolling
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(200)
        else:
            # NEW BEHAVIOR: Stay at viewport top (no scrolling)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(200)

        # Get viewport dimensions for filtering
        viewport_width = cfg.viewport.width
        viewport_height = cfg.viewport.height

        # Collect per frame (main + iframes) with page-relative offsets
        frames = [page.main_frame] + [
            fr for fr in page.frames if fr is not page.main_frame
        ]
        all_elements: List[Dict[str, Any]] = []
        all_texts: List[Dict[str, Any]] = []

        for fr_idx, fr in enumerate(frames):
            # Compute frame offset relative to the main page (0,0 for main frame)
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

            # === ELEMENTS ===
            try:
                # Collect all visible elements with viewport filtering
                els = fr.evaluate(
                    """(viewportHeight) => {
                    function isVisible(el){
                      if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
                      const s = getComputedStyle(el);
                      if (s.display==='none' || s.visibility==='hidden' || parseFloat(s.opacity)===0) return false;
                      const r = el.getBoundingClientRect();
                      // Must have size and be at least partially in viewport
                      return r.width>0 && r.height>0 && r.top < viewportHeight && r.bottom > 0 && r.left < window.innerWidth && r.right > 0;
                    }
                    const out=[];
                    const nodes = Array.from(document.querySelectorAll('*'));
                    for (const n of nodes){
                      if(!isVisible(n)) continue;
                      const r = n.getBoundingClientRect();
                      const attrs={}; for (const a of n.attributes) attrs[a.name]=a.value;
                      const role = n.getAttribute('role');
                      const text = (n.innerText || '').replace(/\\s+/g,' ').trim();
                      out.push({
                        tag: n.tagName.toLowerCase(),
                        role: role || null,
                        id: n.id || null,
                        classes: n.className || null,
                        attrs,
                        rect: {x:r.x, y:r.y, w:r.width, h:r.height},
                        z: Number(getComputedStyle(n).zIndex) || 0,
                        aria_hidden: n.getAttribute('aria-hidden') || null,
                        inner_text: text
                      });
                    }
                    return out;
                }""",
                    viewport_height,
                )
                # normalize + page-relative
                for e in els:
                    e["frame_index"] = fr_idx
                    e["type"] = guess_type(
                        e.get("tag") or "", e.get("role") or "", e.get("attrs") or {}
                    )
                    e["rect"]["x"] = int(round(e["rect"]["x"])) + off["x"]
                    e["rect"]["y"] = int(round(e["rect"]["y"])) + off["y"]
                    e["rect"] = rect_to_int(e["rect"])
                all_elements.extend(els)
            except Exception as _:
                pass

            # === TEXT SPANS ===
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
                        // Filter to viewport
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
            except Exception:
                pass

        # === FILTER ELEMENTS - Import and use advanced filtering ===
        from .filtering import filter_elements, analyze_filtering_impact

        print(f"  Collected {len(all_elements)} raw elements")

        # Save unfiltered if requested (for debugging)
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

        # Analyze filtering impact
        stats = analyze_filtering_impact(all_elements, filtered_elements)

        if stats["protected_removed"] > 0:
            print(
                f"  ⚠️  WARNING: {stats['protected_removed']} protected elements were filtered!"
            )
            print(f"     Types: {stats['protected_removed_types']}")

        print(
            f"  Final: {len(filtered_elements)} elements (removed {stats['total_removed']}, {stats['removal_rate']*100:.1f}%)"
        )

        # Show what was removed by type
        if stats["removed_types"]:
            top_removed = sorted(
                stats["removed_types"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            print(f"  Top removed types: {dict(top_removed)}")

        all_elements = filtered_elements

        # AX tree
        ax_tree = page.accessibility.snapshot(root=None, interesting_only=False)

        # ============================================================
        # CRITICAL CHANGE: Screenshot only viewport (not full page)
        # ============================================================
        # Screenshot: viewport-only or full-page based on config
        img_path = f"{base}.png"
        try:
            if cfg.capture_full_page:
                page.screenshot(path=img_path, full_page=True, scale="css")
            else:
                page.screenshot(path=img_path, scale="css")  # Viewport only
            scale_used = "css"
        except TypeError:
            if cfg.capture_full_page:
                page.screenshot(path=img_path, full_page=True)
            else:
                page.screenshot(path=img_path)
            scale_used = "device"

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
            "capture_type": "full_page" if cfg.capture_full_page else "viewport_only",
            "screenshot_scale": scale_used,
            "css_to_image_scale": (
                1 if scale_used == "css" else cfg.viewport.device_scale_factor
            ),
        }

        if cfg.capture_full_page:
            meta["scroll_height"] = page.evaluate("() => document.body.scrollHeight")

        # Optional OCR per element (simple)
        ocr_results: List[Dict[str, Any]] = []
        if cfg.do_ocr and _has_tesseract():
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
                            ocr_results.append({"element_index": idx, "text": txt})
            except Exception:
                pass

        # Write primary artifacts
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

        # === Semi-raw triplets ===
        ocr_map = (
            {o["element_index"]: o["text"] for o in ocr_results} if ocr_results else {}
        )
        triplets = []
        for idx, el in enumerate(all_elements):
            lt_rb = _xywh_to_lt_rb(el["rect"])
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
                for row in triplets:
                    import json as _json

                    f.write(_json.dumps(row, ensure_ascii=False) + "\n")

        ctx.close()
        browser.close()
        return record
