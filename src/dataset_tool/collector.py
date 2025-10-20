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

        page.goto(url_c, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(cfg.network_idle_wait_ms)

        # Scroll to trigger lazy loads
        page.evaluate(
            """() => new Promise(res => {
            let y=0; const step=window.innerHeight*0.8;
            function s(){ y+=step; window.scrollTo({top:y,behavior:'instant'});
              if(y<document.body.scrollHeight-2*window.innerHeight) requestAnimationFrame(s); else res(); }
            s();
        })"""
        )
        page.wait_for_timeout(400)

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
                # Robust strategy: querySelectorAll('*') then filter for visibility
                els = fr.evaluate(
                    """() => {
                    function visible(el){
                      if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
                      const s = getComputedStyle(el);
                      if (s.display==='none' || s.visibility==='hidden' || parseFloat(s.opacity)===0) return false;
                      const r = el.getBoundingClientRect();
                      return r.width>0 && r.height>0;
                    }
                    const out=[];
                    const nodes = Array.from(document.querySelectorAll('*'));
                    for (const n of nodes){
                      if(!visible(n)) continue;
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
                }"""
                )
                # normalize + page-relative
                for e in els:
                    e["frame_index"] = fr_idx
                    e["type"] = guess_type(e["tag"], e["role"], e.get("attrs", {}))
                    e["rect"]["x"] = int(round(e["rect"]["x"])) + off["x"]
                    e["rect"]["y"] = int(round(e["rect"]["y"])) + off["y"]
                    e["rect"] = rect_to_int(e["rect"])
                all_elements.extend(els)
            except Exception as _:
                # keep going; text boxes might still work
                pass

            # === TEXT SPANS ===
            try:
                tboxes = fr.evaluate(
                    """() => {
                  const res=[];
                  const walker=document.createTreeWalker(document, NodeFilter.SHOW_TEXT);
                  let n=walker.nextNode();  // correct start
                  while(n){
                    const s = (n.nodeValue || '').replace(/\\s+/g,' ').trim();
                    if(s){
                      const range = document.createRange();
                      range.selectNodeContents(n);
                      const rects = Array.from(range.getClientRects());
                      for(const r of rects){
                        if(r.width>0 && r.height>0){
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
                }"""
                )
                for t in tboxes:
                    t["frame_index"] = fr_idx
                    t["rect"]["x"] = int(round(t["rect"]["x"])) + off["x"]
                    t["rect"]["y"] = int(round(t["rect"]["y"])) + off["y"]
                    t["rect"] = rect_to_int(t["rect"])
                all_texts.extend(tboxes)
            except Exception:
                pass

        # AX tree (unchanged)
        ax_tree = page.accessibility.snapshot(root=None, interesting_only=False)

        # Screenshot full page
        img_path = f"{base}.png"
        page.screenshot(path=img_path, full_page=True)

        meta = {
            "url": url_c,
            "ts": int(time.time()),
            "viewport": {
                "w": cfg.viewport.width,
                "h": cfg.viewport.height,
                "dpr": cfg.viewport.device_scale_factor,
            },
            "browser": ctx.browser.version,
            "frames": len(frames),
            "scroll_height": page.evaluate("() => document.body.scrollHeight"),
        }

        # Optional OCR per element (simple)
        ocr_results: List[Dict[str, Any]] = []
        if cfg.do_ocr and _has_tesseract():
            try:
                import cv2, pytesseract

                img = cv2.imread(img_path)
                for idx, el in enumerate(all_elements):
                    x, y, w, h = (
                        el["rect"]["x"],
                        el["rect"]["y"],
                        el["rect"]["w"],
                        el["rect"]["h"],
                    )
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

        # === Semi-raw triplets (text, [l,t,r,b], type) ===
        # Prefer DOM element inner_text; fallback to OCR for that element index; else ''.
        # Always export page-relative [l,t,r,b].
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
                    "bbox_ltrb": lt_rb,  # [l, t, r, b] in CSS px
                    "type": el.get("type") or "unknown",
                    "text": text,
                    "tag": el.get("tag"),
                    "role": el.get("role"),
                }
            )
        if triplets:
            record["triplets_path"] = f"{base}.triplets.jsonl"
            # write JSONL
            with open(record["triplets_path"], "w", encoding="utf-8") as f:
                for row in triplets:
                    # tiny hand-rolled writer to avoid another dependency
                    import json as _json

                    f.write(_json.dumps(row, ensure_ascii=False) + "\n")

        ctx.close()
        browser.close()
        return record
