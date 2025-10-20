from typing import List, Dict, Any
import time, os, shutil, subprocess
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

        # Scroll to force lazy loads
        page.evaluate(
            """() => new Promise(res => {
            let y=0; const step=window.innerHeight*0.8;
            function s(){ y+=step; window.scrollTo({top:y,behavior:'instant'}); 
              if(y<document.body.scrollHeight-2*window.innerHeight) requestAnimationFrame(s); else res(); }
            s();
        })"""
        )
        page.wait_for_timeout(400)

        # Collect per frame (main + iframes)
        frames = [page.main_frame] + page.frames[1:]
        all_elements = []
        all_texts = []
        frame_idx = 0
        for fr in frames:
            frame_idx += 1
            try:
                # elements with boxes
                els = fr.evaluate(
                    """() => {
                    function visible(el){
                      const s = getComputedStyle(el);
                      if (s.display==='none' || s.visibility==='hidden' || parseFloat(s.opacity)===0) return false;
                      const r = el.getBoundingClientRect();
                      return r.width>0 && r.height>0;
                    }
                    const res=[];
                    const walker=document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT);
                    let n=walker.currentNode;
                    while(n){
                      if(visible(n)){
                        const r=n.getBoundingClientRect();
                        const attrs={}; for (const a of n.attributes) attrs[a.name]=a.value;
                        res.push({
                          tag: n.tagName.toLowerCase(),
                          role: n.getAttribute('role') || null,
                          id: n.id || null,
                          classes: n.className || null,
                          attrs,
                          rect: {x:r.x, y:r.y, w:r.width, h:r.height},
                          z: Number(getComputedStyle(n).zIndex) || 0,
                          aria_hidden: n.getAttribute('aria-hidden') || null
                        });
                      }
                      n=walker.nextNode();
                    }
                    return res;
                }"""
                )
                for e in els:
                    e["frame_index"] = frame_idx
                    e["type"] = guess_type(e["tag"], e["role"], e.get("attrs", {}))
                    e["rect"] = rect_to_int(e["rect"])
                all_elements.extend(els)

                # per-text-node boxes
                tboxes = fr.evaluate(
                    """() => {
                  const res=[];
                  const walker=document.createTreeWalker(document, NodeFilter.SHOW_TEXT);
                  let n=walker.nextNode();
                  while(n){
                    const s = n.nodeValue.replace(/\\s+/g,' ').trim();
                    if(s){
                      const range = document.createRange();
                      range.selectNodeContents(n);
                      for(const r of Array.from(range.getClientRects())){
                        if(r.width>0 && r.height>0){
                          const parent = range.startContainer.parentElement;
                          const cs = getComputedStyle(parent);
                          res.push({
                            text: s,
                            rect:{x:r.x, y:r.y, w:r.width, h:r.height},
                            font_family: cs.fontFamily,
                            font_size: cs.fontSize,
                            font_weight: cs.fontWeight
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
                    t["frame_index"] = frame_idx
                    t["rect"] = rect_to_int(t["rect"])
                all_texts.extend(tboxes)
            except Exception:
                continue

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

        # Optional OCR per element (cheap pass, may be noisy)
        ocr_results = []
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
                    # guard bounds
                    x2 = max(0, x)
                    y2 = max(0, y)
                    crop = img[y2 : y2 + h, x2 : x2 + w]
                    if crop.size > 0:
                        txt = pytesseract.image_to_string(crop).strip()
                        if txt:
                            ocr_results.append({"element_index": idx, "text": txt})
            except Exception:
                pass

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

        ctx.close()
        browser.close()
        return record
