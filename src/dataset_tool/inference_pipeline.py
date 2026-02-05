import os
import cv2
import torch
import pytesseract
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
from tqdm import tqdm

from .hierarchy_reconstruct import reconstruct_hierarchy, compute_own_text_for_all
from .utils import elements_to_screentag, save_json, ensure_dir

def auto_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"

class InferencePipeline:
    def __init__(self, weights_path: str, device: str = None, conf: float = 0.25, iou: float = 0.7, ocr_engine: str = "easyocr", imgsz: int = 1280):
        """
        Initialize the inference pipeline.
        """
        self.device = device or auto_device()
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        self.ocr_engine = ocr_engine
        self.imgsz = imgsz
        self.reader = None
        self.has_tesseract = False
        
        # Initialize OCR engine
        if self.ocr_engine == "easyocr":
            try:
                import easyocr
                print(f"Initializing EasyOCR on {self.device}...")
                # EasyOCR expects 'cuda' or 'cpu' (or True/False for gpu)
                use_gpu = self.device != "cpu"
                self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            except ImportError:
                print("Warning: EasyOCR not found. Install with `pip install easyocr`. Falling back to Tesseract.")
                self.ocr_engine = "tesseract"
        
        if self.ocr_engine == "tesseract":
            import shutil
            self.has_tesseract = shutil.which("tesseract") is not None
            if not self.has_tesseract:
                print("Warning: Tesseract not found. OCR will be skipped.")

    def predict_image(self, image_path: str, plot_save_path: str = None) -> List[Dict[str, Any]]:
        """
        Run YOLO inference on a single image.
        Returns a list of detections in the format expected by hierarchy reconstruction.
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            imgsz=self.imgsz,
            max_det=500
        )
        
        if not results:
            return []
            
        result = results[0]
        
        if plot_save_path:
            ensure_dir(os.path.dirname(plot_save_path))
            # plot() returns a BGR numpy array
            im_array = result.plot()
            cv2.imwrite(plot_save_path, im_array)

        elements = []
        
        # Get class names
        names = result.names
        
        for box in result.boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = names[cls_id]
            
            # Convert to x, y, w, h format
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            element = {
                "rect": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                },
                "vlm_label": label,
                "type": label,
                "confidence": conf,
                "inner_text": "" # Placeholder for OCR
            }
            elements.append(element)
            
        return elements

    def run_ocr(self, image_path: str, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run OCR on the detected elements.
        """
        if self.ocr_engine == "tesseract" and not self.has_tesseract:
            return elements
        if self.ocr_engine == "easyocr" and self.reader is None:
            return elements
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image {image_path}")
                return elements
                
            for el in elements:
                rect = el["rect"]
                x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
                
                # Clamp coordinates
                h_img, w_img = img.shape[:2]
                x = max(0, min(x, w_img - 1))
                y = max(0, min(y, h_img - 1))
                w = max(1, min(w, w_img - x))
                h = max(1, min(h, h_img - y))
                
                crop = img[y : y + h, x : x + w]
                
                if crop.size > 0:
                    txt = ""
                    if self.ocr_engine == "easyocr":
                        try:
                            # detail=0 returns list of strings
                            # paragraph=True combines them into lines
                            results = self.reader.readtext(crop, detail=0, paragraph=True)
                            txt = " ".join(results).strip()
                        except Exception:
                            pass
                    elif self.ocr_engine == "tesseract":
                        txt = pytesseract.image_to_string(crop).strip()
                    
                    el["inner_text"] = txt
                    
        except Exception as e:
            print(f"OCR failed for {image_path}: {e}")
            
        return elements

    def process_single(self, image_path: str, output_path: Optional[str] = None, save_json_elements: bool = True, save_viz: bool = True) -> str:
        """
        Run the full pipeline on a single image:
        Inference -> OCR -> Hierarchy -> ScreenTag
        """
        # Determine viz path if needed
        viz_path = None
        if save_viz and output_path:
            if output_path.endswith(".screentag.txt"):
                viz_path = output_path.replace(".screentag.txt", ".viz.png")
            else:
                viz_path = os.path.splitext(output_path)[0] + ".viz.png"

        # 1. Inference
        elements = self.predict_image(image_path, plot_save_path=viz_path)
        
        # 2. OCR
        elements = self.run_ocr(image_path, elements)
        
        # 3. Hierarchy Reconstruction
        # We need to know the image size for some calculations, but reconstruct_hierarchy mainly uses rects
        # However, elements_to_screentag needs viewport dimensions
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            viewport = {"w": w, "h": h}
        else:
            # Fallback if image read fails (shouldn't happen if OCR worked)
            viewport = {"w": 1920, "h": 1080} 

        elements = reconstruct_hierarchy(elements, min_containment=0.90, use_semantic_hints=True)
        
        # 3.5 Compute own_text (deduplicate text)
        elements = compute_own_text_for_all(elements)
        
        # 4. ScreenTag Conversion
        screentag = elements_to_screentag(elements, viewport)
        
        # 5. Save
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(screentag)
            
            if save_json_elements:
                # Determine JSON path
                if output_path.endswith(".screentag.txt"):
                    json_path = output_path.replace(".screentag.txt", ".json")
                else:
                    json_path = os.path.splitext(output_path)[0] + ".json"
                
                # Refine elements to only include requested attributes
                refined_elements = []
                for el in elements:
                    refined_elements.append({
                        "bbox": [el["rect"]["x"], el["rect"]["y"], el["rect"]["w"], el["rect"]["h"]],
                        "type": el["type"],
                        "text": el.get("own_text") or el.get("inner_text", "")
                    })
                
                save_json(json_path, refined_elements)
                
        return screentag

def process_directory(
    pipeline: InferencePipeline, 
    input_dir: str, 
    output_dir: str, 
    extensions: set = {".png", ".jpg", ".jpeg"},
    save_json_elements: bool = True,
    save_viz: bool = True
):
    ensure_dir(output_dir)
    files = [
        f for f in os.listdir(input_dir) 
        if os.path.splitext(f)[1].lower() in extensions
    ]
    
    for f in tqdm(files, desc="Processing images"):
        img_path = os.path.join(input_dir, f)
        out_name = os.path.splitext(f)[0] + ".screentag.txt"
        out_path = os.path.join(output_dir, out_name)
        
        try:
            pipeline.process_single(img_path, out_path, save_json_elements=save_json_elements, save_viz=save_viz)
        except Exception as e:
            print(f"Failed to process {f}: {e}")

