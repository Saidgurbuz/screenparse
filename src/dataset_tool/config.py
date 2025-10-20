from dataclasses import dataclass


@dataclass
class Viewport:
    width: int = 1440
    height: int = 900
    device_scale_factor: float = 2.0


@dataclass
class Config:
    viewport: Viewport = Viewport()
    locale: str = "en-US"
    color_scheme: str = "light"
    network_idle_wait_ms: int = 800
    headless: bool = True
    out_dir: str = "data/raw"
    viz_dir: str = "data/viz"
    do_ocr: bool = False  # set True to run tesseract OCR per element
