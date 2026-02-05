from dataclasses import dataclass, field


@dataclass
class Viewport:
    width: int = 1440
    height: int = 900
    device_scale_factor: float = 2.0


@dataclass
class FilterConfig:
    """Configuration for element filtering."""

    # IoU threshold for duplicate detection (0-1, higher = more lenient)
    iou_threshold: float = 0.95  # Very conservative
    # Containment threshold for parent/child filtering (0-1)
    containment_threshold: float = 0.98  # Very conservative
    # Minimum box size in pixels
    min_box_size: int = 4  # Very permissive
    # Maximum box size in pixels
    max_box_size: int = (Viewport.width * Viewport.height) // 2  # Half of viewport area
    # Minimum fraction of box that must be visible in viewport
    min_viewport_overlap: float = 0.01  # Nearly anything visible counts
    # Save unfiltered elements for debugging
    save_unfiltered: bool = True


@dataclass
class Timeouts:
    """Timeout configuration for navigation and operations."""

    navigation_ms: int = 30000  # 30 seconds for page navigation
    operation_ms: int = 30000  # 30 seconds for other operations


@dataclass
class Config:
    viewport: Viewport = field(default_factory=Viewport)
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    timeouts: Timeouts = field(default_factory=Timeouts)
    locale: str = "en-US"
    color_scheme: str = "light"
    network_idle_wait_ms: int = 800
    headless: bool = True
    out_dir: str = "data/raw"
    viz_dir: str = "data/viz"
    do_ocr: bool = False
    capture_full_page: bool = False
    user_agent: str = None  # Optional custom user agent
