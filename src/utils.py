"""
Utility functions for the Coffee Cup Benchmark.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the benchmark."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict:
    """Load benchmark configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return get_default_config()

    with open(path) as f:
        return yaml.safe_load(f)


def get_default_config() -> Dict:
    """Return default benchmark configuration."""
    return {
        "detector": {
            "yolo_model": "yolov8x.pt",
            "yolo_confidence": 0.15,
            "owl_model": "google/owlvit-base-patch32",
            "owl_confidence": 0.15,
            "nms_threshold": 0.5,
        },
        "evaluator": {
            "clip_model": "ViT-B/32",
            "device": "auto",
        },
        "benchmark": {
            "save_visualizations": True,
            "visualization_dpi": 150,
        },
    }
