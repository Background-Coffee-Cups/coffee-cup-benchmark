"""Tests for utility functions."""

import pytest
from src.utils import get_default_config, load_config


class TestConfig:
    def test_default_config_structure(self):
        """Default config should have expected keys."""
        config = get_default_config()
        assert "detector" in config
        assert "evaluator" in config
        assert "benchmark" in config
        assert config["detector"]["yolo_model"] == "yolov8x.pt"

    def test_load_config_missing_file(self):
        """Loading a missing config should return defaults."""
        config = load_config("/nonexistent/path.yaml")
        assert config == get_default_config()
