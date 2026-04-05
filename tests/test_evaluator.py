"""Tests for the CupQualityEvaluator module."""

import numpy as np
import pytest
from PIL import Image


class TestEvaluatorMetrics:
    """Test individual evaluation metrics with synthetic images."""

    def _make_test_image(self, w=100, h=100, color=(128, 128, 128)):
        """Create a simple test image."""
        img = Image.new("RGB", (w, h), color)
        return img

    def test_tiny_region_returns_error(self):
        """Regions smaller than 20x20 should return error."""
        from src.evaluator import CupQualityEvaluator

        evaluator = CupQualityEvaluator.__new__(CupQualityEvaluator)

        # Create a tiny temp image
        img = self._make_test_image(10, 10)
        img.save("/tmp/test_tiny.png")

        result = evaluator.evaluate_cup_region(
            "/tmp/test_tiny.png", [0, 0, 10, 10], {"confidence": 0.5}
        )
        assert result["error"] == "region_too_small"
        assert result["overall_quality"] == 0.0

    def test_visual_resolution_uniform_image(self):
        """A uniform image should have low visual resolution score."""
        from src.evaluator import CupQualityEvaluator

        evaluator = CupQualityEvaluator.__new__(CupQualityEvaluator)
        img = self._make_test_image(100, 100, (128, 128, 128))
        score = evaluator._evaluate_visual_resolution(img)
        assert 0.0 <= score <= 0.3  # Uniform = low detail

    def test_color_coherence_range(self):
        """Color coherence should return value in [0, 1]."""
        from src.evaluator import CupQualityEvaluator

        evaluator = CupQualityEvaluator.__new__(CupQualityEvaluator)
        img = self._make_test_image(100, 100, (200, 150, 100))
        score = evaluator._evaluate_color_coherence(img)
        assert 0.0 <= score <= 1.0

    def test_edge_quality_range(self):
        """Edge quality should return value in [0, 1]."""
        from src.evaluator import CupQualityEvaluator

        evaluator = CupQualityEvaluator.__new__(CupQualityEvaluator)
        img = self._make_test_image(100, 100)
        score = evaluator._evaluate_edges(img)
        assert 0.0 <= score <= 1.0
