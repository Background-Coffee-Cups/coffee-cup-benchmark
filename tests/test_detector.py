"""Tests for the CoffeeCupDetector module."""

import numpy as np
import pytest


class TestNonMaxSuppression:
    """Test NMS without requiring model downloads."""

    def setup_method(self):
        # Import just the method logic — detector init requires model files
        pass

    def test_nms_empty_input(self):
        """NMS with no detections returns empty list."""
        from src.detector import CoffeeCupDetector

        # Test the static-like NMS method
        detector = CoffeeCupDetector.__new__(CoffeeCupDetector)
        result = detector._non_max_suppression([], iou_threshold=0.5)
        assert result == []

    def test_nms_single_detection(self):
        """NMS with one detection returns it unchanged."""
        detector = CoffeeCupDetector.__new__(CoffeeCupDetector)
        detections = [{"bbox": [10, 10, 50, 50], "confidence": 0.9}]
        result = detector._non_max_suppression(detections, iou_threshold=0.5)
        assert len(result) == 1

    def test_nms_removes_overlapping(self):
        """NMS removes highly overlapping lower-confidence detections."""
        from src.detector import CoffeeCupDetector

        detector = CoffeeCupDetector.__new__(CoffeeCupDetector)
        detections = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.9},
            {"bbox": [12, 12, 52, 52], "confidence": 0.7},  # Overlaps heavily
        ]
        result = detector._non_max_suppression(detections, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_nms_keeps_non_overlapping(self):
        """NMS keeps detections that don't overlap."""
        from src.detector import CoffeeCupDetector

        detector = CoffeeCupDetector.__new__(CoffeeCupDetector)
        detections = [
            {"bbox": [10, 10, 50, 50], "confidence": 0.9},
            {"bbox": [200, 200, 250, 250], "confidence": 0.8},
        ]
        result = detector._non_max_suppression(detections, iou_threshold=0.5)
        assert len(result) == 2


class TestDepthClassification:
    """Test foreground/background classification."""

    def test_small_upper_object_is_background(self):
        """Small object in upper portion should be classified as background."""
        from src.detector import CoffeeCupDetector

        detector = CoffeeCupDetector.__new__(CoffeeCupDetector)
        detections = [{"bbox": [100, 50, 140, 90]}]  # Small, upper region
        result = detector._classify_depth(detections, (1000, 1000))
        assert result[0]["depth_class"] == "background"

    def test_large_lower_object_is_foreground(self):
        """Large object in lower portion should be classified as foreground."""
        from src.detector import CoffeeCupDetector

        detector = CoffeeCupDetector.__new__(CoffeeCupDetector)
        detections = [{"bbox": [100, 600, 400, 900]}]  # Large, lower region
        result = detector._classify_depth(detections, (1000, 1000))
        assert result[0]["depth_class"] == "foreground"
