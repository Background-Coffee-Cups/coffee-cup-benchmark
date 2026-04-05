"""
Coffee Cup Detection Module

Multi-model detection approach:
- YOLO for standard object detection
- OWL-ViT for open-vocabulary detection
- NMS for deduplication
- Depth classification (foreground vs background)
"""

import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CoffeeCupDetector:
    """Multi-model coffee cup detector with depth classification."""

    def __init__(self, yolo_model: str = "yolov8x.pt", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        logger.info(f"Initializing detector on {device}")

        # YOLO model
        self.yolo = YOLO(yolo_model)

        # OWL-ViT for open-vocabulary detection
        try:
            self.owl_detector = pipeline(
                "zero-shot-object-detection",
                model="google/owlvit-base-patch32",
                device=0 if device == "cuda" else -1,
            )
            self.owl_available = True
        except Exception as e:
            logger.warning(f"OWL-ViT unavailable: {e}")
            self.owl_available = False

        # COCO class IDs for cup-like objects
        self.cup_classes = [41, 47]  # cup, bowl

    def detect_cups(self, image_path: str) -> List[Dict]:
        """
        Detect all cup-like objects with multiple methods.

        Returns:
            List of detections with bounding boxes, confidence, and depth class.
        """
        img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        all_detections = []

        # Method 1: YOLO detection
        yolo_results = self.yolo(img, conf=0.15, verbose=False)[0]

        for box in yolo_results.boxes:
            cls = int(box.cls[0])
            if cls in self.cup_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                all_detections.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf[0]),
                        "method": "yolo",
                        "class": cls,
                    }
                )

        # Method 2: OWL-ViT for open-vocab detection
        if self.owl_available:
            try:
                owl_results = self.owl_detector(
                    img,
                    candidate_labels=[
                        "coffee cup",
                        "coffee mug",
                        "tea cup",
                        "ceramic mug",
                        "glass cup",
                        "cup with handle",
                    ],
                )

                for det in owl_results:
                    if det["score"] > 0.15:
                        box = det["box"]
                        all_detections.append(
                            {
                                "bbox": [
                                    box["xmin"],
                                    box["ymin"],
                                    box["xmax"],
                                    box["ymax"],
                                ],
                                "confidence": det["score"],
                                "method": "owl",
                                "label": det["label"],
                            }
                        )
            except Exception as e:
                logger.warning(f"OWL-ViT detection failed: {e}")

        # Merge overlapping detections (NMS)
        detections = self._non_max_suppression(all_detections, iou_threshold=0.5)

        # Classify foreground vs background
        detections = self._classify_depth(detections, (h, w))

        return detections

    def _non_max_suppression(
        self, detections: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Remove duplicate detections via NMS."""
        if not detections:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            order = order[np.where(iou <= iou_threshold)[0] + 1]

        return [detections[i] for i in keep]

    def _classify_depth(
        self, detections: List[Dict], image_shape: tuple
    ) -> List[Dict]:
        """
        Classify detections as foreground or background based on:
        - Position in frame (y-coordinate)
        - Size relative to image
        - Expected perspective
        """
        h, w = image_shape

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            box_w = x2 - x1
            box_h = y2 - y1
            center_y = (y1 + y2) / 2
            size_ratio = (box_w * box_h) / (w * h)

            bg_score = 0

            # Small objects more likely background
            if size_ratio < 0.05:
                bg_score += 2
            elif size_ratio < 0.10:
                bg_score += 1
            elif size_ratio > 0.20:
                bg_score -= 2

            # Upper portion of image = more likely background
            if center_y < h * 0.4:
                bg_score += 2
            elif center_y < h * 0.6:
                bg_score += 1
            else:
                bg_score -= 1

            # Very top corners = likely background
            center_x = (x1 + x2) / 2
            if center_y < h * 0.3 and (center_x < w * 0.3 or center_x > w * 0.7):
                bg_score += 1

            det["depth_class"] = "background" if bg_score > 1 else "foreground"
            det["bg_confidence"] = min(1.0, max(0.0, (bg_score + 2) / 6))

        return detections
