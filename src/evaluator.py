"""
Cup Quality Evaluator Module

Evaluates detected cup regions on 7 metrics:
- Semantic coherence (CLIP-based)
- Visual resolution (sharpness, detail)
- Structural quality (cup-like features)
- Artifact detection
- Color coherence
- Edge quality
- Detection confidence
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional
import clip
import logging

logger = logging.getLogger(__name__)


class CupQualityEvaluator:
    """Comprehensive quality evaluator for detected cup regions."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        # Load CLIP for semantic understanding
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

        # Load reference cup embeddings
        self.reference_embeddings = self._load_reference_embeddings()

    def _load_reference_embeddings(self):
        """Create embeddings for what good coffee cups should look like."""
        reference_descriptions = [
            "a clear, detailed coffee mug with a visible handle",
            "a well-defined ceramic coffee cup",
            "a realistic coffee mug with proper proportions",
            "a cup with clear rim and proper cylindrical shape",
            "a coffee cup with coherent texture and structure",
        ]

        text_tokens = clip.tokenize(reference_descriptions).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def evaluate_cup_region(
        self, image_path: str, bbox: list, detection_info: Dict
    ) -> Dict:
        """
        Comprehensive evaluation of a detected cup region.

        Args:
            image_path: Path to full image.
            bbox: [x1, y1, x2, y2] bounding box.
            detection_info: Metadata from detection phase.

        Returns:
            Dictionary of quality metrics.
        """
        img = Image.open(image_path).convert("RGB")
        cup_crop = img.crop(bbox)

        if cup_crop.size[0] < 20 or cup_crop.size[1] < 20:
            return {"error": "region_too_small", "overall_quality": 0.0}

        scores = {}

        scores["detection_confidence"] = detection_info["confidence"]
        scores["semantic_quality"] = self._evaluate_semantic_quality(cup_crop)
        scores["visual_resolution"] = self._evaluate_visual_resolution(cup_crop)
        scores["structural_quality"] = self._evaluate_structure(cup_crop)
        scores["artifact_score"] = self._detect_artifacts(cup_crop)
        scores["color_coherence"] = self._evaluate_color_coherence(cup_crop)
        scores["edge_quality"] = self._evaluate_edges(cup_crop)

        weights = {
            "detection_confidence": 0.15,
            "semantic_quality": 0.25,
            "visual_resolution": 0.20,
            "structural_quality": 0.15,
            "artifact_score": 0.10,
            "color_coherence": 0.08,
            "edge_quality": 0.07,
        }

        scores["overall_quality"] = sum(
            scores[k] * weights[k] for k in weights.keys()
        )

        return scores

    def _evaluate_semantic_quality(self, cup_crop: Image.Image) -> float:
        """How cup-like is this region according to CLIP?"""
        img_tensor = self.clip_preprocess(cup_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(img_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ self.reference_embeddings.T).squeeze(0)
            avg_similarity = similarities.mean().item()

        return (avg_similarity + 1) / 2

    def _evaluate_visual_resolution(self, cup_crop: Image.Image) -> float:
        """Measure sharpness and detail level."""
        img_array = np.array(cup_crop.convert("L"))

        # Laplacian variance (sharpness)
        laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
        sharpness = laplacian.var()

        # High-frequency content (detail)
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        center_h = h // 2

        high_freq = (
            magnitude[0 : center_h // 2, :].sum()
            + magnitude[center_h + center_h // 2 :, :].sum()
        )
        total = magnitude.sum()
        hf_ratio = high_freq / (total + 1e-6)

        # Gradient magnitude
        grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2).mean()

        sharpness_norm = min(1.0, sharpness / 500)
        hf_norm = min(1.0, hf_ratio * 5)
        gradient_norm = min(1.0, gradient_mag / 50)

        return sharpness_norm * 0.4 + hf_norm * 0.3 + gradient_norm * 0.3

    def _evaluate_structure(self, cup_crop: Image.Image) -> float:
        """Check for cup-specific structural features."""
        img_array = np.array(cup_crop)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        structure_score = 0.0

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

        if 0.05 < edge_density < 0.25:
            structure_score += 0.3

        # Circular/elliptical shapes (rim)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=int(min(gray.shape) * 0.1),
            maxRadius=int(min(gray.shape) * 0.8),
        )

        if circles is not None and len(circles[0]) > 0:
            structure_score += 0.4

        # Vertical symmetry
        h, w = gray.shape
        left_half = gray[:, : w // 2]
        right_half = cv2.flip(gray[:, w // 2 :], 1)

        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]

        symmetry = 1 - (
            np.abs(left_half.astype(float) - right_half.astype(float)).mean() / 255
        )

        if symmetry > 0.7:
            structure_score += 0.3

        return min(1.0, structure_score)

    def _detect_artifacts(self, cup_crop: Image.Image) -> float:
        """Detect common AI generation artifacts. Higher score = fewer artifacts."""
        img_array = np.array(cup_crop)
        artifact_penalties = 0.0

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Very uniform gradients = suspicious
        gradient_std = gradient_mag.std()
        if gradient_std < 10:
            artifact_penalties += 0.2

        # Color banding (posterization)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_peaks = (hist > hist.mean() * 2).sum()
        if hist_peaks < 20:
            artifact_penalties += 0.15

        # Unnatural color blending
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hue_variance = hsv[:, :, 0].var()
        if hue_variance < 50:
            artifact_penalties += 0.1

        # Checkerboard artifacts (periodic patterns in FFT)
        f_transform = np.fft.fft2(gray)
        f_magnitude = np.abs(f_transform)

        f_sorted = np.sort(f_magnitude.flatten())
        if f_sorted[-10] / (f_sorted[-100] + 1) > 5:
            artifact_penalties += 0.15

        return max(0.0, 1.0 - artifact_penalties)

    def _evaluate_color_coherence(self, cup_crop: Image.Image) -> float:
        """Check if colors are realistic and coherent."""
        img_array = np.array(cup_crop)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        l_var = lab[:, :, 0].var()
        a_var = lab[:, :, 1].var()
        b_var = lab[:, :, 2].var()

        l_score = 1.0 - abs(l_var - 400) / 400 if l_var < 800 else 0.3
        ab_score = 1.0 - abs((a_var + b_var) / 2 - 200) / 200

        coherence = l_score * 0.6 + ab_score * 0.4
        return max(0.0, min(1.0, coherence))

    def _evaluate_edges(self, cup_crop: Image.Image) -> float:
        """Evaluate edge quality (clean vs blurry/artifacted)."""
        img_array = np.array(cup_crop.convert("L"))

        edges = cv2.Canny(img_array, 50, 150)

        # Thin edges are better
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        edge_thickness = (dilated.sum() - edges.sum()) / (edges.sum() + 1)
        thickness_score = max(0, 1.0 - edge_thickness)

        # Edge connectivity
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            connectivity_score = min(1.0, areas.max() / (areas.sum() + 1))
        else:
            connectivity_score = 0.5

        return thickness_score * 0.6 + connectivity_score * 0.4
