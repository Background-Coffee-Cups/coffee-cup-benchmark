"""
Benchmark Pipeline Module

Orchestrates detection -> evaluation -> reporting pipeline.
Supports single image, batch processing, and model comparison.
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

from src.detector import CoffeeCupDetector
from src.evaluator import CupQualityEvaluator

logger = logging.getLogger(__name__)


class CoffeeCupBenchmark:
    """Complete benchmark pipeline for coffee cup background resolution."""

    def __init__(self, device: Optional[str] = None):
        self.detector = CoffeeCupDetector(device=device)
        self.evaluator = CupQualityEvaluator(device=device)

    def run_single_image(
        self, image_path: str, save_visualization: bool = True
    ) -> Dict:
        """
        Complete pipeline for one image:
        1. Detect cups
        2. Filter to background cups
        3. Evaluate each cup
        4. Return aggregate scores
        """
        detections = self.detector.detect_cups(image_path)
        bg_cups = [d for d in detections if d["depth_class"] == "background"]

        if not bg_cups:
            return {
                "image": str(image_path),
                "num_bg_cups_detected": 0,
                "detection_success": False,
                "avg_quality": 0.0,
                "message": "No background cups detected",
            }

        cup_evaluations = []
        for i, cup_det in enumerate(bg_cups):
            try:
                scores = self.evaluator.evaluate_cup_region(
                    image_path, cup_det["bbox"], cup_det
                )
                scores["cup_id"] = i
                scores["bbox"] = cup_det["bbox"]
                cup_evaluations.append(scores)
            except Exception as e:
                logger.warning(f"Error evaluating cup {i}: {e}")
                continue

        if not cup_evaluations:
            return {
                "image": str(image_path),
                "num_bg_cups_detected": len(bg_cups),
                "detection_success": True,
                "avg_quality": 0.0,
                "message": "Cups detected but evaluation failed",
            }

        result = {
            "image": str(image_path),
            "timestamp": datetime.utcnow().isoformat(),
            "num_bg_cups_detected": len(bg_cups),
            "detection_success": True,
            "cups": cup_evaluations,
            "avg_quality": float(
                np.mean([c["overall_quality"] for c in cup_evaluations])
            ),
            "max_quality": float(
                np.max([c["overall_quality"] for c in cup_evaluations])
            ),
            "avg_detection_conf": float(
                np.mean([c["detection_confidence"] for c in cup_evaluations])
            ),
            "avg_semantic_quality": float(
                np.mean([c["semantic_quality"] for c in cup_evaluations])
            ),
            "avg_visual_resolution": float(
                np.mean([c["visual_resolution"] for c in cup_evaluations])
            ),
            "avg_structural_quality": float(
                np.mean([c["structural_quality"] for c in cup_evaluations])
            ),
            "avg_artifact_score": float(
                np.mean([c["artifact_score"] for c in cup_evaluations])
            ),
        }

        if save_visualization:
            self._visualize_results(image_path, bg_cups, cup_evaluations)

        return result

    def _visualize_results(
        self, image_path: str, detections: List[Dict], evaluations: List[Dict]
    ):
        """Create annotated image showing detections and scores."""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for det, eval_scores in zip(detections, evaluations):
            x1, y1, x2, y2 = det["bbox"]
            quality = eval_scores["overall_quality"]

            if quality > 0.7:
                color = (0, 255, 0)
            elif quality > 0.4:
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"Q: {quality:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        output_path = Path(image_path).stem + "_annotated.jpg"
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Background Cup Detection & Quality")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def run_batch(
        self, image_dir: str, output_file: str = "benchmark_results.json"
    ) -> List[Dict]:
        """Run benchmark on all images in directory."""
        image_paths = list(Path(image_dir).glob("*.png")) + list(
            Path(image_dir).glob("*.jpg")
        )

        all_results = []

        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.run_single_image(str(img_path), save_visualization=True)
                all_results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        self._generate_report(all_results)

        return all_results

    def _generate_report(self, results: List[Dict]):
        """Generate comprehensive benchmark report."""
        successful = [r for r in results if r["detection_success"]]

        if not successful:
            logger.warning("No successful detections in batch.")
            return

        print("\n" + "=" * 60)
        print("COFFEE CUP BACKGROUND RESOLUTION BENCHMARK")
        print("=" * 60)

        print(f"\nDETECTION STATISTICS")
        print(f"  Total images processed: {len(results)}")
        print(
            f"  Images with background cups: {len(successful)} "
            f"({len(successful)/len(results)*100:.1f}%)"
        )
        print(
            f"  Total background cups found: "
            f"{sum(r['num_bg_cups_detected'] for r in successful)}"
        )
        print(
            f"  Avg cups per image: "
            f"{np.mean([r['num_bg_cups_detected'] for r in successful]):.2f}"
        )

        print(f"\nQUALITY METRICS (0.0 - 1.0 scale)")
        metrics = {
            "Overall Quality": "avg_quality",
            "Detection Confidence": "avg_detection_conf",
            "Semantic Quality": "avg_semantic_quality",
            "Visual Resolution": "avg_visual_resolution",
            "Structural Quality": "avg_structural_quality",
            "Artifact-Free Score": "avg_artifact_score",
        }

        for name, key in metrics.items():
            values = [r[key] for r in successful if key in r]
            if values:
                print(
                    f"  {name:.<25} {np.mean(values):.3f} +/- {np.std(values):.3f}"
                )

        qualities = [r["avg_quality"] for r in successful]
        print(f"\nQUALITY DISTRIBUTION")
        excellent = sum(1 for q in qualities if q > 0.7)
        good = sum(1 for q in qualities if 0.5 <= q <= 0.7)
        poor = sum(1 for q in qualities if q < 0.5)
        total = len(qualities)
        print(f"  Excellent (>0.7): {excellent} ({excellent/total*100:.1f}%)")
        print(f"  Good (0.5-0.7):  {good} ({good/total*100:.1f}%)")
        print(f"  Poor (<0.5):     {poor} ({poor/total*100:.1f}%)")

        best = max(successful, key=lambda x: x["avg_quality"])
        worst = min(successful, key=lambda x: x["avg_quality"])

        print(
            f"\n  BEST: {Path(best['image']).name} "
            f"(Quality: {best['avg_quality']:.3f})"
        )
        print(
            f"  WORST: {Path(worst['image']).name} "
            f"(Quality: {worst['avg_quality']:.3f})"
        )

        self._create_plots(successful)

    def _create_plots(self, results: List[Dict]):
        """Generate visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        qualities = [r["avg_quality"] for r in results]

        # Quality distribution histogram
        axes[0, 0].hist(qualities, bins=20, edgecolor="black", alpha=0.7)
        axes[0, 0].axvline(
            np.mean(qualities),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(qualities):.3f}",
        )
        axes[0, 0].set_xlabel("Overall Quality Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Quality Score Distribution")
        axes[0, 0].legend()

        # Metric comparison
        metric_keys = [
            "avg_detection_conf",
            "avg_semantic_quality",
            "avg_visual_resolution",
            "avg_structural_quality",
            "avg_artifact_score",
        ]
        metric_names = ["Detection", "Semantic", "Resolution", "Structure", "Artifacts"]
        metric_means = [
            np.mean([r[m] for r in results if m in r]) for m in metric_keys
        ]

        axes[0, 1].barh(metric_names, metric_means, color="skyblue", edgecolor="black")
        axes[0, 1].set_xlabel("Score")
        axes[0, 1].set_title("Average Metric Scores")
        axes[0, 1].set_xlim(0, 1)

        # Detection count vs quality
        num_cups = [r["num_bg_cups_detected"] for r in results]
        axes[1, 0].scatter(num_cups, qualities, alpha=0.6)
        axes[1, 0].set_xlabel("Number of Background Cups Detected")
        axes[1, 0].set_ylabel("Average Quality Score")
        axes[1, 0].set_title("Detection Count vs Quality")

        # Correlation matrix
        df = pd.DataFrame(
            [
                {
                    k: r[k]
                    for k in [
                        "avg_quality",
                        "avg_detection_conf",
                        "avg_semantic_quality",
                        "avg_visual_resolution",
                        "avg_structural_quality",
                    ]
                    if k in r
                }
                for r in results
            ]
        )

        if not df.empty:
            corr = df.corr()
            sns.heatmap(
                corr,
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=axes[1, 1],
                xticklabels=[
                    "Quality",
                    "Detection",
                    "Semantic",
                    "Resolution",
                    "Structure",
                ],
                yticklabels=[
                    "Quality",
                    "Detection",
                    "Semantic",
                    "Resolution",
                    "Structure",
                ],
            )
        axes[1, 1].set_title("Metric Correlation Matrix")

        plt.tight_layout()
        plt.savefig("benchmark_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"\n  Saved analysis plots to: benchmark_analysis.png")

    def export_submission(self, results: List[Dict], model_name: str) -> Dict:
        """
        Format results for leaderboard submission.

        Returns a JSON-serializable dict ready for the web API.
        """
        successful = [r for r in results if r["detection_success"]]

        if not successful:
            raise ValueError("No successful detections to submit.")

        qualities = [r["avg_quality"] for r in successful]

        submission = {
            "model_name": model_name,
            "submitted_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "summary": {
                "total_images": len(results),
                "detection_rate": len(successful) / len(results),
                "avg_quality": float(np.mean(qualities)),
                "median_quality": float(np.median(qualities)),
                "std_quality": float(np.std(qualities)),
                "max_quality": float(np.max(qualities)),
                "min_quality": float(np.min(qualities)),
                "excellent_pct": sum(1 for q in qualities if q > 0.7) / len(qualities),
                "good_pct": sum(1 for q in qualities if 0.5 <= q <= 0.7)
                / len(qualities),
                "poor_pct": sum(1 for q in qualities if q < 0.5) / len(qualities),
            },
            "metrics": {
                "avg_detection_conf": float(
                    np.mean([r["avg_detection_conf"] for r in successful])
                ),
                "avg_semantic_quality": float(
                    np.mean([r["avg_semantic_quality"] for r in successful])
                ),
                "avg_visual_resolution": float(
                    np.mean([r["avg_visual_resolution"] for r in successful])
                ),
                "avg_structural_quality": float(
                    np.mean([r["avg_structural_quality"] for r in successful])
                ),
                "avg_artifact_score": float(
                    np.mean([r["avg_artifact_score"] for r in successful])
                ),
            },
            "per_image": [
                {
                    "image": r["image"],
                    "num_cups": r["num_bg_cups_detected"],
                    "avg_quality": r["avg_quality"],
                }
                for r in successful
            ],
        }

        return submission
