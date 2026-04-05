#!/usr/bin/env python3
"""
Compare multiple AI models on the Coffee Cup Background Resolution Benchmark.

Usage:
    python scripts/compare_models.py --models dalle3 midjourney stable_diffusion flux
    python scripts/compare_models.py --results-dir ./output/results/
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import CoffeeCupBenchmark
from src.utils import setup_logging


class ModelComparison:
    """Compare benchmark results across multiple models."""

    def compare_from_results(self, results_dir: str) -> pd.DataFrame:
        """Load and compare pre-computed results from a directory."""
        results_dir = Path(results_dir)
        model_results = {}

        for results_file in results_dir.glob("*_results.json"):
            model_name = results_file.stem.replace("_results", "")
            with open(results_file) as f:
                model_results[model_name] = json.load(f)

        if not model_results:
            print(f"No result files found in {results_dir}")
            return pd.DataFrame()

        return self._build_comparison(model_results)

    def compare_from_dirs(self, model_dirs: dict, device: str = None) -> pd.DataFrame:
        """Run benchmarks on image directories and compare."""
        benchmark = CoffeeCupBenchmark(device=device)
        model_results = {}

        for model_name, image_dir in model_dirs.items():
            print(f"\nBenchmarking: {model_name}")
            results = benchmark.run_batch(
                image_dir, output_file=f"output/results/{model_name}_results.json"
            )
            model_results[model_name] = results

        return self._build_comparison(model_results)

    def _build_comparison(self, model_results: dict) -> pd.DataFrame:
        """Build comparison table from model results."""
        rows = []

        for model, results in model_results.items():
            successful = [r for r in results if r.get("detection_success")]
            if not successful:
                rows.append({"model": model, "detection_rate": 0.0})
                continue

            qualities = [r["avg_quality"] for r in successful]

            rows.append(
                {
                    "model": model,
                    "total_images": len(results),
                    "detection_rate": len(successful) / len(results),
                    "avg_quality": np.mean(qualities),
                    "median_quality": np.median(qualities),
                    "std_quality": np.std(qualities),
                    "excellent_pct": sum(1 for q in qualities if q > 0.7)
                    / len(qualities),
                    "avg_semantic": np.mean(
                        [r["avg_semantic_quality"] for r in successful]
                    ),
                    "avg_resolution": np.mean(
                        [r["avg_visual_resolution"] for r in successful]
                    ),
                    "avg_structure": np.mean(
                        [r["avg_structural_quality"] for r in successful]
                    ),
                    "avg_artifacts": np.mean(
                        [r["avg_artifact_score"] for r in successful]
                    ),
                }
            )

        df = pd.DataFrame(rows).sort_values("avg_quality", ascending=False)

        # Print comparison
        print("\n" + "=" * 70)
        print("MODEL COMPARISON — COFFEE CUP BACKGROUND RESOLUTION BENCHMARK")
        print("=" * 70)

        for _, row in df.iterrows():
            print(f"\n{row['model'].upper()}")
            print(f"  Detection Rate:   {row.get('detection_rate', 0):.1%}")
            print(f"  Avg Quality:      {row.get('avg_quality', 0):.3f}")
            print(f"  Median Quality:   {row.get('median_quality', 0):.3f}")
            print(f"  Excellent (>0.7): {row.get('excellent_pct', 0):.1%}")
            print(f"  Semantic:         {row.get('avg_semantic', 0):.3f}")
            print(f"  Resolution:       {row.get('avg_resolution', 0):.3f}")
            print(f"  Structure:        {row.get('avg_structure', 0):.3f}")
            print(f"  Artifacts:        {row.get('avg_artifacts', 0):.3f}")

        # Save comparison
        df.to_csv("output/model_comparison.csv", index=False)
        print(f"\nComparison saved to: output/model_comparison.csv")

        # Generate plot
        self._plot_comparison(df)

        return df

    def _plot_comparison(self, df: pd.DataFrame):
        """Generate comparison visualization."""
        if df.empty or "avg_quality" not in df.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        models = df["model"].tolist()

        # Overall quality bar chart
        axes[0].barh(
            models,
            df["avg_quality"],
            xerr=df.get("std_quality", 0),
            color="steelblue",
            edgecolor="black",
            capsize=3,
        )
        axes[0].set_xlabel("Average Quality Score")
        axes[0].set_title("Overall Quality by Model")
        axes[0].set_xlim(0, 1)

        # Metric radar-style grouped bar
        metric_cols = ["avg_semantic", "avg_resolution", "avg_structure", "avg_artifacts"]
        metric_labels = ["Semantic", "Resolution", "Structure", "Artifacts"]

        x = np.arange(len(metric_labels))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            row = df[df["model"] == model].iloc[0]
            values = [row.get(col, 0) for col in metric_cols]
            axes[1].bar(x + i * width, values, width, label=model)

        axes[1].set_xticks(x + width * (len(models) - 1) / 2)
        axes[1].set_xticklabels(metric_labels)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Metric Breakdown by Model")
        axes[1].set_ylim(0, 1)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("output/model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison plot saved to: output/model_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Compare models on CCBench")

    parser.add_argument(
        "--models",
        nargs="+",
        help="Model names (expects image dirs at ./output/{model}/)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing *_results.json files",
    )
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    setup_logging("INFO")

    comparison = ModelComparison()

    if args.results_dir:
        comparison.compare_from_results(args.results_dir)
    elif args.models:
        model_dirs = {m: f"./output/{m}/" for m in args.models}
        comparison.compare_from_dirs(model_dirs, device=args.device)
    else:
        parser.error("Provide --models or --results-dir")


if __name__ == "__main__":
    main()
