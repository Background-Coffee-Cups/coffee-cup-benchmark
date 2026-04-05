#!/usr/bin/env python3
"""
CLI entry point for the Coffee Cup Background Resolution Benchmark.

Usage:
    python scripts/run_benchmark.py --image path/to/image.jpg
    python scripts/run_benchmark.py --image-dir ./images/ --output results.json
    python scripts/run_benchmark.py --submit --model "model-name" --results results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import CoffeeCupBenchmark
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Coffee Cup Background Resolution Benchmark"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--image-dir", type=str, help="Directory of images to benchmark")
    group.add_argument(
        "--submit", action="store_true", help="Submit results to leaderboard"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization output",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (required for --submit)",
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Results JSON file to submit (required for --submit)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (auto-detected if omitted)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.submit:
        if not args.model or not args.results:
            parser.error("--submit requires --model and --results")
        submit_results(args.model, args.results)
        return

    benchmark = CoffeeCupBenchmark(device=args.device)
    save_viz = not args.no_viz

    if args.image:
        result = benchmark.run_single_image(args.image, save_visualization=save_viz)
        print(json.dumps(result, indent=2))

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    elif args.image_dir:
        results = benchmark.run_batch(args.image_dir, output_file=args.output)
        print(f"\nProcessed {len(results)} images. Results saved to: {args.output}")


def submit_results(model_name: str, results_path: str):
    """Submit benchmark results to the leaderboard."""
    import requests
    import yaml

    with open(results_path) as f:
        results = json.load(f)

    # Handle single result vs batch
    if isinstance(results, dict):
        results = [results]

    benchmark = CoffeeCupBenchmark.__new__(CoffeeCupBenchmark)
    # We only need export_submission which doesn't need detector/evaluator
    submission = CoffeeCupBenchmark.export_submission(
        benchmark, results, model_name
    )

    # Load API URL from config
    config_path = Path(__file__).parent.parent / "config" / "benchmark_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        api_url = config.get("leaderboard", {}).get(
            "api_url", "https://background-coffee-cups.github.io/coffee-cup-benchmark/api"
        )
    else:
        api_url = "https://background-coffee-cups.github.io/coffee-cup-benchmark/api"

    # Save submission locally
    submission_file = f"submission_{model_name.replace(' ', '_')}.json"
    with open(submission_file, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved locally: {submission_file}")

    # Attempt API submission
    try:
        resp = requests.post(f"{api_url}/submit", json=submission, timeout=30)
        if resp.status_code == 200:
            print(f"Submitted to leaderboard: {resp.json().get('url', api_url)}")
        else:
            print(
                f"Leaderboard submission returned {resp.status_code}. "
                f"Local file saved — you can submit manually later."
            )
    except requests.RequestException as e:
        print(f"Could not reach leaderboard API ({e}). Local file saved.")


if __name__ == "__main__":
    main()
