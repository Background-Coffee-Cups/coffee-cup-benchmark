#!/usr/bin/env python3
"""
Generate test prompts for the Coffee Cup Background Resolution Benchmark.

Creates structured prompts across categories and difficulty levels
for consistent cross-model evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))


FOCAL_SUBJECTS = [
    "a person typing on a laptop",
    "a person reading a book",
    "hands holding a smartphone",
    "a person in conversation",
    "a barista making coffee",
    "a person writing in a notebook",
]

BACKGROUNDS = [
    "office desk",
    "cafe table",
    "kitchen counter",
    "living room side table",
    "park bench",
    "co-working space desk",
    "library reading table",
    "restaurant table",
]

CUP_DESCRIPTIONS = {
    "easy": [
        "a white ceramic coffee mug",
        "a simple coffee cup on a saucer",
    ],
    "medium": [
        "a half-full mug of coffee next to some papers",
        "a ceramic cup with a visible handle, partially obscured",
        "an espresso cup on a small saucer",
    ],
    "hard": [
        "multiple coffee cups at varying distances",
        "a travel mug, a ceramic cup, and a glass of water",
        "three different mugs scattered across background surfaces",
        "a translucent glass cup with coffee visible through it",
    ],
}

LIGHTING = [
    "warm ambient lighting",
    "natural window light",
    "soft daylight",
    "moody, cinematic lighting",
    "bright overhead fluorescent lighting",
]

STYLE = [
    "shallow depth of field, realistic photography",
    "deep focus, documentary style",
    "portrait style with bokeh background",
    "editorial photography, clean composition",
]


class PromptGenerator:
    """Generate structured benchmark prompts."""

    def generate(self, count: int = 60) -> list:
        """Generate a set of diverse benchmark prompts."""
        prompts = []
        prompt_id = 0

        for difficulty in ["easy", "medium", "hard"]:
            target_count = count // 3
            generated = 0

            combos = list(
                product(
                    FOCAL_SUBJECTS,
                    BACKGROUNDS,
                    CUP_DESCRIPTIONS[difficulty],
                    LIGHTING,
                    STYLE,
                )
            )

            import random
            random.shuffle(combos)

            for subject, bg, cup, light, style in combos:
                if generated >= target_count:
                    break

                prompt_text = (
                    f"{style.split(',')[0]} scene. {subject.capitalize()} in focus "
                    f"at a {bg}. In the background, {cup}. {light}, {style}."
                )

                prompts.append(
                    {
                        "id": f"{difficulty}_{prompt_id:03d}",
                        "category": bg.split()[0],
                        "difficulty": difficulty,
                        "prompt": prompt_text,
                    }
                )

                prompt_id += 1
                generated += 1

        return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark prompts")
    parser.add_argument(
        "--output",
        type=str,
        default="config/prompts.json",
        help="Output file (default: config/prompts.json)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=60,
        help="Number of prompts to generate (default: 60)",
    )

    args = parser.parse_args()

    generator = PromptGenerator()
    prompts = generator.generate(args.count)

    output = {
        "version": "1.0",
        "description": "Generated prompts for coffee cup background resolution benchmark",
        "prompts": prompts,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(prompts)} prompts -> {args.output}")
    for diff in ["easy", "medium", "hard"]:
        n = sum(1 for p in prompts if p["difficulty"] == diff)
        print(f"  {diff}: {n}")


if __name__ == "__main__":
    main()
