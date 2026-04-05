#!/usr/bin/env python3
"""
Vision-based benchmark evaluator using Claude's vision API.

Replaces the heavy YOLO/CLIP/torch pipeline with Claude's multimodal
understanding. More nuanced than CV2-only evaluation — Claude can assess
semantic coherence, structural quality, and artifact detection in ways
that programmatic metrics cannot.

Requires: anthropic SDK + API key in macOS Keychain.
"""

import anthropic
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_PROMPT = """You are a benchmark evaluator for the Coffee Cup Background Resolution Benchmark (CCBench).

Your job: find every coffee cup in the BACKGROUND of this AI-generated image and score how well the model resolved each one.

## What to look for

Background coffee cups — cups held by people who are NOT the main subject, cups sitting on surfaces behind the focal point, cups at distance. NOT foreground cups that are the hero of the shot.

## Scoring criteria (each 0.0 to 1.0)

1. **detection** — Can you identify a distinct coffee cup in the background? (1.0 = clearly a cup, 0.0 = unrecognizable blob)
2. **structural_quality** — Does it have cup-like features? Handle, rim, cylindrical body, proper proportions? (1.0 = anatomically correct cup, 0.0 = shapeless mass)
3. **semantic_quality** — Does it look like a real coffee cup in context? Correct scale, appropriate for the scene? (1.0 = photorealistic, 0.0 = nonsensical)
4. **visual_resolution** — Is the level of detail appropriate for its depth? Background cups should be softer but still coherent — not just low-effort blobs. (1.0 = appropriate detail for depth, 0.0 = model gave up)
5. **artifact_score** — Freedom from AI generation artifacts. Watch for: sharp text on soft cups (sharpness inconsistency), melting/morphing shapes, impossible geometry, hallucinated logos. (1.0 = no artifacts, 0.0 = severe artifacts)
6. **color_coherence** — Realistic, consistent colors? Matches the scene lighting? (1.0 = natural colors, 0.0 = color banding/impossible hues)
7. **edge_quality** — Clean edges appropriate for depth, not mushy blobs or unnaturally sharp cutouts? (1.0 = clean natural edges, 0.0 = mushy or artifacted)

## Response format

Return ONLY valid JSON, no markdown fencing:

{
  "image_description": "Brief description of the scene",
  "hero_subject": "What the model focused on",
  "background_cups": [
    {
      "location": "description of where in the image",
      "held_by": "who is holding it or where it sits",
      "cup_type": "paper/ceramic/travel/etc",
      "detection": 0.0,
      "structural_quality": 0.0,
      "semantic_quality": 0.0,
      "visual_resolution": 0.0,
      "artifact_score": 0.0,
      "color_coherence": 0.0,
      "edge_quality": 0.0,
      "notes": "specific observations about this cup"
    }
  ],
  "foreground_cups": [
    {
      "location": "description",
      "notes": "brief note — these are NOT scored, just logged"
    }
  ],
  "scene_notes": "overall observations about how the model handled background detail",
  "gave_up_score": 0.0
}

The **gave_up_score** (0.0–1.0) measures how much the model "gave up" on background cups vs the hero subject. 0.0 = background cups are just as detailed as the hero. 1.0 = model clearly allocated zero effort to background cups (shapeless blobs while hero is crisp).

Be harsh but fair. A slightly soft background cup at appropriate depth-of-field is NOT a failure — it's correct photography. A featureless blob where a cup should be IS a failure."""


def get_api_key():
    import os

    # 1. Environment variable
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key

    # 2. macOS Keychain fallback
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "com.shadow.control",
             "-a", "apiKey_anthropic", "-w"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass  # Not on macOS

    return ""


def encode_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.standard_b64encode(data).decode("utf-8")


def evaluate_image(client, image_path, prompt_info=None):
    """Evaluate a single image using Claude's vision."""
    b64 = encode_image(image_path)

    # Determine media type
    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    context = ""
    if prompt_info:
        context = f"\n\nThis image was generated from the prompt: \"{prompt_info['prompt']}\"\nCategory: {prompt_info.get('category', 'unknown')} | Difficulty: {prompt_info.get('difficulty', 'unknown')}"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": EVAL_PROMPT + context,
                    },
                ],
            }
        ],
    )

    # Parse JSON response
    response_text = message.content[0].text.strip()
    # Handle potential markdown fencing
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

    return json.loads(response_text)


def compute_scores(eval_result):
    """Compute aggregate scores from Claude's evaluation."""
    cups = eval_result.get("background_cups", [])

    if not cups:
        return {
            "num_bg_cups": 0,
            "detection_success": False,
            "avg_quality": 0.0,
            "gave_up_score": eval_result.get("gave_up_score", 1.0),
        }

    metrics = [
        "detection", "structural_quality", "semantic_quality",
        "visual_resolution", "artifact_score", "color_coherence", "edge_quality"
    ]

    weights = {
        "detection": 0.15,
        "semantic_quality": 0.25,
        "visual_resolution": 0.20,
        "structural_quality": 0.15,
        "artifact_score": 0.10,
        "color_coherence": 0.08,
        "edge_quality": 0.07,
    }

    cup_qualities = []
    for cup in cups:
        quality = sum(cup.get(m, 0) * weights[m] for m in weights)
        cup["overall_quality"] = round(quality, 3)
        cup_qualities.append(quality)

    avg_scores = {}
    for m in metrics:
        vals = [cup.get(m, 0) for cup in cups]
        avg_scores[f"avg_{m}"] = round(sum(vals) / len(vals), 3)

    return {
        "num_bg_cups": len(cups),
        "detection_success": True,
        "avg_quality": round(sum(cup_qualities) / len(cup_qualities), 3),
        "max_quality": round(max(cup_qualities), 3),
        "min_quality": round(min(cup_qualities), 3),
        "gave_up_score": eval_result.get("gave_up_score", 0),
        **avg_scores,
    }


def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: No Anthropic API key in keychain")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    image_dir = Path(__file__).parent.parent / "output" / "seedream_4_5"
    prompts_file = Path(__file__).parent.parent / "config" / "prompts.json"

    # Load prompts for context
    with open(prompts_file) as f:
        prompts_data = json.load(f)
    prompt_map = {p["id"]: p for p in prompts_data["prompts"]}

    # Get scene_* images
    images = sorted(image_dir.glob("scene_*.jpg"))
    if not images:
        images = sorted(image_dir.glob("*.jpg"))

    print(f"Evaluating {len(images)} images with Claude Vision...\n")

    all_results = []
    for i, img_path in enumerate(images):
        prompt_id = img_path.stem
        prompt_info = prompt_map.get(prompt_id)

        print(f"  [{i+1}/{len(images)}] {img_path.name}...", end=" ", flush=True)

        try:
            eval_result = evaluate_image(client, str(img_path), prompt_info)
            scores = compute_scores(eval_result)

            result = {
                "image": img_path.name,
                "prompt_id": prompt_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluation": eval_result,
                **scores,
            }

            all_results.append(result)

            cups = scores["num_bg_cups"]
            quality = scores["avg_quality"]
            gave_up = scores["gave_up_score"]
            print(f"cups={cups}  quality={quality:.3f}  gave_up={gave_up:.1f}")

            if eval_result.get("scene_notes"):
                print(f"           {eval_result['scene_notes'][:100]}")

        except Exception as e:
            print(f"FAILED: {e}")
            all_results.append({
                "image": img_path.name,
                "prompt_id": prompt_id,
                "error": str(e),
                "detection_success": False,
                "avg_quality": 0.0,
            })

    # Summary
    successful = [r for r in all_results if r.get("detection_success")]

    print("\n" + "=" * 60)
    print("SEEDREAM 4.5 — CLAUDE VISION BENCHMARK RESULTS")
    print("=" * 60)

    if successful:
        qualities = [r["avg_quality"] for r in successful]
        gave_ups = [r["gave_up_score"] for r in successful]

        print(f"  Images processed:    {len(all_results)}")
        print(f"  Detection success:   {len(successful)}/{len(all_results)}")
        print(f"  Total BG cups found: {sum(r['num_bg_cups'] for r in successful)}")
        print(f"")
        print(f"  Avg Quality:         {sum(qualities)/len(qualities):.3f}")
        print(f"  Avg Gave-Up Score:   {sum(gave_ups)/len(gave_ups):.2f}")
        print(f"")
        print(f"  Sub-metrics:")

        for key in ["avg_detection", "avg_semantic_quality", "avg_visual_resolution",
                     "avg_structural_quality", "avg_artifact_score", "avg_color_coherence",
                     "avg_edge_quality"]:
            vals = [r.get(key, 0) for r in successful if key in r]
            if vals:
                label = key.replace("avg_", "").replace("_", " ").title()
                print(f"    {label:.<25} {sum(vals)/len(vals):.3f}")

        print(f"\n  Per-image:")
        for r in successful:
            q = r["avg_quality"]
            g = r["gave_up_score"]
            c = r["num_bg_cups"]
            indicator = "!!" if g > 0.6 else "ok" if g < 0.3 else ".."
            print(f"    {indicator} {r['image']:<30} q={q:.3f}  cups={c}  gave_up={g:.1f}")

    # Save full results
    out_path = image_dir / "vision_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results: {out_path}")

    # Build leaderboard submission
    if successful:
        qualities = [r["avg_quality"] for r in successful]
        submission = {
            "model_name": "SeedReam 4.5",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "eval_mode": "claude-vision",
            "evaluator_model": "claude-sonnet-4-20250514",
            "summary": {
                "total_images": len(all_results),
                "detection_rate": round(len(successful) / len(all_results), 3),
                "avg_quality": round(sum(qualities) / len(qualities), 3),
                "avg_gave_up": round(sum(gave_ups) / len(gave_ups), 2),
            },
            "per_image": all_results,
        }

        sub_path = image_dir / "submission_vision_seedream_4_5.json"
        with open(sub_path, "w") as f:
            json.dump(submission, f, indent=2)
        print(f"  Submission:   {sub_path}")


if __name__ == "__main__":
    main()
