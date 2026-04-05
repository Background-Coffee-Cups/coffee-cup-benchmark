#!/usr/bin/env python3
"""Generate benchmark test images using SeedReam 4.5 API."""

import json
import subprocess
import requests
import time
from pathlib import Path


def get_api_key():
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "com.shadow.control",
         "-a", "apiKey_seedream", "-w"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def generate_image(prompt, api_key, output_path):
    """Generate a single image via SeedReam 4.5 API."""
    url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "seedream-4-5-251128",
        "prompt": prompt,
        "response_format": "url",
        "size": "2K",
        "watermark": False,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        print(f"    API error {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()

    data = resp.json()
    image_url = None
    for img in data.get("data", []):
        image_url = img.get("url") or img.get("image_url")
        if image_url:
            break
    if not image_url:
        raise ValueError(f"No image URL in response: {json.dumps(data)[:300]}")

    # Download image
    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(img_resp.content)

    return output_path


def main():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: No SeedReam API key found in keychain")
        return

    # Load benchmark prompts
    prompts_file = Path(__file__).parent.parent / "config" / "prompts.json"
    with open(prompts_file) as f:
        prompts_data = json.load(f)

    output_dir = Path(__file__).parent.parent / "output" / "seedream_4_5"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only scene_* prompts (complex scenes with BG cup holders)
    prompts = [p for p in prompts_data["prompts"] if p["id"].startswith("scene_")]
    if not prompts:
        prompts = prompts_data["prompts"]
    print(f"Generating {len(prompts)} images with SeedReam 4.5...\n")

    results = []
    for i, prompt_entry in enumerate(prompts):
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]
        output_path = output_dir / f"{prompt_id}.jpg"

        print(f"[{i+1}/{len(prompts)}] {prompt_id} ({prompt_entry['difficulty']})...")

        try:
            generate_image(prompt_text, api_key, str(output_path))
            print(f"  -> Saved: {output_path}")
            results.append({"id": prompt_id, "status": "ok", "path": str(output_path)})
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({"id": prompt_id, "status": "error", "error": str(e)})

        # Rate limit
        if i < len(prompts) - 1:
            time.sleep(2)

    # Save generation log
    log_path = output_dir / "generation_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone: {ok}/{len(prompts)} images generated -> {output_dir}")


if __name__ == "__main__":
    main()
