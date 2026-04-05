# Coffee Cup Background Resolution Benchmark

AI image generators put all their effort into the hero subject and *give up* on background objects. This benchmark measures exactly that.

Generate a normal scene — an office, a garage, a hospital corridor — where someone in the background is holding a coffee cup. Then score whether the model actually resolved that cup into a coherent object or just painted a shapeless blob.

**Leaderboard:** [background-coffee-cups.github.io/coffee-cup-benchmark](https://background-coffee-cups.github.io/coffee-cup-benchmark/)

## The Problem

![Failure example](examples/seedream_fail_01.jpeg)

*SeedReam 4.5 rendering a BMW pit lane scene. The cars are flawless. The background coffee cups are shapeless blobs — no handles, no rims, no structure. The model allocated its entire detail budget to the hero subjects and gave up on the cups.*

## How the Benchmark Works

1. **Generate images** using any AI model with the standard prompts below
2. **Score each image** by pasting it into Claude with the scoring prompt (or run the script)
3. **Submit your results** to the leaderboard

The evaluator uses Claude's vision to find every background coffee cup and score it on 7 metrics. No GPU required, no model downloads, no torch — just an Anthropic API key.

---

## Step 1: Generate Images

Use these 10 prompts with any image generation model. Each prompt creates a complex, realistic scene where the hero subject is NOT a coffee cup — but someone in the background is holding one.

### Prompt 1 — Garage
```
Two mechanics inspecting the underside of a lifted sports car in a professional garage. Sharp focus on the car and mechanics. In the background near the tool bench, a third mechanic leans against the wall holding a white paper coffee cup. Industrial fluorescent lighting, photojournalistic style, 35mm lens.
```

### Prompt 2 — Wedding
```
A bride adjusting her veil in a full-length mirror, sharp focus, soft window light. Behind her in the doorway, two bridesmaids chat — one holding a ceramic coffee mug in her left hand. Shallow depth of field, editorial wedding photography.
```

### Prompt 3 — Film Set
```
Film set behind the scenes. A director gestures at a monitor bank in the foreground, sharp focus. Background crew members mill around equipment — one grip holds a to-go coffee cup, a PA has a thermos. Mixed tungsten and daylight, documentary style, handheld feel.
```

### Prompt 4 — Hospital
```
Hospital corridor. A surgeon in scrubs reviews a chart in the foreground. Behind them at the nurses' station, two staff members — one holding a small white styrofoam cup of coffee. Harsh overhead fluorescent lighting, realistic medical drama aesthetic.
```

### Prompt 5 — Construction Site
```
Construction site morning briefing. A foreman points at blueprints spread on a folding table, hard hat and high-vis vest sharp in frame. Three workers listen in the background, two holding paper coffee cups. Golden hour side light, dust in the air, photojournalistic.
```

### Prompt 6 — Lecture Hall
```
University lecture hall from the back row. Professor at the whiteboard is in focus writing equations. Students scattered across seats in midground — one has a travel mug on the desk, another holds a takeaway cup mid-sip. Flat institutional lighting, candid photography.
```

### Prompt 7 — Airport
```
Airport gate waiting area. A businesswoman works on a laptop in the foreground, sharp. Behind her, passengers sit in rows — a man in a suit holds a coffee cup, a woman nearby has a ceramic airport lounge mug. Large windows with overcast daylight, wide angle.
```

### Prompt 8 — Street Food Market
```
Street food market at dusk. A vendor flips noodles in a flaming wok, sharp focus, dramatic fire light. Customers queue behind — two people in the crowd hold small espresso cups from a nearby coffee cart. Neon signs in background, cinematic 85mm bokeh.
```

### Prompt 9 — Recording Studio
```
Recording studio control room. A producer adjusts faders on a mixing console, close-up sharp. Through the glass, a singer stands at a mic. To the right of the console, an engineer sits back holding a ceramic mug. Dim moody lighting, warm tones.
```

### Prompt 10 — Press Conference
```
Police press conference. An officer speaks at a podium with microphones, sharp focus. Behind them, a row of officials sit at a long table — one holds a white coffee cup, another has a paper cup next to their nameplate. Harsh camera flash lighting, press pool angle.
```

---

## Step 2: Score Your Images

### Option A: Paste into Claude (no code required)

Upload your generated image to [claude.ai](https://claude.ai) and paste this scoring prompt:

<details>
<summary><strong>Click to expand the scoring prompt</strong></summary>

```
You are a benchmark evaluator for the Coffee Cup Background Resolution Benchmark (CCBench).

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

Be harsh but fair. A slightly soft background cup at appropriate depth-of-field is NOT a failure — it's correct photography. A featureless blob where a cup should be IS a failure.
```

</details>

### Option B: Run the script (batch evaluation)

Requires Python 3.9+ and an Anthropic API key.

```bash
git clone https://github.com/Background-Coffee-Cups/coffee-cup-benchmark.git
cd coffee-cup-benchmark
pip install anthropic
```

Set your API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or if you use macOS Keychain:
```bash
security add-generic-password -s "com.shadow.control" -a "apiKey_anthropic" -w "sk-ant-..." -U
```

Put your generated images in a folder and run:
```bash
python scripts/eval_vision.py
```

The script will evaluate each image, score every background cup, and output a submission JSON.

---

## Step 3: Submit to the Leaderboard

Open a [GitHub Issue](https://github.com/Background-Coffee-Cups/coffee-cup-benchmark/issues/new?labels=submission) with:
- Model name and version
- Your submission JSON (from the script or manually assembled)
- Number of images tested

Or use the submit button on the [leaderboard site](https://background-coffee-cups.github.io/coffee-cup-benchmark/).

---

## Scoring

### Per-Cup Metrics (0.0 – 1.0)

| Metric | Weight | What it measures |
|--------|--------|------------------|
| Semantic Quality | 25% | Does it look like a real cup in context? |
| Visual Resolution | 20% | Appropriate detail for depth (not just a blob) |
| Detection | 15% | Can you identify it as a cup at all? |
| Structural Quality | 15% | Handle, rim, cylindrical body, proportions |
| Artifact Score | 10% | Freedom from AI artifacts (sharp text on soft cups, melting, hallucinated logos) |
| Color Coherence | 8% | Realistic colors matching scene lighting |
| Edge Quality | 7% | Clean edges appropriate for depth |

### Gave-Up Score (0.0 – 1.0)

Measures how much the model abandoned background detail compared to the hero subject.
- **0.0** — Background cups have appropriate detail for their depth
- **0.5** — Noticeable quality drop in background
- **1.0** — Hero is crisp, background cups are shapeless blobs

### Quality Scale

| Score | Rating |
|-------|--------|
| 0.8+ | Excellent — cup is coherent and appropriate for depth |
| 0.6–0.8 | Good — identifiable cup with minor issues |
| 0.4–0.6 | Fair — cup-like but losing structure |
| < 0.4 | Poor — model gave up |

---

## Current Results

### SeedReam 4.5

| Metric | Score |
|--------|-------|
| **Overall Quality** | **0.829** |
| Detection | 0.913 |
| Semantic Quality | 0.878 |
| Artifact Score | 0.839 |
| Structural Quality | 0.813 |
| Visual Resolution | 0.730 |
| Gave-Up Score | 0.18 |
| Images | 10 |
| BG Cups Found | 13 |

Best scene: Garage (0.888). Worst: Studio (cup placed in foreground instead of background).

---

## Example Output

```json
{
  "image_description": "Professional garage with mechanics and a lifted sports car",
  "hero_subject": "Sports car on lift and two mechanics inspecting it",
  "background_cups": [
    {
      "location": "Right side, near tool bench",
      "held_by": "Third mechanic leaning against wall",
      "cup_type": "paper",
      "detection": 0.95,
      "structural_quality": 0.9,
      "semantic_quality": 0.9,
      "visual_resolution": 0.85,
      "artifact_score": 0.9,
      "color_coherence": 0.85,
      "edge_quality": 0.8,
      "notes": "Well-defined white paper cup with steam, appropriate softness for background depth"
    }
  ],
  "scene_notes": "Model maintained background detail well",
  "gave_up_score": 0.1
}
```

---

## Repository Structure

```
coffee-cup-benchmark/
├── README.md                          # This file
├── config/
│   └── prompts.json                   # All 22 standard prompts
├── scripts/
│   ├── eval_vision.py                 # Claude Vision evaluator
│   └── generate_seedream.py           # SeedReam 4.5 image generator
├── docs/
│   ├── index.html                     # Leaderboard website
│   └── leaderboard.json               # Current rankings
├── examples/
│   ├── seedream_4_5_test.jpeg         # Good result example
│   └── seedream_fail_*.jpeg           # Failure examples (blob cups)
├── output/
│   └── seedream_4_5/                  # SeedReam benchmark images + results
├── requirements.txt
├── LICENSE
└── .gitignore
```

## License

MIT — [Pushing Squares](https://github.com/Background-Coffee-Cups)
