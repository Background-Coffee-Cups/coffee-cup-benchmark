# Coffee Cup Background Resolution Benchmark

A benchmarking framework for evaluating how well AI image generation models resolve coffee cups in the background of images.

**Leaderboard:** [ccbench.pushingsquares.com](https://ccbench.pushingsquares.com) (coming soon)

## Overview

This toolkit provides:
- **Multi-model detection** (YOLO + OWL-ViT) for robust cup identification
- **Background/foreground classification** based on spatial reasoning
- **7-metric quality evaluation** system assessing:
  - Semantic coherence (CLIP-based)
  - Visual resolution (sharpness, detail)
  - Structural quality (cup-like features)
  - Artifact detection
  - Color coherence
  - Edge quality
  - Detection confidence
- **Comprehensive reporting** with visualizations
- **Model comparison** tools
- **Leaderboard submission** for cross-model ranking

## Why Background Coffee Cups?

- Tests spatial reasoning beyond the focal point
- Reveals if models "forget" background details during generation
- Measures depth-of-field understanding
- Practical for product photography and scene composition
- Exposes common AI artifacts in peripheral regions

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (recommended for GPU acceleration)
- 8GB+ RAM

### Setup

```bash
git clone https://github.com/pushingsquares/coffee-cup-benchmark.git
cd coffee-cup-benchmark
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Models download automatically on first run, or manually:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
python -c "import clip; clip.load('ViT-B/32')"
```

## Quick Start

### Single Image Evaluation

```python
from src.benchmark import CoffeeCupBenchmark
import json

benchmark = CoffeeCupBenchmark()
result = benchmark.run_single_image("path/to/image.jpg", save_visualization=True)
print(json.dumps(result, indent=2))
```

### Batch Processing

```python
results = benchmark.run_batch("./generated_images/", output_file="results.json")
```

### Compare Models

```bash
python scripts/compare_models.py --models dalle3 midjourney stable_diffusion flux
```

### Submit to Leaderboard

```python
submission = benchmark.export_submission(results, model_name="my-model-v1")
# Upload via: python scripts/run_benchmark.py --submit --model "my-model-v1"
```

## Command Line

```bash
# Benchmark single image
python scripts/run_benchmark.py --image path/to/image.jpg --output result.json

# Batch benchmark
python scripts/run_benchmark.py --image-dir ./images/ --output batch_results.json

# Generate test prompts
python scripts/generate_prompts.py --output config/prompts.json --count 60

# Compare models
python scripts/compare_models.py --models dalle3 midjourney stable_diffusion

# Submit results to leaderboard
python scripts/run_benchmark.py --submit --model "model-name" --results batch_results.json
```

## Metrics

### Overall Quality (Weighted Average)
| Metric | Weight | Description |
|--------|--------|-------------|
| Semantic Quality | 25% | CLIP-based cup coherence |
| Visual Resolution | 20% | Sharpness and detail level |
| Detection Confidence | 15% | Detector certainty |
| Structural Quality | 15% | Cup-like features (rim, handle, proportions) |
| Artifact Score | 10% | Freedom from AI generation artifacts |
| Color Coherence | 8% | Realistic color distribution |
| Edge Quality | 7% | Clean vs mushy edges |

**Scale:** 0.0 - 1.0
- 0.8+ : Excellent
- 0.6-0.8: Good
- 0.4-0.6: Fair
- < 0.4 : Poor

## Output Files

- `benchmark_results.json` — Raw evaluation scores
- `*_annotated.jpg` — Images with detected cups highlighted (color-coded by quality)
- `benchmark_analysis.png` — 4-panel analysis (histogram, metrics, scatter, correlation)

## Configuration

Edit `config/benchmark_config.yaml`:

```yaml
detector:
  yolo_model: "yolov8x.pt"
  yolo_confidence: 0.15
  owl_model: "google/owlvit-base-patch32"
  owl_confidence: 0.15
  nms_threshold: 0.5

evaluator:
  clip_model: "ViT-B/32"
  device: "cuda"  # or "cpu" or "auto"

benchmark:
  save_visualizations: true
  visualization_dpi: 150
```

## Testing

```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## Performance

On NVIDIA RTX 3090:
- Single image detection: ~2 seconds
- Single cup evaluation: ~0.5 seconds
- Batch of 100 images: ~5-7 minutes

## Contributing

Contributions welcome! Areas for improvement:
- Additional detection models (SAM, Grounding DINO)
- More quality metrics
- GPU optimization
- Support for video sequences
- Real-world cup dataset for validation

## License

MIT License - see [LICENSE](LICENSE)

## References

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32)
- [CLIP](https://github.com/openai/CLIP)

---

**Made with coffee, for evaluating coffee in AI images** | [Pushing Squares](https://pushingsquares.com)
