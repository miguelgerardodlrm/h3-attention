# h3-attention

**Attention control for MiniMax H3 inference** — reduce hallucination, identity drift, reference loss, and prompt deviation without retraining.

## Overview

`h3-attention` is an experimental Python package that intercepts and modifies the attention patterns of MiniMax H3 during inference. It works by replacing/wrapping the `MiniMaxH3AttnProcessor` in each transformer block, allowing configurable control over how different modalities (text, video, audio, references) attend to each other.

### Key Features

- **7 independent control mechanisms** — reference boost, identity lock, prompt adherence, temporal lock, subject separation, audio strength, adaptive mode
- **Compatible with acceleration methods** — LoRA, SageAttn, Flash Attention, SDPA, BlockSwap, TeaCache
- **Sol-attn aware** — wraps existing Sol-attn processors instead of replacing them
- **ComfyUI integration** — 3 custom nodes for controller, settings, and debug
- **Timestep scheduling** — different control strengths at early/middle/late denoising phases
- **Adaptive mode** — automatically adjusts parameters based on attention metrics
- **Debug/inspection** — lightweight logging of affinity metrics per layer/step

## Installation

```bash
# From source
git clone https://github.com/your-repo/h3-attention.git
cd h3-attention
pip install -e .

# With ComfyUI support
pip install -e .[comfyui]

# With SageAttn support
pip install -e .[sage]

# Development
pip install -e .[dev]
```

## Quick Start

### Python API

```python
import torch
from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks
from h3_attention import H3AttentionController, PRESETS, install_controller_on_model

# Load H3 Ref2VA model
pipe = MiniMaxH3Ref2VABlocks().init_pipeline("MiniMaxAI/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

# Install attention controller with maximum coherence preset
controller = H3AttentionController(preset="maximum_coherence")
controller.install(pipe.transformer_ref)

# Generate with attention control
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

reference = MiniMaxH3ImageReference(image="subject.png")

result = pipe(
    prompt="The subject dances in a ballroom, elegant movements",
    references=[reference],
    num_frames=124,
    num_inference_steps=30,
)

# Check metrics
print(controller.get_metrics_summary())
# {'avg_reference_affinity': 0.61, 'avg_prompt_affinity': 0.54, ...}

# Clean up
controller.uninstall()
```

### Using Presets

```python
from h3_attention import PRESETS, AttentionControlConfig

# Use built-in presets
config = PRESETS["identity_preservation"].copy()
config.reference_strength = 0.3  # Override specific values

controller = H3AttentionController(config=config)
```

### With Schedule (Timestep-Dependent Control)

```python
from h3_attention import SCHEDULE_PRESETS, H3AttentionController

# Early: strong reference + prompt, Late: strong temporal consistency
schedule = SCHEDULE_PRESETS["reference_heavy_early"]
controller = H3AttentionController(preset="maximum_coherence", schedule=schedule)
```

### ComfyUI Nodes

Install the package in your ComfyUI environment, then restart ComfyUI. Three nodes will be available under **MiniMax H3/Attention**:

1. **H3 Attention Controller** — Main node, connects between model loader and sampler
2. **H3 Attention Settings** — Detailed parameter configuration (connects to controller's optional `settings` input)
3. **H3 Attention Debug** — Captures and logs metrics (connects to controller's optional `debug_node` input)

See `example_workflows/h3_attention_maximum_coherence.json` for a complete example workflow.

## Control Mechanisms

| Parameter            | Range   | Default | Description                                                          |
| -------------------- | ------- | ------- | -------------------------------------------------------------------- |
| `reference_strength` | 0.0–1.0 | 0.0     | Boost attention from reference tokens to target generation           |
| `identity_lock`      | 0.0–1.0 | 0.0     | Preserve identity consistency between reference and generated frames |
| `prompt_adherence`   | 0.0–1.0 | 0.0     | Strengthen attention from text tokens to generated content           |
| `temporal_lock`      | 0.0–1.0 | 0.0     | Enforce temporal consistency across generated frames                 |
| `subject_separation` | 0.0–1.0 | 0.0     | Reduce cross-attention between different subject references          |
| `audio_strength`     | 0.0–1.0 | 0.0     | Boost audio-video cross-modal attention                              |
| `adaptive_mode`      | bool    | False   | Automatically adjust parameters based on attention metrics           |

Each mechanism can be independently enabled/disabled by setting its strength to 0.0.

## Architecture

### Intervention Point

The package operates at **`MiniMaxH3AttnProcessor.__call__`** in `diffusers/models/transformers/transformer_minimax_h3.py`. This is the single-stream self-attention over the packed multimodal sequence.

```
Packed Sequence Layout (Ref2VA):
[TEXT | REFERENCE_BLOCKS | TARGET_AUDIO | TARGET_VIDEO]
       ↑              ↑              ↑            ↑
   token_tag=1   token_tag=0/2  token_tag=2    token_tag=0
```

### Modality Identification

Uses `token_tags` (0=video, 1=text, 2=audio) and `position_ids` (t, h, w coordinates) from the packed sequence, plus index tensors (`video_indices`, `audio_indices`, `text_indices`, `ref_video_indices`, `ref_audio_indices`) provided by the pipeline's layout step.

### Compatibility Strategy

| Method                  | Compatibility | How                                                             |
| ----------------------- | ------------- | --------------------------------------------------------------- |
| **LoRA**                | ✅ Full       | Modifies Q/K/V projections before processor                     |
| **SageAttn/Flash/SDPA** | ✅ Full       | Backend selected via `dispatch_attention_fn`                    |
| **BlockSwap/Offload**   | ✅ Full       | Processor travels with block                                    |
| **TeaCache**            | ✅ Full       | Skips blocks; processor runs when block runs                    |
| **Sol-attn**            | ⚠️ Wrapped    | Detects `HyperFlowSolAttnProcessor`, wraps instead of replacing |

## Presets

### Config Presets (`PRESETS`)

```python
PRESETS = {
    "disabled": AttentionControlConfig(),
    "reference_only": ...,           # Boost reference attention
    "identity_preservation": ...,    # Focus on identity consistency
    "prompt_adherence": ...,         # Strengthen prompt following
    "temporal_consistency": ...,     # Smooth temporal transitions
    "multi_subject": ...,            # Multiple character separation
    "audio_sync": ...,               # Audio-video synchronization
    "adaptive_balanced": ...,        # Adaptive with moderate settings
    "maximum_coherence": ...,        # All mechanisms active
    "debug": ...,                    # Debug-friendly settings
}
```

### Schedule Presets (`SCHEDULE_PRESETS`)

```python
SCHEDULE_PRESETS = {
    "disabled": AttentionSchedule(phases={}),
    "reference_heavy_early": ...,    # Early: ref+prompt, Late: temporal
    "identity_focused": ...,         # Identity throughout
    "temporal_coherence": ...,       # Progressive temporal locking
    "prompt_adherence": ...,         # Prompt-heavy early
    "balanced": ...,                 # Balanced three-phase
}
```

## ComfyUI Workflow

The package includes example workflows in `example_workflows/`:

- **`h3_attention_maximum_coherence.json`** — Full multi-character test (3 reference images) with the acceleration chain (Spectrum, Sage/BlockSparse attention, quantized UNet)
- **`h3_attention_single_character.json`** — Simplified single-character test for isolating identity vs interaction issues

### Workflow Templates (ComfyUI)

Both workflows appear in ComfyUI's **Workflow → Browse Workflow Templates** (Plantillas) browser, under the `h3-attention` category, because ComfyUI indexes the canonical `example_workflows/` folder of every custom node pack.

For a template to **open** (not just appear), two conditions must hold:

1. The pack loads without errors at ComfyUI startup (watch the console for `h3-attention` import errors).
2. **Fully restart ComfyUI after `git pull`** — the template file routes are registered only once, at startup.

Verify the endpoint:

```bash
# 1. Index must list the pack:
curl -s http://127.0.0.1:8188/api/workflow_templates
# -> {"h3-attention":["h3_attention_maximum_coherence","h3_attention_single_character"], ...}

# 2. Template file must be served (expect 200, not 404):
curl -s -o /dev/null -w "%{http_code}\n" \
  http://127.0.0.1:8188/api/workflow_templates/h3-attention/h3_attention_single_character.json
```

If step 2 returns `404`, the pack did not load — fix the startup error first (the index lists files on disk even when the pack fails to load, so templates can "appear" while clicking silently does nothing). If both return OK but a click still does nothing, check the browser DevTools (F12) → Network/Console for the failing request.

### Workflow Structure (Non-Overlapping)

```
Model Loading → Acceleration → Attention Control → Conditioning → Sampling → Output
     ↓              ↓               ↓                ↓            ↓         ↓
 UNETLoader   Spectrum +       H3Attention      MiniMaxH3      Sampler-    VAEDecode +
              Sage +           Controller +     ReferenceTo    Custom-     CreateVideo +
              BlockSparse +    Settings +       Video          Advanced    SaveVideo
              ModelPreview     Debug
```

## Metrics & Debugging

When `debug_attention=True`, the controller logs per-layer/step metrics:

```
[H3-Attention] step= 23 layer=17 ref_aff=0.612 prompt_aff=0.541 temp_aff=0.423 bias=0.184
```

Available metrics:

- `reference_affinity` — Mean attention from target to reference tokens
- `prompt_affinity` — Mean attention from target to text tokens
- `temporal_affinity` — Mean attention between adjacent frames
- `audio_video_affinity` — Cross-modal audio-video attention
- `applied_bias_mean/max` — Magnitude of injected attention bias
- `attention_entropy/concentration` — Attention distribution diversity

Access programmatically:

```python
summary = controller.get_metrics_summary()
# {'steps_recorded': 30, 'avg_reference_affinity': 0.61, ...}

history = debug_node.get_metrics_history()  # Full per-layer history
debug_node.save_metrics("my_metrics.json")
```

## Testing

```bash
# Unit tests (no GPU required)
pytest tests/test_config.py tests/test_processor.py tests/test_metrics.py -v

# Integration tests (requires GPU + model)
pytest tests/test_integration.py -v --run-slow
```

## Benchmarking

```bash
python examples/benchmark.py \
    --model-path /path/to/MiniMax-H3 \
    --variant ref2va \
    --prompt "Three characters dance duranguense together" \
    --reference-dir ./refs \
    --configs baseline reference_only identity_preservation maximum_coherence \
    --runs 3 \
    --output results.json
```

## Experimental Test Cases

### Three-Character Stress Test (Mikhael, Kingpin, Vesper)

**Prompt**: Three characters perform duranguense choreography in same space, maintaining exact identities and clothing from references.

**Observed failure modes without control**:

- Character disappearance
- Identity mixing (Kingpin gets Mikhael's vest)
- Clothing changes
- Invented elements
- Loss of interaction
- Temporal drift

### Single-Character Baseline (Mikhael only)

Isolates identity preservation from multi-subject interaction complexity.

## Limitations & Risks

| Limitation                                | Mitigation                                             |
| ----------------------------------------- | ------------------------------------------------------ |
| **No sparse attention support yet**       | Will adapt when MiniMax releases sparse kernels        |
| **Sol-attn wrapping is limited**          | Full control requires native processor replacement     |
| **Subject separation needs segmentation** | Currently uses heuristic; needs reference segmentation |
| **Adaptive mode is heuristic**            | Not a learned controller; may oscillate                |
| **VRAM overhead**                         | Bias tensors [B, H, S, S] — use `clamp_bias=True`      |

## Degradation Risks

- **Over-boosting references** → Hallucinated reference artifacts, reduced creativity
- **Excessive identity lock** → Frozen poses, lack of natural movement
- **Strong temporal lock** → Over-smoothed motion, loss of dynamics
- **Aggressive subject separation** → Spatial separation artifacts

Start with conservative values (0.1–0.3) and increase gradually.

## Project Structure

```
h3-attention/
├── h3_attention/
│   ├── __init__.py          # Public API
│   ├── config.py            # AttentionControlConfig, PRESETS
│   ├── controller.py        # H3AttentionController
│   ├── processor.py         # H3ControlledAttnProcessor, H3AttnProcessorWrapper
│   ├── scheduler.py         # AttentionSchedule, PhaseConfig
│   ├── metrics.py           # AttentionMetrics, computation functions
│   └── comfyui.py           # ComfyUI convenience functions
├── comfyui_nodes/
│   ├── __init__.py          # Node registration
│   ├── controller_node.py   # H3AttentionControllerNode
│   ├── settings_node.py     # H3AttentionSettingsNode
│   └── debug_node.py        # H3AttentionDebugNode
├── tests/
│   ├── test_config.py
│   ├── test_processor.py
│   ├── test_metrics.py
│   └── test_integration.py
├── example_workflows/
│   ├── h3_attention_maximum_coherence.json
│   └── h3_attention_single_character.json
├── examples/
│   └── benchmark.py
├── pyproject.toml
└── README.md
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (with CUDA)
- Diffusers ≥ 0.30 (Modular Pipeline support)
- Transformers ≥ 4.40
- Accelerate ≥ 0.30
- ComfyUI ≥ 0.3.0 (for nodes)

## License

Apache-2.0 — Compatible with MiniMax H3 and Diffusers licenses.

## Citation

If you use this in research, please cite:

```bibtex
@software{h3_attention,
  title = {h3-attention: Attention Control for MiniMax H3},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/h3-attention}
}
```

## Acknowledgments

- MiniMax AI for releasing H3
- Hugging Face Diffusers team for Modular Pipeline integration
- ComfyUI community for native H3 nodes
- HyperFlow-Sol for Sol-attn implementation
- SageAttention authors for efficient attention kernels
