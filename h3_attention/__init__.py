"""
h3-attention: Attention control for MiniMax H3 inference.

Provides H3AttentionController to modify attention patterns during generation
without retraining, targeting:
- Hallucination reduction
- Identity drift prevention
- Reference adherence improvement
- Prompt adherence enhancement
- Temporal consistency
"""

from .config import PRESETS
from .controller import H3AttentionController, AttentionControlConfig

try:
    # Requires diffusers (present in offline/Diffusers deployments and tests).
    # ComfyUI runtimes may not have diffusers installed — never hard-fail.
    from .processor import H3ControlledAttnProcessor, H3AttnProcessorWrapper
except ImportError:  # pragma: no cover - environment dependent
    H3ControlledAttnProcessor = None
    H3AttnProcessorWrapper = None

from .scheduler import AttentionSchedule, SchedulePhase, PhaseConfig, SCHEDULE_PRESETS
from .metrics import (
    AttentionMetrics,
    compute_attention_metrics,
    compute_reference_affinity,
    compute_prompt_affinity,
    compute_temporal_affinity,
    compute_subject_separation,
    aggregate_metrics,
)
from .comfyui import (
    install_controller_on_model,
    remove_controller_from_model,
    install_comfyui,
    MAX_CONTROLLED_SEQ,
)
from .native import NativeBiasPlanner, make_controlled_attention

__version__ = "0.1.0"

__all__ = [
    "H3AttentionController",
    "AttentionControlConfig",
    "H3ControlledAttnProcessor",
    "H3AttnProcessorWrapper",
    "AttentionSchedule",
    "SchedulePhase",
    "PhaseConfig",
    "AttentionMetrics",
    "compute_attention_metrics",
    "compute_reference_affinity",
    "compute_prompt_affinity",
    "compute_temporal_affinity",
    "compute_subject_separation",
    "aggregate_metrics",
    "install_controller_on_model",
    "remove_controller_from_model",
    "install_comfyui",
    "MAX_CONTROLLED_SEQ",
    "NativeBiasPlanner",
    "make_controlled_attention",
    "PRESETS",
    "SCHEDULE_PRESETS",
]
