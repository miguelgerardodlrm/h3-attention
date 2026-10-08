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

from .controller import H3AttentionController, AttentionControlConfig
from .processor import H3ControlledAttnProcessor, H3AttnProcessorWrapper
from .scheduler import AttentionSchedule, SchedulePhase
from .metrics import AttentionMetrics, compute_reference_affinity, compute_prompt_affinity
from .comfyui import install_controller_on_model, remove_controller_from_model

__version__ = "0.1.0"

__all__ = [
    "H3AttentionController",
    "AttentionControlConfig",
    "H3ControlledAttnProcessor",
    "H3AttnProcessorWrapper",
    "AttentionSchedule",
    "SchedulePhase",
    "AttentionMetrics",
    "compute_reference_affinity",
    "compute_prompt_affinity",
    "install_controller_on_model",
    "remove_controller_from_model",
]
