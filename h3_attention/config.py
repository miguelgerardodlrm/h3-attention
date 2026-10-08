"""
Configuration for H3 Attention Control.

All parameters have safe defaults (0.0 = disabled) and can be independently toggled.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch


@dataclass
class AttentionControlConfig:
    """
    Configuration for attention modification during H3 inference.

    All strength parameters range from 0.0 (disabled) to 1.0 (maximum).
    Each mechanism can be independently enabled/disabled.
    """

    # === Reference Attention Control ===
    reference_strength: float = 0.0
    """Boost attention from reference tokens (images/video/audio) to target generation."""

    reference_boost_mode: str = "additive"
    """How to apply reference boost: 'additive' (bias) or 'multiplicative' (scale)."""

    reference_target_layers: Optional[List[int]] = None
    """Which transformer layers to apply reference boost (None = all)."""

    reference_target_modalities: List[str] = field(default_factory=lambda: ["video", "audio"])
    """Which target modalities receive reference boost."""

    # === Identity Lock ===
    identity_lock: float = 0.0
    """Preserve identity consistency between reference and generated frames."""

    identity_reference_idx: int = 0
    """Index of reference block to use as identity anchor (0 = first image/video ref)."""

    identity_temporal_decay: float = 0.95
    """Decay factor for identity influence over time steps."""

    identity_spatial_scope: str = "global"
    """Scope of identity lock: 'global', 'local', or 'patch'."""

    # === Prompt Adherence ===
    prompt_adherence: float = 0.0
    """Strengthen attention from text tokens to generated content."""

    prompt_text_tag: int = 1
    """Token tag value for text modality (default: MINIMAX_H3_TEXT_TAG = 1)."""

    # === Temporal Control ===
    temporal_lock: float = 0.0
    """Enforce temporal consistency across generated frames."""

    temporal_window: int = 5
    """Number of frames to consider for temporal consistency."""

    temporal_mode: str = "smooth"
    """Temporal control mode: 'smooth', 'strict', or 'adaptive'."""

    # === Subject Separation ===
    subject_separation: float = 0.0
    """Reduce cross-attention between different subject references."""

    min_subject_distance: float = 0.3
    """Minimum attention distance between different subjects."""

    # === Audio Strength ===
    audio_strength: float = 0.0
    """Boost audio-video cross-modal attention."""

    audio_video_sync: bool = True
    """Maintain audio-video synchronization in attention."""

    # === Adaptive Mode ===
    adaptive_mode: bool = False
    """Enable adaptive adjustment based on attention metrics."""

    adaptive_threshold: float = 0.5
    """Threshold for triggering adaptive adjustments."""

    adaptive_learning_rate: float = 0.1
    """Rate of adaptive parameter adjustment."""

    adaptive_metrics_window: int = 10
    """Window size for computing running metrics."""

    # === Debug/Inspection ===
    debug_attention: bool = False
    """Enable debug logging and metrics collection."""

    debug_log_interval: int = 10
    """Log metrics every N steps."""

    debug_save_matrices: bool = False
    """Save full attention matrices (memory intensive!)."""

    debug_output_dir: Optional[str] = None
    """Directory for debug outputs."""

    # === Advanced ===
    attention_bias_scale: float = 1.0
    """Global scale for all attention biases."""

    use_rope_relative: bool = True
    """Use relative positional encoding for bias computation."""

    clamp_bias: bool = True
    """Clamp attention bias to prevent numerical instability."""

    bias_clamp_value: float = 10.0
    """Maximum absolute bias value."""

    def __post_init__(self):
        """Validate configuration values."""
        # Clamp all strength parameters to [0, 1]
        for attr in [
            "reference_strength",
            "identity_lock",
            "prompt_adherence",
            "temporal_lock",
            "subject_separation",
            "audio_strength",
        ]:
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr} must be in [0, 1], got {value}")

        if not 0.0 <= self.identity_temporal_decay <= 1.0:
            raise ValueError("identity_temporal_decay must be in [0, 1]")

        if self.reference_boost_mode not in ("additive", "multiplicative"):
            raise ValueError("reference_boost_mode must be 'additive' or 'multiplicative'")

        if self.identity_spatial_scope not in ("global", "local", "patch"):
            raise ValueError("identity_spatial_scope must be 'global', 'local', or 'patch'")

        if self.temporal_mode not in ("smooth", "strict", "adaptive"):
            raise ValueError("temporal_mode must be 'smooth', 'strict', or 'adaptive'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionControlConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def copy(self) -> "AttentionControlConfig":
        """Create a deep copy."""
        return AttentionControlConfig.from_dict(self.to_dict())

    def with_overrides(self, **overrides) -> "AttentionControlConfig":
        """Create new config with overridden values."""
        new_data = self.to_dict()
        new_data.update(overrides)
        return self.from_dict(new_data)

    @property
    def is_active(self) -> bool:
        """Check if any control mechanism is active."""
        return any(
            [
                self.reference_strength > 0,
                self.identity_lock > 0,
                self.prompt_adherence > 0,
                self.temporal_lock > 0,
                self.subject_separation > 0,
                self.audio_strength > 0,
                self.adaptive_mode,
            ]
        )

    @property
    def active_mechanisms(self) -> List[str]:
        """List of active mechanism names."""
        mechanisms = []
        if self.reference_strength > 0:
            mechanisms.append("reference")
        if self.identity_lock > 0:
            mechanisms.append("identity")
        if self.prompt_adherence > 0:
            mechanisms.append("prompt")
        if self.temporal_lock > 0:
            mechanisms.append("temporal")
        if self.subject_separation > 0:
            mechanisms.append("subject_separation")
        if self.audio_strength > 0:
            mechanisms.append("audio")
        if self.adaptive_mode:
            mechanisms.append("adaptive")
        return mechanisms


# Preset configurations for common use cases
PRESETS = {
    "disabled": AttentionControlConfig(),
    "reference_only": AttentionControlConfig(
        reference_strength=0.3,
        reference_boost_mode="additive",
    ),
    "identity_preservation": AttentionControlConfig(
        reference_strength=0.2,
        identity_lock=0.4,
        identity_temporal_decay=0.97,
    ),
    "prompt_adherence": AttentionControlConfig(
        prompt_adherence=0.3,
        reference_strength=0.1,
    ),
    "temporal_consistency": AttentionControlConfig(
        temporal_lock=0.3,
        temporal_window=7,
        temporal_mode="smooth",
    ),
    "multi_subject": AttentionControlConfig(
        reference_strength=0.25,
        identity_lock=0.3,
        subject_separation=0.4,
        min_subject_distance=0.4,
    ),
    "audio_sync": AttentionControlConfig(
        audio_strength=0.3,
        audio_video_sync=True,
        reference_strength=0.15,
    ),
    "adaptive_balanced": AttentionControlConfig(
        reference_strength=0.2,
        identity_lock=0.2,
        prompt_adherence=0.15,
        temporal_lock=0.15,
        adaptive_mode=True,
        adaptive_threshold=0.45,
        adaptive_learning_rate=0.05,
    ),
    "maximum_coherence": AttentionControlConfig(
        reference_strength=0.4,
        identity_lock=0.5,
        prompt_adherence=0.3,
        temporal_lock=0.35,
        subject_separation=0.3,
        audio_strength=0.2,
        adaptive_mode=True,
        adaptive_threshold=0.4,
        adaptive_learning_rate=0.08,
        attention_bias_scale=1.2,
    ),
    "debug": AttentionControlConfig(
        reference_strength=0.3,
        identity_lock=0.3,
        debug_attention=True,
        debug_log_interval=5,
        debug_save_matrices=False,
    ),
}
