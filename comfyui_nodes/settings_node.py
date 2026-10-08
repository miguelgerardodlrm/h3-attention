"""
H3 Attention Settings Node for ComfyUI.

Provides detailed configuration of all attention control mechanisms.
"""

from typing import Dict, Any, Tuple
from ...h3_attention import PRESETS, AttentionControlConfig


class H3AttentionSettingsNode:
    """
    ComfyUI Node: Configure attention control parameters.

    Provides all 7 control mechanisms with individual toggles.
    Output connects to H3AttentionController's optional settings input.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": (list(PRESETS.keys()), {"default": "maximum_coherence"}),
            },
            "optional": {
                # Reference Attention
                "reference_strength": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Boost attention from reference tokens to target generation",
                    },
                ),
                "reference_boost_mode": (
                    ["additive", "multiplicative"],
                    {"default": "additive", "tooltip": "How to apply reference boost"},
                ),
                # Identity Lock
                "identity_lock": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Preserve identity consistency between reference and generated frames",
                    },
                ),
                "identity_temporal_decay": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Decay factor for identity influence over time",
                    },
                ),
                "identity_spatial_scope": (
                    ["global", "local", "patch"],
                    {"default": "global", "tooltip": "Scope of identity lock"},
                ),
                # Prompt Adherence
                "prompt_adherence": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Strengthen attention from text tokens to generated content",
                    },
                ),
                # Temporal Control
                "temporal_lock": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Enforce temporal consistency across generated frames",
                    },
                ),
                "temporal_window": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 20,
                        "step": 1,
                        "tooltip": "Number of frames for temporal consistency",
                    },
                ),
                "temporal_mode": (
                    ["smooth", "strict", "adaptive"],
                    {"default": "smooth", "tooltip": "Temporal control mode"},
                ),
                # Subject Separation
                "subject_separation": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Reduce cross-attention between different subjects",
                    },
                ),
                "min_subject_distance": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Minimum attention distance between subjects",
                    },
                ),
                # Audio Strength
                "audio_strength": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Boost audio-video cross-modal attention",
                    },
                ),
                "audio_video_sync": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Maintain audio-video synchronization"},
                ),
                # Adaptive Mode
                "adaptive_mode": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable adaptive parameter adjustment"},
                ),
                "adaptive_threshold": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Threshold for triggering adaptive adjustments",
                    },
                ),
                "adaptive_learning_rate": (
                    "FLOAT",
                    {
                        "default": 0.08,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Rate of adaptive parameter adjustment",
                    },
                ),
                # Debug
                "debug_attention": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable debug logging"},
                ),
                "debug_log_interval": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Log metrics every N steps",
                    },
                ),
            },
        }

    RETURN_TYPES = ("H3_ATTENTION_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "create_settings"
    CATEGORY = "MiniMax H3/Attention"

    def create_settings(
        self,
        preset: str,
        reference_strength: float = 0.3,
        reference_boost_mode: str = "additive",
        identity_lock: float = 0.4,
        identity_temporal_decay: float = 0.95,
        identity_spatial_scope: str = "global",
        prompt_adherence: float = 0.25,
        temporal_lock: float = 0.3,
        temporal_window: int = 5,
        temporal_mode: str = "smooth",
        subject_separation: float = 0.25,
        min_subject_distance: float = 0.3,
        audio_strength: float = 0.2,
        audio_video_sync: bool = True,
        adaptive_mode: bool = False,
        adaptive_threshold: float = 0.45,
        adaptive_learning_rate: float = 0.08,
        debug_attention: bool = False,
        debug_log_interval: int = 10,
    ) -> Tuple[Dict]:
        """Create settings dictionary from parameters."""

        # Start with preset
        config = PRESETS.get(preset, PRESETS["maximum_coherence"]).copy()

        # Override with explicit parameters
        config.reference_strength = reference_strength
        config.reference_boost_mode = reference_boost_mode
        config.identity_lock = identity_lock
        config.identity_temporal_decay = identity_temporal_decay
        config.identity_spatial_scope = identity_spatial_scope
        config.prompt_adherence = prompt_adherence
        config.temporal_lock = temporal_lock
        config.temporal_window = temporal_window
        config.temporal_mode = temporal_mode
        config.subject_separation = subject_separation
        config.min_subject_distance = min_subject_distance
        config.audio_strength = audio_strength
        config.audio_video_sync = audio_video_sync
        config.adaptive_mode = adaptive_mode
        config.adaptive_threshold = adaptive_threshold
        config.adaptive_learning_rate = adaptive_learning_rate
        config.debug_attention = debug_attention
        config.debug_log_interval = debug_log_interval

        return (config.to_dict(),)


NODE_CLASS_MAPPINGS = {
    "H3AttentionSettings": H3AttentionSettingsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AttentionSettings": "H3 Attention Settings",
}
