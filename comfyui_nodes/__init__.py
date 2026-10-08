"""
ComfyUI Integration for H3 Attention Control.

Provides three custom nodes:
1. H3AttentionControllerNode - Main controller installation
2. H3AttentionSettingsNode - Configuration presets and parameters
3. H3AttentionDebugNode - Metrics visualization and logging
"""

import torch
from typing import Dict, Any, Optional, Tuple, List
import json

# ComfyUI node registration
try:
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sd
    import folder_paths

    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False

    # Mock for type hints
    class ModelPatcher:
        pass

    class Model:
        pass


from ..h3_attention import (
    H3AttentionController,
    AttentionControlConfig,
    AttentionSchedule,
    PRESETS,
    SCHEDULE_PRESETS,
    install_controller_on_model,
    remove_controller_from_model,
)


class H3AttentionControllerNode:
    """
    ComfyUI Node: Install/configure H3 Attention Controller on a model.

    Connect between model loader and sampler.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (list(PRESETS.keys()) + ["custom"], {"default": "maximum_coherence"}),
                "schedule": (list(SCHEDULE_PRESETS.keys()) + ["none"], {"default": "balanced"}),
            },
            "optional": {
                "settings": ("H3_ATTENTION_SETTINGS",),
                "debug_node": ("H3_ATTENTION_DEBUG",),
            },
        }

    RETURN_TYPES = ("MODEL", "H3_ATTENTION_CONTROLLER")
    RETURN_NAMES = ("model", "controller")
    FUNCTION = "install_controller"
    CATEGORY = "MiniMax H3/Attention"

    def install_controller(
        self,
        model,
        preset: str,
        schedule: str,
        settings: Optional[Dict] = None,
        debug_node: Optional[Any] = None,
    ) -> Tuple[Any, H3AttentionController]:
        """Install attention controller on model."""

        # Get config
        if preset == "custom" and settings is not None:
            config = AttentionControlConfig.from_dict(settings)
        else:
            config = PRESETS.get(preset, PRESETS["maximum_coherence"]).copy()

        # Get schedule
        schedule_obj = None if schedule == "none" else SCHEDULE_PRESETS.get(schedule)

        # Install controller
        controller = H3AttentionController(config=config, schedule=schedule_obj)
        controller.install(model)

        # Connect debug node if provided
        if debug_node is not None and hasattr(debug_node, "register_controller"):
            debug_node.register_controller(controller)

        # Store controller reference on model for cleanup
        if not hasattr(model, "_h3_attention_controllers"):
            model._h3_attention_controllers = []
        model._h3_attention_controllers.append(controller)

        return (model, controller)


class H3AttentionSettingsNode:
    """
    ComfyUI Node: Configure attention control parameters.

    Provides all 7 control mechanisms with individual toggles.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "preset": (list(PRESETS.keys()), {"default": "maximum_coherence"}),
            },
            "optional": {
                # Reference
                "reference_strength": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "reference_boost_mode": (["additive", "multiplicative"], {"default": "additive"}),
                # Identity
                "identity_lock": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "identity_temporal_decay": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "identity_spatial_scope": (["global", "local", "patch"], {"default": "global"}),
                # Prompt
                "prompt_adherence": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                # Temporal
                "temporal_lock": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temporal_window": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "temporal_mode": (["smooth", "strict", "adaptive"], {"default": "smooth"}),
                # Subject Separation
                "subject_separation": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "min_subject_distance": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                # Audio
                "audio_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_video_sync": ("BOOLEAN", {"default": True}),
                # Adaptive
                "adaptive_mode": ("BOOLEAN", {"default": False}),
                "adaptive_threshold": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "adaptive_learning_rate": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
                # Debug
                "debug_attention": ("BOOLEAN", {"default": False}),
                "debug_log_interval": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
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


class H3AttentionDebugNode:
    """
    ComfyUI Node: Debug and visualize attention metrics.

    Connect to controller node to capture metrics during generation.
    """

    def __init__(self):
        self._controller: Optional[H3AttentionController] = None
        self._captured_metrics: List[Dict] = []

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "controller": ("H3_ATTENTION_CONTROLLER",),
                "log_to_console": ("BOOLEAN", {"default": True}),
                "save_to_file": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": "h3_attention_metrics.json"}),
            },
        }

    RETURN_TYPES = ("H3_ATTENTION_DEBUG", "STRING")
    RETURN_NAMES = ("debug_node", "metrics_summary")
    FUNCTION = "setup_debug"
    CATEGORY = "MiniMax H3/Attention"
    OUTPUT_NODE = True

    def setup_debug(
        self,
        controller: Optional[H3AttentionController] = None,
        log_to_console: bool = True,
        save_to_file: bool = False,
        output_path: str = "h3_attention_metrics.json",
    ) -> Tuple[Any, str]:
        """Setup debug node."""

        if controller is not None:
            self.register_controller(controller)

        self._log_to_console = log_to_console
        self._save_to_file = save_to_file
        self._output_path = output_path

        summary = self.get_summary()

        return (self, summary)

    def register_controller(self, controller: H3AttentionController):
        """Register controller for metrics capture."""
        self._controller = controller
        # Hook into controller's record_metrics
        original_record = controller.record_metrics

        def hooked_record(metrics):
            original_record(metrics)
            self._captured_metrics.append(metrics.to_dict())
            if self._log_to_console:
                print(
                    f"[H3-Attention] step={metrics.step:3d} "
                    f"layer={metrics.layer_idx:2d} "
                    f"ref_aff={metrics.reference_affinity:.3f} "
                    f"prompt_aff={metrics.prompt_affinity:.3f} "
                    f"temp_aff={metrics.temporal_affinity:.3f} "
                    f"bias={metrics.applied_bias_mean:.3f}"
                )

        controller.record_metrics = hooked_record

    def get_summary(self) -> str:
        """Get metrics summary as JSON string."""
        if self._controller is None:
            return "No controller registered"

        summary = self._controller.get_metrics_summary()
        return json.dumps(summary, indent=2)

    def get_metrics_history(self) -> List[Dict]:
        """Get full metrics history."""
        return self._captured_metrics.copy()

    def save_metrics(self, path: Optional[str] = None):
        """Save metrics to file."""
        import json

        path = path or self._output_path
        with open(path, "w") as f:
            json.dump(self._captured_metrics, f, indent=2)

    def clear(self):
        """Clear captured metrics."""
        self._captured_metrics.clear()


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "H3AttentionController": H3AttentionControllerNode,
    "H3AttentionSettings": H3AttentionSettingsNode,
    "H3AttentionDebug": H3AttentionDebugNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AttentionController": "H3 Attention Controller",
    "H3AttentionSettings": "H3 Attention Settings",
    "H3AttentionDebug": "H3 Attention Debug",
}
