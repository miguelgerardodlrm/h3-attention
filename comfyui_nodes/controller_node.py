"""
H3 Attention Controller node for ComfyUI.

Installs the native (block-patch) attention intervention on a MiniMax H3
MODEL. Works between any model chain (loaders -> acceleration -> controller ->
guiders/samplers). Fail-safe: if the model is not a native H3 diffusion model,
or installation raises, the model is passed through unchanged.
"""

import logging
from typing import Any, Dict, Optional, Tuple

try:
    from ..h3_attention import (
        H3AttentionController,
        AttentionControlConfig,
        PRESETS,
        SCHEDULE_PRESETS,
        install_comfyui,
    )
except ImportError:  # installed as a top-level package (pip)
    from h3_attention import (
        H3AttentionController,
        AttentionControlConfig,
        PRESETS,
        SCHEDULE_PRESETS,
        install_comfyui,
    )

logger = logging.getLogger(__name__)


class H3AttentionControllerNode:
    """
    ComfyUI Node: Install/configure H3 Attention Controller on a model.

    Connect between model loader/acceleration chain and the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (
                    list(PRESETS.keys()) + ["custom"],
                    {"default": "maximum_coherence"},
                ),
                "schedule": (
                    list(SCHEDULE_PRESETS.keys()) + ["none"],
                    {"default": "balanced"},
                ),
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
        model: Any,
        preset: str,
        schedule: str,
        settings: Optional[Dict] = None,
        debug_node: Optional[Any] = None,
    ) -> Tuple[Any, H3AttentionController]:
        """Install attention controller on model (native ComfyUI path)."""

        # Config: explicit Settings node wins when preset == "custom".
        if preset == "custom" and settings is not None:
            config = AttentionControlConfig.from_dict(settings)
        else:
            config = PRESETS.get(preset, PRESETS["maximum_coherence"]).copy()

        schedule_obj = None if schedule == "none" else SCHEDULE_PRESETS.get(schedule)

        controller = H3AttentionController(config=config, schedule=schedule_obj)

        patched = model
        installed = False
        try:
            patched, installed = install_comfyui(model, controller)
        except Exception:
            logger.exception(
                "h3-attention: native install failed; passing the model through unchanged"
            )
            patched = model

        if not installed:
            logger.warning(
                "h3-attention: controller ready but inactive (model passed through unchanged)"
            )

        # Connect debug node if provided
        if debug_node is not None and hasattr(debug_node, "register_controller"):
            debug_node.register_controller(controller)

        # Store controller reference on the patched model for introspection
        try:
            if not hasattr(patched, "_h3_attention_controllers"):
                patched._h3_attention_controllers = []
            patched._h3_attention_controllers.append(controller)
        except Exception:  # pragma: no cover - attribute storage is best effort
            pass

        return (patched, controller)


NODE_CLASS_MAPPINGS = {
    "H3AttentionController": H3AttentionControllerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AttentionController": "H3 Attention Controller",
}
