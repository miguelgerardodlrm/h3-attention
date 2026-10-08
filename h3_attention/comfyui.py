"""
ComfyUI convenience functions for H3 Attention Control.
"""

from typing import Optional, Any
from .controller import H3AttentionController, AttentionControlConfig
from .scheduler import AttentionSchedule


def install_controller_on_model(
    model: Any,
    config: Optional[AttentionControlConfig] = None,
    preset: Optional[str] = None,
    schedule: Optional[AttentionSchedule] = None,
) -> H3AttentionController:
    """
    Convenience function to install controller on a model.

    Args:
        model: The MiniMax H3 model (pipeline or transformer)
        config: AttentionControlConfig
        preset: Preset name from PRESETS
        schedule: Optional AttentionSchedule

    Returns:
        Installed H3AttentionController
    """
    # Find transformer in model
    if hasattr(model, "transformer"):
        transformer = model.transformer
    elif hasattr(model, "transformer_ref"):
        transformer = model.transformer_ref
    elif hasattr(model, "model") and hasattr(model.model, "transformer"):
        transformer = model.model.transformer
    else:
        transformer = model

    controller = H3AttentionController(config=config, preset=preset, schedule=schedule)
    controller.install(transformer)
    return controller


def remove_controller_from_model(model: Any, controller: H3AttentionController):
    """Remove controller from model."""
    controller.uninstall()
