"""
ComfyUI Integration for H3 Attention Control.

Provides three custom nodes:
1. H3AttentionControllerNode - Main controller installation
2. H3AttentionSettingsNode - Configuration presets and parameters
3. H3AttentionDebugNode - Metrics visualization and logging

The classes live in their dedicated modules (also referenced by the package's
entry points); this package __init__ only aggregates the registration maps.
"""

try:
    from .controller_node import H3AttentionControllerNode
    from .settings_node import H3AttentionSettingsNode
    from .debug_node import H3AttentionDebugNode
except ImportError:  # pragma: no cover - flat (non-package) import fallback
    from comfyui_nodes.controller_node import H3AttentionControllerNode
    from comfyui_nodes.settings_node import H3AttentionSettingsNode
    from comfyui_nodes.debug_node import H3AttentionDebugNode

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

__all__ = [
    "H3AttentionControllerNode",
    "H3AttentionSettingsNode",
    "H3AttentionDebugNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
