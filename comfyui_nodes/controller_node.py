"""
H3 Attention Controller Node for ComfyUI.

Main node to install the attention controller on a model.
"""

from .controller_node import H3AttentionControllerNode
from .settings_node import H3AttentionSettingsNode
from .debug_node import H3AttentionDebugNode

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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
