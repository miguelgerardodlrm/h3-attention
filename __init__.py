"""h3-attention: ComfyUI node pack for MiniMax H3 attention control.

ComfyUI loads a custom node directory by executing this file and reading
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS from it.
"""

try:
    from .comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Imported without a parent package (tooling, e.g. pytest collecting the
    # repository root). Re-import in absolute mode; genuine errors inside the
    # node package still surface from this second attempt.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from comfyui_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
