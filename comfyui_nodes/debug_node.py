"""
H3 Attention Debug Node for ComfyUI.

Captures and visualizes attention metrics during generation.
"""

import json
from typing import Dict, Any, Tuple, List, Optional
from ...h3_attention import H3AttentionController, AttentionMetrics


class H3AttentionDebugNode:
    """
    ComfyUI Node: Debug and visualize attention metrics.

    Connect to controller node to capture metrics during generation.
    Outputs metrics summary and can save to file.
    """

    def __init__(self):
        self._controller: Optional[H3AttentionController] = None
        self._captured_metrics: List[Dict] = []
        self._log_to_console = True
        self._save_to_file = False
        self._output_path = "h3_attention_metrics.json"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "controller": ("H3_ATTENTION_CONTROLLER",),
                "log_to_console": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Print metrics to console"},
                ),
                "save_to_file": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Save metrics to JSON file"},
                ),
                "output_path": (
                    "STRING",
                    {"default": "h3_attention_metrics.json", "tooltip": "Output file path"},
                ),
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

        def hooked_record(metrics: AttentionMetrics):
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
            return json.dumps({"error": "No controller registered"}, indent=2)

        summary = self._controller.get_metrics_summary()
        return json.dumps(summary, indent=2)

    def get_metrics_history(self) -> List[Dict]:
        """Get full metrics history."""
        return self._captured_metrics.copy()

    def save_metrics(self, path: Optional[str] = None):
        """Save metrics to file."""
        path = path or self._output_path
        with open(path, "w") as f:
            json.dump(self._captured_metrics, f, indent=2)

    def clear(self):
        """Clear captured metrics."""
        self._captured_metrics.clear()


NODE_CLASS_MAPPINGS = {
    "H3AttentionDebug": H3AttentionDebugNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AttentionDebug": "H3 Attention Debug",
}
