"""
H3AttentionController: Main entry point for attention control.

Intercepts MiniMax H3 attention processors and applies configurable modifications.
Compatible with Sol-attn, SageAttn, LoRA, and standard attention backends.
"""

import logging
import warnings
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from contextlib import contextmanager

from .config import AttentionControlConfig, PRESETS
from .processor import H3ControlledAttnProcessor, H3AttnProcessorWrapper
from .metrics import AttentionMetrics
from .scheduler import AttentionSchedule

logger = logging.getLogger(__name__)

# Constants from MiniMax H3 (matching diffusers implementation)
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_TAG = 2
MINIMAX_H3_MODALITY_NUM = 3


@dataclass
class ControllerState:
    """Runtime state of the controller during inference."""

    step: int = 0
    total_steps: int = 0
    current_timestep: float = 0.0
    metrics_history: List[AttentionMetrics] = field(default_factory=list)
    adaptive_params: Dict[str, float] = field(default_factory=dict)
    reference_indices: Dict[str, torch.Tensor] = field(default_factory=dict)
    layout_info: Dict[str, Any] = field(default_factory=dict)


class H3AttentionController:
    """
    Controls attention patterns in MiniMax H3 during inference.

    Works by replacing/wrapping the attention processor in each transformer block.
    Supports all H3 variants (FL2VA, Ref2VA) and acceleration backends.

    Example:
        >>> controller = H3AttentionController(config=PRESETS["maximum_coherence"])
        >>> controller.install(model)
        >>> # ... run generation ...
        >>> controller.uninstall(model)
    """

    def __init__(
        self,
        config: Optional[AttentionControlConfig] = None,
        preset: Optional[str] = None,
        schedule: Optional[AttentionSchedule] = None,
    ):
        """
        Initialize controller.

        Args:
            config: AttentionControlConfig instance (takes priority over preset)
            preset: Name of preset from PRESETS dict
            schedule: Optional AttentionSchedule for timestep-dependent control
        """
        if config is not None and preset is not None:
            raise ValueError("Provide either config or preset, not both")

        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
            config = PRESETS[preset].copy()
        elif config is None:
            config = AttentionControlConfig()

        self.config = config
        self.schedule = schedule
        self.state = ControllerState()

        # Storage for original processors to enable clean uninstall
        self._original_processors: Dict[int, Any] = {}
        self._installed_transformer: Optional[nn.Module] = None
        self._processor_factories: Dict[int, Callable] = {}

        # Detect Sol-attn availability
        self._sol_attn_available = self._check_sol_attn()

        if self.config.debug_attention:
            self._setup_debug_logging()

    def _check_sol_attn(self) -> bool:
        """Check if HyperFlow Sol-attn is available."""
        try:
            from hyperflow_h3.sol_attn import HyperFlowSolAttnProcessor

            return True
        except ImportError:
            return False

    def _setup_debug_logging(self):
        """Configure debug logging."""
        logging.getLogger("h3_attention").setLevel(logging.DEBUG)
        if self.config.debug_output_dir:
            import os

            os.makedirs(self.config.debug_output_dir, exist_ok=True)

    def install(self, transformer: nn.Module) -> "H3AttentionController":
        """
        Install controller on a MiniMax H3 transformer.

        Args:
            transformer: MiniMaxH3Transformer3DModel instance

        Returns:
            Self for chaining
        """
        if not hasattr(transformer, "transformer_blocks"):
            raise ValueError(
                "Expected MiniMaxH3Transformer3DModel with 'transformer_blocks' attribute. "
                f"Got {type(transformer).__name__}"
            )

        if self._installed_transformer is not None:
            warnings.warn(
                "Controller already installed on another transformer. Call uninstall() first."
            )
            self.uninstall()

        self._installed_transformer = transformer
        self._original_processors = {}
        self._processor_factories = {}

        # Analyze layout if Ref2VA (need reference indices)
        self._analyze_layout(transformer)

        for i, block in enumerate(transformer.transformer_blocks):
            if not hasattr(block, "attn"):
                logger.warning(f"Block {i} has no 'attn' attribute, skipping")
                continue

            original_processor = block.attn.processor
            self._original_processors[i] = original_processor

            # Create controlled processor
            if isinstance(original_processor, type) and "SolAttn" in original_processor.__name__:
                # Wrap existing Sol-attn processor
                logger.info(
                    f"Block {i}: Wrapping Sol-attn processor ({original_processor.__class__.__name__})"
                )
                controlled = H3AttnProcessorWrapper(original_processor, self)
            elif self._sol_attn_available and self.config.adaptive_mode:
                # Could optionally enable Sol-attn here if desired
                logger.info(
                    f"Block {i}: Installing controlled processor (Sol-attn available but not used)"
                )
                controlled = H3ControlledAttnProcessor(self, layer_idx=i)
            else:
                # Standard processor replacement
                logger.info(f"Block {i}: Installing H3ControlledAttnProcessor")
                controlled = H3ControlledAttnProcessor(self, layer_idx=i)

            block.attn.set_processor(controlled)
            self._processor_factories[i] = lambda ctrl=self, idx=i: H3ControlledAttnProcessor(
                ctrl, layer_idx=idx
            )

        logger.info(f"Installed attention controller on {len(self._original_processors)} blocks")
        logger.info(f"Active mechanisms: {self.config.active_mechanisms}")

        return self

    def uninstall(self) -> "H3AttentionController":
        """
        Restore original attention processors.

        Returns:
            Self for chaining
        """
        if self._installed_transformer is None:
            logger.warning("Controller not installed, nothing to uninstall")
            return self

        for i, block in enumerate(self._installed_transformer.transformer_blocks):
            if i in self._original_processors:
                block.attn.set_processor(self._original_processors[i])

        logger.info(f"Uninstalled controller from {len(self._original_processors)} blocks")

        self._installed_transformer = None
        self._original_processors = {}
        self._processor_factories = {}
        self.state = ControllerState()

        return self

    @contextmanager
    def patch(self, transformer: nn.Module):
        """
        Context manager for temporary patching.

        Example:
            >>> with controller.patch(model):
            ...     output = model.generate(...)
        """
        self.install(transformer)
        try:
            yield self
        finally:
            self.uninstall()

    def _analyze_layout(self, transformer: nn.Module):
        """
        Analyze packed sequence layout to identify reference/target regions.

        This is called once at install time. For Ref2VA, we need to know
        where reference blocks end and target generation begins.
        """
        # The layout info is typically passed via the pipeline's denoiser step
        # We'll populate this during the first forward pass via the processor
        self.state.layout_info = {
            "num_layers": len(transformer.transformer_blocks),
            "heads": transformer.transformer_blocks[0].attn.heads
            if transformer.transformer_blocks
            else 0,
            "head_dim": transformer.transformer_blocks[0].attn.head_dim
            if transformer.transformer_blocks
            else 0,
        }

    def update_step(self, step: int, total_steps: int, timestep: float):
        """Update controller state for current denoising step."""
        self.state.step = step
        self.state.total_steps = total_steps
        self.state.current_timestep = timestep

        # Apply schedule if configured
        if self.schedule is not None:
            schedule_config = self.schedule.get_config_at_step(step, total_steps)
            # Merge schedule config with base config
            for key, value in schedule_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Adaptive mode adjustments
        if (
            self.config.adaptive_mode
            and len(self.state.metrics_history) >= self.config.adaptive_metrics_window
        ):
            self._adaptive_adjust()

    def _adaptive_adjust(self):
        """Adjust parameters based on recent metrics."""
        recent = self.state.metrics_history[-self.config.adaptive_metrics_window :]
        avg_ref_affinity = sum(m.reference_affinity for m in recent) / len(recent)
        avg_prompt_affinity = sum(m.prompt_affinity for m in recent) / len(recent)

        lr = self.config.adaptive_learning_rate
        threshold = self.config.adaptive_threshold

        # Adjust reference strength
        if avg_ref_affinity < threshold:
            new_strength = min(1.0, self.config.reference_strength + lr)
            self.config.reference_strength = new_strength
            logger.debug(f"Adaptive: increased reference_strength to {new_strength:.3f}")
        elif avg_ref_affinity > threshold + 0.1:
            new_strength = max(0.0, self.config.reference_strength - lr * 0.5)
            self.config.reference_strength = new_strength
            logger.debug(f"Adaptive: decreased reference_strength to {new_strength:.3f}")

        # Adjust prompt adherence
        if avg_prompt_affinity < threshold:
            new_adherence = min(1.0, self.config.prompt_adherence + lr)
            self.config.prompt_adherence = new_adherence
            logger.debug(f"Adaptive: increased prompt_adherence to {new_adherence:.3f}")

    def record_metrics(self, metrics: AttentionMetrics):
        """Record metrics from a processor call."""
        self.state.metrics_history.append(metrics)

        if self.config.debug_attention and self.state.step % self.config.debug_log_interval == 0:
            logger.debug(
                f"[H3-Attention] step={self.state.step} "
                f"ref_aff={metrics.reference_affinity:.3f} "
                f"prompt_aff={metrics.prompt_affinity:.3f} "
                f"temp_aff={metrics.temporal_affinity:.3f} "
                f"bias={metrics.applied_bias_mean:.3f}"
            )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.state.metrics_history:
            return {}

        m = self.state.metrics_history
        return {
            "steps_recorded": len(m),
            "avg_reference_affinity": sum(x.reference_affinity for x in m) / len(m),
            "avg_prompt_affinity": sum(x.prompt_affinity for x in m) / len(m),
            "avg_temporal_affinity": sum(x.temporal_affinity for x in m) / len(m),
            "avg_applied_bias": sum(x.applied_bias_mean for x in m) / len(m),
            "max_reference_affinity": max(x.reference_affinity for x in m),
            "min_reference_affinity": min(x.reference_affinity for x in m),
        }

    def set_reference_indices(
        self,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        ref_video_indices: Optional[torch.Tensor] = None,
        ref_audio_indices: Optional[torch.Tensor] = None,
    ):
        """Set indices for different modalities in packed sequence (called by processor)."""
        self.state.reference_indices = {
            "video": video_indices,
            "audio": audio_indices,
            "text": text_indices,
            "ref_video": ref_video_indices,
            "ref_audio": ref_audio_indices,
        }

    def compute_attention_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute attention bias based on control configuration.

        Returns bias tensor of shape [batch, heads, seq_len, seq_len] or
        broadcastable to that shape.
        """
        if not self.config.is_active:
            return torch.zeros_like(query[..., :1, :1])  # No bias

        batch, heads, seq_len, head_dim = query.shape
        device = query.device
        dtype = query.dtype

        # Initialize bias
        bias = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)

        # Get modality masks
        is_video = token_tags == MINIMAX_H3_VIDEO_TAG
        is_text = token_tags == MINIMAX_H3_TEXT_TAG
        is_audio = token_tags == MINIMAX_H3_AUDIO_TAG

        # === Reference Attention Boost ===
        if self.config.reference_strength > 0:
            bias += self._compute_reference_bias(
                query, key, token_tags, position_ids, is_video, is_text, is_audio
            )

        # === Identity Lock ===
        if self.config.identity_lock > 0:
            bias += self._compute_identity_bias(
                query, key, token_tags, position_ids, is_video, is_text
            )

        # === Prompt Adherence ===
        if self.config.prompt_adherence > 0:
            bias += self._compute_prompt_bias(query, key, token_tags, is_text, is_video, is_audio)

        # === Temporal Lock ===
        if self.config.temporal_lock > 0:
            bias += self._compute_temporal_bias(
                query, key, token_tags, position_ids, is_video, is_audio
            )

        # === Subject Separation ===
        if self.config.subject_separation > 0:
            bias += self._compute_subject_separation_bias(
                query, key, token_tags, position_ids, is_video
            )

        # === Audio Strength ===
        if self.config.audio_strength > 0:
            bias += self._compute_audio_bias(
                query, key, token_tags, position_ids, is_audio, is_video
            )

        # Scale and clamp
        bias = bias * self.config.attention_bias_scale
        if self.config.clamp_bias:
            bias = bias.clamp(-self.config.bias_clamp_value, self.config.bias_clamp_value)

        return bias

    def _compute_reference_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        is_video: torch.Tensor,
        is_text: torch.Tensor,
        is_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Boost attention from reference tokens to target tokens."""
        batch, heads, seq_len, _ = query.shape
        device = query.device
        dtype = query.dtype
        bias = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)

        # Find reference vs target regions
        ref_indices = self.state.reference_indices

        if "ref_video" in ref_indices and ref_indices["ref_video"] is not None:
            ref_vid = ref_indices["ref_video"]
            # Target video indices (generated portion)
            tgt_vid = ref_indices["video"]
            if tgt_vid is not None:
                # Boost: reference attends to target, target attends to reference
                strength = self.config.reference_strength
                if self.config.reference_boost_mode == "additive":
                    bias[:, :, tgt_vid.unsqueeze(1), ref_vid.unsqueeze(0)] += strength
                    bias[:, :, ref_vid.unsqueeze(1), tgt_vid.unsqueeze(0)] += strength * 0.5
                else:
                    # Multiplicative would be applied differently (scale attention weights)
                    pass

        if "ref_audio" in ref_indices and ref_indices["ref_audio"] is not None:
            ref_aud = ref_indices["ref_audio"]
            tgt_aud = ref_indices["audio"]
            if tgt_aud is not None:
                strength = self.config.reference_strength * 0.8
                if self.config.reference_boost_mode == "additive":
                    bias[:, :, tgt_aud.unsqueeze(1), ref_aud.unsqueeze(0)] += strength

        return bias

    def _compute_identity_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        is_video: torch.Tensor,
        is_text: torch.Tensor,
    ) -> torch.Tensor:
        """Maintain identity consistency between reference and generated frames."""
        batch, heads, seq_len, _ = query.shape
        device = query.device
        dtype = query.dtype
        bias = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)

        ref_indices = self.state.reference_indices
        identity_idx = self.config.identity_reference_idx

        # Use first video reference as identity anchor
        if "ref_video" in ref_indices and ref_indices["ref_video"] is not None:
            ref_vid = ref_indices["ref_video"]
            if len(ref_vid) > identity_idx:
                anchor_idx = ref_vid[identity_idx : identity_idx + 1]
                tgt_vid = ref_indices["video"]
                if tgt_vid is not None and len(tgt_vid) > 0:
                    strength = self.config.identity_lock
                    # Apply temporal decay
                    step_ratio = self.state.step / max(1, self.state.total_steps)
                    decayed_strength = strength * (
                        self.config.identity_temporal_decay ** (step_ratio * 10)
                    )

                    if self.config.identity_spatial_scope == "global":
                        # Global: all target frames attend to anchor
                        bias[:, :, tgt_vid.unsqueeze(1), anchor_idx.unsqueeze(0)] += (
                            decayed_strength
                        )
                    elif self.config.identity_spatial_scope == "local":
                        # Local: only nearby temporal positions
                        for i, t in enumerate(tgt_vid):
                            # Find temporal distance via position_ids
                            t_pos = position_ids[t, 0] if t < len(position_ids) else 0
                            a_pos = (
                                position_ids[anchor_idx.item(), 0]
                                if anchor_idx.item() < len(position_ids)
                                else 0
                            )
                            temp_dist = abs(t_pos - a_pos)
                            local_strength = decayed_strength * (0.9**temp_dist)
                            bias[:, :, t : t + 1, anchor_idx.unsqueeze(0)] += local_strength

        return bias

    def _compute_prompt_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_tags: torch.Tensor,
        is_text: torch.Tensor,
        is_video: torch.Tensor,
        is_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Strengthen attention from text tokens to generated modalities."""
        batch, heads, seq_len, _ = query.shape
        device = query.device
        dtype = query.dtype
        bias = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)

        ref_indices = self.state.reference_indices

        text_idx = ref_indices.get("text")
        if text_idx is not None and len(text_idx) > 0:
            strength = self.config.prompt_adherence

            # Text -> target video
            tgt_vid = ref_indices.get("video")
            if tgt_vid is not None:
                bias[:, :, tgt_vid.unsqueeze(1), text_idx.unsqueeze(0)] += strength

            # Text -> target audio
            tgt_aud = ref_indices.get("audio")
            if tgt_aud is not None:
                bias[:, :, tgt_aud.unsqueeze(1), text_idx.unsqueeze(0)] += strength * 0.8

        return bias

    def _compute_temporal_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        is_video: torch.Tensor,
        is_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Enforce temporal consistency across generated frames."""
        batch, heads, seq_len, _ = query.shape
        device = query.device
        dtype = query.dtype
        bias = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)

        ref_indices = self.state.reference_indices
        tgt_vid = ref_indices.get("video")
        tgt_aud = ref_indices.get("audio")

        window = self.config.temporal_window
        strength = self.config.temporal_lock

        if tgt_vid is not None and len(tgt_vid) > 1:
            # Create temporal smoothness bias
            for i in range(len(tgt_vid)):
                for j in range(max(0, i - window), min(len(tgt_vid), i + window + 1)):
                    if i != j:
                        if self.config.temporal_mode == "smooth":
                            dist = abs(i - j)
                            bias[
                                :, :, tgt_vid[i] : tgt_vid[i] + 1, tgt_vid[j] : tgt_vid[j] + 1
                            ] += strength * (0.8**dist)
                        elif self.config.temporal_mode == "strict":
                            bias[
                                :, :, tgt_vid[i] : tgt_vid[i] + 1, tgt_vid[j] : tgt_vid[j] + 1
                            ] += strength

        if tgt_aud is not None and len(tgt_aud) > 1:
            for i in range(len(tgt_aud)):
                for j in range(max(0, i - window), min(len(tgt_aud), i + window + 1)):
                    if i != j:
                        bias[:, :, tgt_aud[i] : tgt_aud[i] + 1, tgt_aud[j] : tgt_aud[j] + 1] += (
                            strength * 0.5
                        )

        return bias

    def _compute_subject_separation_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        is_video: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce cross-attention between different subject references."""
        batch, heads, seq_len, _ = query.shape
        device = query.device
        dtype = query.dtype
        bias = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)

        ref_indices = self.state.reference_indices

        # This requires knowing which reference corresponds to which subject
        # For now, apply repulsion between different reference blocks
        if "ref_video" in ref_indices and ref_indices["ref_video"] is not None:
            ref_vid = ref_indices["ref_video"]
            # In a real implementation, we'd segment by subject
            # Here we apply a small repulsive bias between distant reference frames
            strength = -self.config.subject_separation * self.config.min_subject_distance
            # This is a placeholder - real implementation needs subject segmentation

        return bias

    def _compute_audio_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        is_audio: torch.Tensor,
        is_video: torch.Tensor,
    ) -> torch.Tensor:
        """Boost audio-video cross-modal attention."""
        batch, heads, seq_len, _ = query.shape
        device = query.device
        dtype = query.dtype
        bias = torch.zeros(batch, heads, seq_len, seq_len, device=device, dtype=dtype)

        ref_indices = self.state.reference_indices

        tgt_vid = ref_indices.get("video")
        tgt_aud = ref_indices.get("audio")
        ref_aud = ref_indices.get("ref_audio")

        strength = self.config.audio_strength

        # Target video <-> target audio sync
        if tgt_vid is not None and tgt_aud is not None and self.config.audio_video_sync:
            # Align by temporal position
            min_len = min(len(tgt_vid), len(tgt_aud))
            for i in range(min_len):
                bias[:, :, tgt_vid[i] : tgt_vid[i] + 1, tgt_aud[i] : tgt_aud[i] + 1] += strength
                bias[:, :, tgt_aud[i] : tgt_aud[i] + 1, tgt_vid[i] : tgt_vid[i] + 1] += strength

        # Reference audio -> target audio
        if ref_aud is not None and tgt_aud is not None:
            bias[:, :, tgt_aud.unsqueeze(1), ref_aud.unsqueeze(0)] += strength * 0.7

        return bias


def install_controller_on_model(
    model: nn.Module,
    config: Optional[AttentionControlConfig] = None,
    preset: Optional[str] = None,
    schedule: Optional[AttentionSchedule] = None,
) -> H3AttentionController:
    """
    Convenience function to install controller on a model.

    Args:
        model: The MiniMax H3 model (pipeline or transformer)
        config: AttentionControlConfig
        preset: Preset name
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


def remove_controller_from_model(model: nn.Module, controller: H3AttentionController):
    """Remove controller from model."""
    controller.uninstall()
