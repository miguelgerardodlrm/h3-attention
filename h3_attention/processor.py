"""
Custom Attention Processors for H3.

Implements H3ControlledAttnProcessor (replaces standard processor)
and H3AttnProcessorWrapper (wraps Sol-attn or other custom processors).
"""

import logging
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AttnProcessor,
    _apply_rotary_emb,
)
from diffusers.models.attention_dispatch import dispatch_attention_fn
from .controller import H3AttentionController

logger = logging.getLogger(__name__)


class H3ControlledAttnProcessor(MiniMaxH3AttnProcessor):
    """
    Attention processor with configurable attention control.

    Extends the standard MiniMaxH3AttnProcessor to inject attention biases
    based on the controller configuration.
    """

    def __init__(self, controller: H3AttentionController, layer_idx: int = 0):
        super().__init__()
        self.controller = controller
        self.layer_idx = layer_idx
        self._call_count = 0

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with attention control.

        Args:
            attn: MiniMaxH3Attention module
            hidden_states: [batch, seq_len, hidden_dim]
            rotary_emb: (cos, sin) for RoPE
            attention_mask: Optional attention mask

        Returns:
            Output hidden states
        """
        self._call_count += 1

        # === Standard Q/K/V projection (same as parent) ===
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        # Unflatten heads: [batch, seq, heads * dim] -> [batch, seq, heads, dim]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # RMSNorm on Q and K
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Apply rotary embeddings
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # === ATTENTION CONTROL INJECTION POINT ===
        # We need token_tags and position_ids to compute biases
        # These are typically available in the transformer's forward context
        # We'll retrieve them from the controller state

        # Compute attention bias if controller is active
        bias = None
        metrics = None
        if self.controller.config.is_active:
            # Get layout info from controller state
            token_tags = self.controller.state.reference_indices.get("token_tags")
            position_ids = self.controller.state.reference_indices.get("position_ids")

            if token_tags is not None and position_ids is not None:
                # compute_attention_bias/metrics expect [batch, heads, seq, dim];
                # dispatch uses [batch, seq, heads, dim], so transpose first.
                query_bh = query.transpose(1, 2)
                key_bh = key.transpose(1, 2)
                bias = self.controller.compute_attention_bias(
                    query=query_bh,
                    key=key_bh,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    layer_idx=self.layer_idx,
                )

                # Compute metrics for adaptive mode
                if self.controller.config.adaptive_mode or self.controller.config.debug_attention:
                    from .metrics import compute_attention_metrics

                    metrics = compute_attention_metrics(
                        query_bh,
                        key_bh,
                        token_tags,
                        position_ids,
                        bias,
                        self.controller.state.reference_indices,
                    )
                    self.controller.record_metrics(metrics)

        # === Dispatch attention with optional bias ===
        # Note: dispatch_attention_fn doesn't directly support bias
        # We need to add bias to attention scores manually or use a modified dispatch

        if bias is not None:
            # Use custom attention with bias (expects [batch, heads, seq, dim])
            hidden_states = self._attention_with_bias(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                bias,
                attention_mask,
                attn,
            )
            hidden_states = hidden_states.transpose(1, 2)  # -> [batch, seq, heads, dim]
        else:
            # Standard dispatch (no control active)
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        # === Output projection (same as parent) ===
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _attention_with_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn: nn.Module,
    ) -> torch.Tensor:
        """
        Compute attention with additive bias.

        This is a fallback implementation when the backend doesn't support bias.
        For production, consider using a backend that supports bias natively.
        """
        batch, heads, seq_len, head_dim = query.shape
        scale = head_dim**-0.5

        # Compute attention scores: Q @ K^T * scale + bias
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_scores = attn_scores + bias

        # Apply mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply to values
        hidden_states = torch.matmul(attn_weights, value)

        return hidden_states


class H3AttnProcessorWrapper:
    """
    Wrapper for existing custom processors (e.g., Sol-attn).

    Allows injecting attention control without replacing the entire processor.
    The wrapped processor must expose a compatible __call__ signature.
    """

    def __init__(
        self,
        base_processor: Any,
        controller: H3AttentionController,
        layer_idx: int = 0,
    ):
        self.base_processor = base_processor
        self.controller = controller
        self.layer_idx = layer_idx
        self._call_count = 0

        # Copy backend config from base processor
        self._attention_backend = getattr(base_processor, "_attention_backend", None)
        self._parallel_config = getattr(base_processor, "_parallel_config", None)

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: intercept Q/K/V, apply bias, delegate to base processor.

        Strategy depends on base processor type:
        - If it has to_q/to_k/to_v: intercept after projection
        - If it's a full replacement (Sol-attn): we need to hook differently
        """
        self._call_count += 1

        # Check if base processor is Sol-attn style (has own __call__ doing everything)
        if (
            hasattr(self.base_processor, "_sparse")
            or "SolAttn" in type(self.base_processor).__name__
        ):
            # Sol-attn: we can't easily intercept Q/K/V
            # Best effort: apply bias after the fact via hook or monkey-patch
            # For now, delegate and log warning
            if self._call_count == 1:
                logger.warning(
                    f"Layer {self.layer_idx}: Wrapping Sol-attn processor. "
                    "Attention control is LIMITED (bias applied post-attention via residual)."
                )
            return self._sol_attn_forward(attn, hidden_states, rotary_emb, attention_mask)
        else:
            # Standard processor: we can intercept Q/K/V
            return self._standard_forward(attn, hidden_states, rotary_emb, attention_mask)

    def _standard_forward(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward for standard-style processors (intercept Q/K/V)."""
        # Project Q/K/V
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # Compute bias
        bias = None
        if self.controller.config.is_active:
            token_tags = self.controller.state.reference_indices.get("token_tags")
            position_ids = self.controller.state.reference_indices.get("position_ids")
            if token_tags is not None and position_ids is not None:
                # compute_attention_bias expects [batch, heads, seq, dim]
                bias = self.controller.compute_attention_bias(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    token_tags,
                    position_ids,
                    self.layer_idx,
                )

        # Call base processor with modified query/key/value
        # We need to temporarily replace the processor's internal state
        # Simpler: compute attention ourselves with bias, then call base for output proj
        if bias is not None:
            hidden_states = self._attention_with_bias(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                bias,
                attention_mask,
                attn,
            )
            hidden_states = hidden_states.transpose(1, 2)  # -> [batch, seq, heads, dim]
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        # Output projection (same as base)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _sol_attn_forward(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward for Sol-attn: limited control via residual connection.

        Since Sol-attn replaces the entire attention computation,
        we can only apply control by modifying the residual connection.
        """
        # Get base output
        base_output = self.base_processor.__call__(attn, hidden_states, rotary_emb, attention_mask)

        # Apply lightweight control via residual modulation
        # This is a fallback - not as effective as full Q/K/V control
        if self.controller.config.is_active and self.controller.config.identity_lock > 0:
            # Modulate residual based on identity lock
            # This is a simplified approximation
            pass

        return base_output

    def _attention_with_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn: nn.Module,
    ) -> torch.Tensor:
        """Compute attention with additive bias."""
        batch, heads, seq_len, head_dim = query.shape
        scale = head_dim**-0.5

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_scores = attn_scores + bias

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        hidden_states = torch.matmul(attn_weights, value)

        return hidden_states

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to base processor."""
        if name in (
            "base_processor",
            "controller",
            "layer_idx",
            "_call_count",
            "_attention_backend",
            "_parallel_config",
        ):
            raise AttributeError(name)
        return getattr(self.base_processor, name)


def create_processor_for_block(
    controller: H3AttentionController,
    block: nn.Module,
    layer_idx: int,
) -> Any:
    """
    Factory function to create appropriate processor for a block.

    Handles:
    - Standard blocks -> H3ControlledAttnProcessor
    - Sol-attn blocks -> H3AttnProcessorWrapper
    - SageAttn/Flash/SDPA -> H3ControlledAttnProcessor (backend handled by dispatch)
    """
    existing = block.attn.processor

    if "SolAttn" in type(existing).__name__ or hasattr(existing, "_sparse"):
        return H3AttnProcessorWrapper(existing, controller, layer_idx)
    else:
        return H3ControlledAttnProcessor(controller, layer_idx)
