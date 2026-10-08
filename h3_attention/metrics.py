"""
Attention Metrics for monitoring and adaptive control.

Computes interpretable metrics from attention patterns during inference.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import torch
import torch.nn.functional as F


@dataclass
class AttentionMetrics:
    """Metrics computed from a single attention layer/step."""

    # Affinity metrics (0-1, higher = more attention to target modality)
    reference_affinity: float = 0.0
    """Mean attention from target tokens to reference tokens."""

    prompt_affinity: float = 0.0
    """Mean attention from target tokens to text tokens."""

    temporal_affinity: float = 0.0
    """Mean attention between adjacent temporal positions."""

    audio_video_affinity: float = 0.0
    """Mean cross-modal attention between audio and video."""

    # Bias metrics
    applied_bias_mean: float = 0.0
    """Mean absolute value of applied attention bias."""

    applied_bias_max: float = 0.0
    """Max absolute value of applied attention bias."""

    # Entropy/diversity
    attention_entropy: float = 0.0
    """Entropy of attention distribution (higher = more diffuse)."""

    attention_concentration: float = 0.0
    """Concentration of attention (1 - entropy/max_entropy)."""

    # Layer/step info
    layer_idx: int = 0
    step: int = 0
    timestep: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reference_affinity": self.reference_affinity,
            "prompt_affinity": self.prompt_affinity,
            "temporal_affinity": self.temporal_affinity,
            "audio_video_affinity": self.audio_video_affinity,
            "applied_bias_mean": self.applied_bias_mean,
            "applied_bias_max": self.applied_bias_max,
            "attention_entropy": self.attention_entropy,
            "attention_concentration": self.attention_concentration,
            "layer_idx": self.layer_idx,
            "step": self.step,
            "timestep": self.timestep,
        }


def compute_attention_metrics(
    query: torch.Tensor,
    key: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    bias: Optional[torch.Tensor],
    reference_indices: Dict[str, Any],
    layer_idx: int = 0,
    step: int = 0,
    timestep: float = 0.0,
) -> AttentionMetrics:
    """
    Compute interpretable metrics from attention tensors.

    Args:
        query: [batch, heads, seq_len, head_dim]
        key: [batch, heads, seq_len, head_dim]
        token_tags: [seq_len] modality tags
        position_ids: [seq_len, 3] (t, h, w)
        bias: Optional applied bias [batch, heads, seq_len, seq_len]
        reference_indices: Dict with video/audio/text/ref indices
        layer_idx: Current layer index
        step: Current denoising step
        timestep: Current timestep value

    Returns:
        AttentionMetrics instance
    """
    batch, heads, seq_len, head_dim = query.shape
    device = query.device

    # Compute attention scores (without softmax for metrics)
    scale = head_dim**-0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B, H, S, S]

    if bias is not None:
        scores = scores + bias

    attn_weights = torch.softmax(scores, dim=-1)  # [B, H, S, S]

    # Modality masks
    is_video = token_tags == 0  # MINIMAX_H3_VIDEO_TAG
    is_text = token_tags == 1  # MINIMAX_H3_TEXT_TAG
    is_audio = token_tags == 2  # MINIMAX_H3_AUDIO_TAG

    # Get indices
    vid_idx = reference_indices.get("video")
    aud_idx = reference_indices.get("audio")
    txt_idx = reference_indices.get("text")
    ref_vid_idx = reference_indices.get("ref_video")
    ref_aud_idx = reference_indices.get("ref_audio")

    metrics = AttentionMetrics(layer_idx=layer_idx, step=step, timestep=timestep)

    # === Reference Affinity ===
    # Mean attention from target video/audio to reference video/audio
    ref_affinities = []
    if (
        vid_idx is not None
        and ref_vid_idx is not None
        and len(vid_idx) > 0
        and len(ref_vid_idx) > 0
    ):
        # Target video attending to reference video
        tgt_to_ref = attn_weights[:, :, vid_idx.unsqueeze(1), ref_vid_idx.unsqueeze(0)]
        ref_affinities.append(tgt_to_ref.mean().item())

    if (
        aud_idx is not None
        and ref_aud_idx is not None
        and len(aud_idx) > 0
        and len(ref_aud_idx) > 0
    ):
        tgt_to_ref = attn_weights[:, :, aud_idx.unsqueeze(1), ref_aud_idx.unsqueeze(0)]
        ref_affinities.append(tgt_to_ref.mean().item())

    metrics.reference_affinity = (
        sum(ref_affinities) / len(ref_affinities) if ref_affinities else 0.0
    )

    # === Prompt Affinity ===
    # Mean attention from target modalities to text tokens
    prompt_affinities = []
    if vid_idx is not None and txt_idx is not None and len(vid_idx) > 0 and len(txt_idx) > 0:
        tgt_to_txt = attn_weights[:, :, vid_idx.unsqueeze(1), txt_idx.unsqueeze(0)]
        prompt_affinities.append(tgt_to_txt.mean().item())

    if aud_idx is not None and txt_idx is not None and len(aud_idx) > 0 and len(txt_idx) > 0:
        tgt_to_txt = attn_weights[:, :, aud_idx.unsqueeze(1), txt_idx.unsqueeze(0)]
        prompt_affinities.append(tgt_to_txt.mean().item())

    metrics.prompt_affinity = (
        sum(prompt_affinities) / len(prompt_affinities) if prompt_affinities else 0.0
    )

    # === Temporal Affinity ===
    # Attention between adjacent frames in target video
    temp_affinities = []
    if vid_idx is not None and len(vid_idx) > 1:
        for i in range(len(vid_idx) - 1):
            # Frame i attending to frame i+1
            adj_attn = attn_weights[
                :, :, vid_idx[i] : vid_idx[i] + 1, vid_idx[i + 1] : vid_idx[i + 1] + 1
            ]
            temp_affinities.append(adj_attn.mean().item())

    metrics.temporal_affinity = (
        sum(temp_affinities) / len(temp_affinities) if temp_affinities else 0.0
    )

    # === Audio-Video Affinity ===
    av_affinities = []
    if vid_idx is not None and aud_idx is not None and len(vid_idx) > 0 and len(aud_idx) > 0:
        # Cross-modal attention
        v2a = attn_weights[:, :, vid_idx.unsqueeze(1), aud_idx.unsqueeze(0)]
        a2v = attn_weights[:, :, aud_idx.unsqueeze(1), vid_idx.unsqueeze(0)]
        av_affinities.extend([v2a.mean().item(), a2v.mean().item()])

    metrics.audio_video_affinity = sum(av_affinities) / len(av_affinities) if av_affinities else 0.0

    # === Bias Metrics ===
    if bias is not None:
        metrics.applied_bias_mean = bias.abs().mean().item()
        metrics.applied_bias_max = bias.abs().max().item()

    # === Entropy ===
    # Compute entropy of attention distribution per query position
    eps = 1e-8
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)  # [B, H, S]
    max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float, device=device))
    metrics.attention_entropy = (entropy / max_entropy).mean().item()
    metrics.attention_concentration = 1.0 - metrics.attention_entropy

    return metrics


def compute_reference_affinity(
    attn_weights: torch.Tensor,
    target_indices: torch.Tensor,
    reference_indices: torch.Tensor,
) -> float:
    """Fraction of the target's attention mass that flows to reference tokens.

    Robust to target rows carrying zero attention (unlike a per-cell mean).
    """
    if target_indices is None or reference_indices is None:
        return 0.0
    if len(target_indices) == 0 or len(reference_indices) == 0:
        return 0.0

    ref_mass = attn_weights[:, :, target_indices.unsqueeze(1), reference_indices.unsqueeze(0)].sum(
        dim=-1
    )
    total_mass = attn_weights[:, :, target_indices, :].sum(dim=-1)
    denom = total_mass.sum().clamp(min=1e-8)
    return (ref_mass.sum() / denom).clamp(0.0, 1.0).item()


def compute_prompt_affinity(
    attn_weights: torch.Tensor,
    target_indices: torch.Tensor,
    text_indices: torch.Tensor,
) -> float:
    """Fraction of the target's attention mass that flows to text tokens."""
    if target_indices is None or text_indices is None:
        return 0.0
    if len(target_indices) == 0 or len(text_indices) == 0:
        return 0.0

    txt_mass = attn_weights[:, :, target_indices.unsqueeze(1), text_indices.unsqueeze(0)].sum(
        dim=-1
    )
    total_mass = attn_weights[:, :, target_indices, :].sum(dim=-1)
    denom = total_mass.sum().clamp(min=1e-8)
    return (txt_mass.sum() / denom).clamp(0.0, 1.0).item()


def compute_temporal_affinity(
    attn_weights: torch.Tensor,
    video_indices: torch.Tensor,
    window: int = 1,
) -> float:
    """Compute mean attention between temporally adjacent video tokens."""
    if video_indices is None or len(video_indices) <= window:
        return 0.0

    affinities = []
    for i in range(len(video_indices) - window):
        adj_attn = attn_weights[
            :,
            :,
            video_indices[i] : video_indices[i] + 1,
            video_indices[i + window] : video_indices[i + window] + 1,
        ]
        affinities.append(adj_attn.mean().item())

    return sum(affinities) / len(affinities) if affinities else 0.0


def compute_subject_separation(
    attn_weights: torch.Tensor,
    subject_a_indices: torch.Tensor,
    subject_b_indices: torch.Tensor,
) -> float:
    """Cross-attention fraction between two subjects (lower = better separation)."""
    if subject_a_indices is None or subject_b_indices is None:
        return 0.0
    if len(subject_a_indices) == 0 or len(subject_b_indices) == 0:
        return 0.0

    a_to_b = attn_weights[:, :, subject_a_indices.unsqueeze(1), subject_b_indices.unsqueeze(0)].sum(
        dim=-1
    )
    b_to_a = attn_weights[:, :, subject_b_indices.unsqueeze(1), subject_a_indices.unsqueeze(0)].sum(
        dim=-1
    )
    a_total = attn_weights[:, :, subject_a_indices, :].sum(dim=-1)
    b_total = attn_weights[:, :, subject_b_indices, :].sum(dim=-1)

    frac_ab = (a_to_b.sum() / a_total.sum().clamp(min=1e-8)).clamp(0.0, 1.0)
    frac_ba = (b_to_a.sum() / b_total.sum().clamp(min=1e-8)).clamp(0.0, 1.0)
    return ((frac_ab + frac_ba) / 2).item()


def aggregate_metrics(metrics_list: list[AttentionMetrics]) -> Dict[str, float]:
    """Aggregate metrics across layers/steps."""
    if not metrics_list:
        return {}

    keys = [
        "reference_affinity",
        "prompt_affinity",
        "temporal_affinity",
        "audio_video_affinity",
        "applied_bias_mean",
        "applied_bias_max",
        "attention_entropy",
        "attention_concentration",
    ]

    result = {}
    for key in keys:
        values = [getattr(m, key) for m in metrics_list]
        result[f"mean_{key}"] = sum(values) / len(values)
        result[f"max_{key}"] = max(values)
        result[f"min_{key}"] = min(values)
        result[f"std_{key}"] = torch.tensor(values).std().item() if len(values) > 1 else 0.0

    return result
