"""
Unit tests for Attention Metrics.
"""

import pytest
import torch

from h3_attention import (
    AttentionMetrics,
    compute_attention_metrics,
    compute_reference_affinity,
    compute_prompt_affinity,
    compute_temporal_affinity,
    compute_subject_separation,
    aggregate_metrics,
)


class TestAttentionMetrics:
    """Tests for AttentionMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = AttentionMetrics(
            reference_affinity=0.5,
            prompt_affinity=0.4,
            temporal_affinity=0.3,
            audio_video_affinity=0.2,
            applied_bias_mean=0.1,
            applied_bias_max=0.5,
            attention_entropy=0.7,
            attention_concentration=0.3,
            layer_idx=5,
            step=10,
            timestep=0.5,
        )

        assert metrics.reference_affinity == 0.5
        assert metrics.layer_idx == 5

    def test_to_dict(self):
        """Test serialization."""
        metrics = AttentionMetrics(reference_affinity=0.5, layer_idx=3, step=7)
        data = metrics.to_dict()

        assert data["reference_affinity"] == 0.5
        assert data["layer_idx"] == 3
        assert data["step"] == 7


class TestComputeMetrics:
    """Tests for metric computation functions."""

    def setup_method(self):
        """Set up test tensors."""
        self.batch = 1
        self.heads = 4
        self.seq_len = 50
        self.head_dim = 64

        self.query = torch.randn(self.batch, self.heads, self.seq_len, self.head_dim)
        self.key = torch.randn(self.batch, self.heads, self.seq_len, self.head_dim)

        # Token tags: 10 text, 20 video, 10 audio, 10 video
        self.token_tags = torch.tensor([1] * 10 + [0] * 20 + [2] * 10 + [0] * 10)
        self.position_ids = torch.randint(0, 10, (self.seq_len, 3))

        self.reference_indices = {
            "video": torch.tensor(list(range(10, 30)) + list(range(40, 50))),
            "audio": torch.tensor(list(range(30, 40))),
            "text": torch.tensor(list(range(10))),
            "ref_video": torch.tensor(list(range(10, 15))),
            "ref_audio": torch.tensor(list(range(30, 32))),
        }

    def test_compute_attention_metrics(self):
        """Test full metrics computation."""
        bias = torch.zeros(self.batch, self.heads, self.seq_len, self.seq_len)

        metrics = compute_attention_metrics(
            self.query,
            self.key,
            self.token_tags,
            self.position_ids,
            bias,
            self.reference_indices,
            layer_idx=0,
            step=5,
            timestep=0.5,
        )

        assert isinstance(metrics, AttentionMetrics)
        assert 0.0 <= metrics.reference_affinity <= 1.0
        assert 0.0 <= metrics.prompt_affinity <= 1.0
        assert 0.0 <= metrics.temporal_affinity <= 1.0
        assert 0.0 <= metrics.attention_entropy <= 1.0
        assert 0.0 <= metrics.attention_concentration <= 1.0
        assert metrics.layer_idx == 0
        assert metrics.step == 5

    def test_compute_reference_affinity(self):
        """Test reference affinity computation."""
        # Create attention weights with known pattern
        attn_weights = torch.zeros(self.batch, self.heads, self.seq_len, self.seq_len)
        # High attention from target video (20:30) to ref video (10:15)
        attn_weights[:, :, 20:30, 10:15] = 0.8
        attn_weights[:, :, 20:30, 15:20] = 0.2
        # Normalize
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        affinity = compute_reference_affinity(
            attn_weights, self.reference_indices["video"], self.reference_indices["ref_video"]
        )

        assert 0.0 <= affinity <= 1.0
        assert affinity > 0.5  # Should detect high affinity

    def test_compute_prompt_affinity(self):
        """Test prompt affinity computation."""
        attn_weights = torch.zeros(self.batch, self.heads, self.seq_len, self.seq_len)
        # High attention from video to text
        attn_weights[:, :, 10:50, 0:10] = 0.7
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        affinity = compute_prompt_affinity(
            attn_weights, self.reference_indices["video"], self.reference_indices["text"]
        )

        assert 0.0 <= affinity <= 1.0
        assert affinity > 0.5

    def test_compute_temporal_affinity(self):
        """Test temporal affinity computation."""
        attn_weights = torch.zeros(self.batch, self.heads, self.seq_len, self.seq_len)
        # High attention between adjacent frames
        video_idx = self.reference_indices["video"]
        for i in range(len(video_idx) - 1):
            attn_weights[
                :, :, video_idx[i] : video_idx[i] + 1, video_idx[i + 1] : video_idx[i + 1] + 1
            ] = 0.6
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        affinity = compute_temporal_affinity(attn_weights, video_idx, window=1)

        assert 0.0 <= affinity <= 1.0
        assert affinity > 0.3

    def test_compute_subject_separation(self):
        """Test subject separation metric."""
        attn_weights = torch.zeros(self.batch, self.heads, self.seq_len, self.seq_len)
        subject_a = torch.tensor([10, 11, 12])
        subject_b = torch.tensor([20, 21, 22])

        # Low cross-attention = good separation
        attn_weights[:, :, subject_a.unsqueeze(1), subject_b.unsqueeze(0)] = 0.1
        attn_weights[:, :, subject_b.unsqueeze(1), subject_a.unsqueeze(0)] = 0.1
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        separation = compute_subject_separation(attn_weights, subject_a, subject_b)

        assert 0.0 <= separation <= 1.0
        assert separation < 0.2  # Should be low

    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        metrics_list = [
            AttentionMetrics(reference_affinity=0.5, prompt_affinity=0.4, layer_idx=i, step=i)
            for i in range(5)
        ]

        agg = aggregate_metrics(metrics_list)

        assert "mean_reference_affinity" in agg
        assert "std_reference_affinity" in agg
        assert agg["mean_reference_affinity"] == 0.5
        assert agg["max_reference_affinity"] == 0.5
        assert agg["min_reference_affinity"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
