"""
Unit tests for H3 Attention Processor.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from h3_attention import (
    H3AttentionController,
    AttentionControlConfig,
    H3ControlledAttnProcessor,
    H3AttnProcessorWrapper,
)


class MockAttention:
    """Mock attention module for testing."""

    def __init__(self, heads=8, dim_head=64, fused=False):
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.fused_projections = fused
        self.use_bias = False

        if fused:
            self.to_qkv = Mock(side_effect=lambda x: torch.randn(*x.shape[:-1], self.inner_dim * 3))
        else:
            self.to_q = Mock(side_effect=lambda x: torch.randn(*x.shape[:-1], self.inner_dim))
            self.to_k = Mock(side_effect=lambda x: torch.randn(*x.shape[:-1], self.inner_dim))
            self.to_v = Mock(side_effect=lambda x: torch.randn(*x.shape[:-1], self.inner_dim))

        self.norm_q = Mock(side_effect=lambda x: x)
        self.norm_k = Mock(side_effect=lambda x: x)
        self.to_out = [
            Mock(side_effect=lambda x: x),
            Mock(side_effect=lambda x: x),
        ]


class TestH3ControlledAttnProcessor:
    """Tests for H3ControlledAttnProcessor."""

    def test_processor_creation(self):
        """Test processor can be created."""
        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)
        processor = H3ControlledAttnProcessor(controller, layer_idx=0)

        assert processor.controller is controller
        assert processor.layer_idx == 0

    def test_forward_no_control(self):
        """Test forward pass with no active control."""
        config = AttentionControlConfig()  # All zeros
        controller = H3AttentionController(config=config)
        processor = H3ControlledAttnProcessor(controller, layer_idx=0)

        attn = MockAttention()
        hidden_states = torch.randn(1, 100, 512)

        with patch("h3_attention.processor.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(1, 100, 8, 64)

            output = processor(attn, hidden_states)

            assert output.shape == (1, 100, 512)
            mock_dispatch.assert_called_once()

    def test_forward_with_control(self):
        """Test forward pass with active control."""
        config = AttentionControlConfig(reference_strength=0.3, debug_attention=True)
        controller = H3AttentionController(config=config)
        processor = H3ControlledAttnProcessor(controller, layer_idx=0)

        # Set up controller state with required indices
        controller.state.reference_indices = {
            "token_tags": torch.tensor(
                [1] * 10 + [0] * 50 + [2] * 20 + [0] * 20
            ),  # text, video, audio, video
            "position_ids": torch.randint(0, 10, (100, 3)),
            "video": torch.tensor(list(range(10, 60)) + list(range(80, 100))),
            "audio": torch.tensor(list(range(60, 80))),
            "text": torch.tensor(list(range(10))),
            "ref_video": torch.tensor(list(range(10, 20))),
            "ref_audio": torch.tensor([]),
        }

        attn = MockAttention()
        hidden_states = torch.randn(1, 100, 512)
        rotary_emb = (torch.randn(100, 64), torch.randn(100, 64))

        with patch("h3_attention.processor.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(1, 100, 8, 64)

            output = processor(attn, hidden_states, rotary_emb=rotary_emb)

            assert output.shape == (1, 100, 512)
            # With active control the processor takes the manual bias path
            # (dispatch cannot apply an additive bias), so dispatch is unused.
            mock_dispatch.assert_not_called()
            # debug_attention=True records one metrics entry per forward
            assert len(controller.state.metrics_history) == 1


class TestH3AttnProcessorWrapper:
    """Tests for H3AttnProcessorWrapper."""

    def test_wrapper_creation(self):
        """Test wrapper can be created."""
        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)

        base_processor = Mock()
        base_processor.__call__ = Mock(return_value=torch.randn(1, 100, 512))

        wrapper = H3AttnProcessorWrapper(base_processor, controller, layer_idx=0)

        assert wrapper.base_processor is base_processor
        assert wrapper.controller is controller

    def test_wrapper_standard_forward(self):
        """Test wrapper with standard processor."""
        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)

        base_processor = Mock()
        base_processor.__call__ = Mock(return_value=torch.randn(1, 100, 512))
        base_processor._sparse = False

        wrapper = H3AttnProcessorWrapper(base_processor, controller, layer_idx=0)

        attn = MockAttention()
        hidden_states = torch.randn(1, 100, 512)

        # Set up controller state
        controller.state.reference_indices = {
            "token_tags": torch.tensor([1] * 10 + [0] * 90),
            "position_ids": torch.randint(0, 10, (100, 3)),
        }

        with patch("h3_attention.processor.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(1, 100, 8, 64)

            output = wrapper(attn, hidden_states)

            assert output.shape == (1, 100, 512)


class TestH3AttentionController:
    """Tests for H3AttentionController."""

    def test_controller_creation(self):
        """Test controller creation with preset."""
        controller = H3AttentionController(preset="maximum_coherence")

        assert controller.config.reference_strength > 0
        assert controller.config.identity_lock > 0
        assert controller.config.is_active

    def test_controller_creation_with_config(self):
        """Test controller creation with custom config."""
        config = AttentionControlConfig(reference_strength=0.5, identity_lock=0.3)
        controller = H3AttentionController(config=config)

        assert controller.config.reference_strength == 0.5
        assert controller.config.identity_lock == 0.3

    def test_install_uninstall(self):
        """Test controller install/uninstall on mock transformer."""
        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)

        # Create mock transformer with blocks
        mock_transformer = Mock()
        mock_blocks = []
        for i in range(4):
            block = Mock()
            block.attn = Mock()
            block.attn.processor = Mock()
            block.attn.set_processor = Mock()
            mock_blocks.append(block)

        mock_transformer.transformer_blocks = mock_blocks

        controller.install(mock_transformer)

        # Check all blocks got new processors
        for block in mock_blocks:
            block.attn.set_processor.assert_called_once()

        assert controller._installed_transformer is mock_transformer
        assert len(controller._original_processors) == 4

        # Uninstall
        controller.uninstall()

        for block in mock_blocks:
            assert block.attn.set_processor.call_count == 2  # install + uninstall

        assert controller._installed_transformer is None

    def test_context_manager(self):
        """Test context manager patch/unpatch."""
        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)

        mock_transformer = Mock()
        mock_blocks = []
        for i in range(2):
            block = Mock()
            block.attn = Mock()
            block.attn.processor = Mock()
            block.attn.set_processor = Mock()
            mock_blocks.append(block)
        mock_transformer.transformer_blocks = mock_blocks

        with controller.patch(mock_transformer):
            assert controller._installed_transformer is mock_transformer

        assert controller._installed_transformer is None

    def test_update_step(self):
        """Test step update."""
        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)

        controller.update_step(step=5, total_steps=30, timestep=0.5)

        assert controller.state.step == 5
        assert controller.state.total_steps == 30
        assert controller.state.current_timestep == 0.5

    def test_set_reference_indices(self):
        """Test setting reference indices."""
        controller = H3AttentionController()

        controller.set_reference_indices(
            video_indices=torch.tensor([10, 11, 12]),
            audio_indices=torch.tensor([20, 21]),
            text_indices=torch.tensor([0, 1, 2]),
            ref_video_indices=torch.tensor([5, 6]),
            ref_audio_indices=torch.tensor([15]),
        )

        assert "video" in controller.state.reference_indices
        assert "ref_video" in controller.state.reference_indices

    def test_compute_attention_bias_disabled(self):
        """Test bias computation when disabled."""
        config = AttentionControlConfig()
        controller = H3AttentionController(config=config)

        query = torch.randn(1, 8, 100, 64)
        key = torch.randn(1, 8, 100, 64)
        token_tags = torch.randint(0, 3, (100,))
        position_ids = torch.randint(0, 10, (100, 3))

        bias = controller.compute_attention_bias(query, key, token_tags, position_ids, 0)

        # When disabled the bias is a broadcastable zero (avoids materializing
        # a full [B, H, S, S] tensor, which is gigabytes at long sequence lengths).
        expanded = bias.expand(1, 8, 100, 100)
        assert expanded.shape == (1, 8, 100, 100)
        assert bias.abs().max() == 0.0  # No bias when disabled

    def test_compute_attention_bias_enabled(self):
        """Test bias computation with active control."""
        config = AttentionControlConfig(reference_strength=0.5)
        controller = H3AttentionController(config=config)

        query = torch.randn(1, 8, 100, 64)
        key = torch.randn(1, 8, 100, 64)
        token_tags = torch.tensor([1] * 10 + [0] * 90)  # 10 text, 90 video
        position_ids = torch.randint(0, 10, (100, 3))

        controller.state.reference_indices = {
            "video": torch.tensor(list(range(10, 100))),
            "text": torch.tensor(list(range(10))),
            "ref_video": torch.tensor([5, 6, 7]),
        }

        bias = controller.compute_attention_bias(query, key, token_tags, position_ids, 0)

        assert bias.shape == (1, 8, 100, 100)
        assert bias.abs().max() > 0.0  # Should have bias

    def test_record_metrics(self):
        """Test metrics recording."""
        from h3_attention import AttentionMetrics

        config = AttentionControlConfig(reference_strength=0.3, debug_attention=True)
        controller = H3AttentionController(config=config)

        metrics = AttentionMetrics(
            reference_affinity=0.5,
            prompt_affinity=0.4,
            temporal_affinity=0.3,
            layer_idx=0,
            step=10,
            timestep=0.5,
        )

        controller.record_metrics(metrics)

        assert len(controller.state.metrics_history) == 1
        assert controller.state.metrics_history[0].reference_affinity == 0.5

    def test_get_metrics_summary(self):
        """Test metrics summary."""
        from h3_attention import AttentionMetrics

        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)

        for i in range(5):
            metrics = AttentionMetrics(
                reference_affinity=0.5 + i * 0.02,
                prompt_affinity=0.4 + i * 0.01,
                temporal_affinity=0.3 + i * 0.03,
                layer_idx=0,
                step=i,
                timestep=0.5,
            )
            controller.record_metrics(metrics)

        summary = controller.get_metrics_summary()

        assert summary["steps_recorded"] == 5
        assert "avg_reference_affinity" in summary
        assert "max_reference_affinity" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
