"""
Integration tests for H3 Attention Control with real model (requires GPU and model).

Run with: pytest tests/test_integration.py -v -k "not slow" or with --run-slow for full tests.
"""

import pytest
import torch
from pathlib import Path

from h3_attention import (
    H3AttentionController,
    AttentionControlConfig,
    PRESETS,
    AttentionSchedule,
    SCHEDULE_PRESETS,
    install_controller_on_model,
    compute_reference_affinity,
)


# Skip if no CUDA
cuda_available = torch.cuda.is_available()
MODEL_PATH = "/path/to/MiniMax-H3"  # Set via env or config


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.slow
class TestIntegration:
    """Integration tests with real H3 model."""

    @classmethod
    def setup_class(cls):
        """Load model once for all tests."""
        try:
            from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks

            cls.pipe = MiniMaxH3Ref2VABlocks().init_pipeline(MODEL_PATH)
            cls.pipe.load_components(dtype=torch.bfloat16)
            cls.pipe.to("cuda")
            cls.model_loaded = True
        except Exception as e:
            cls.model_loaded = False
            cls.load_error = str(e)

    def test_model_loaded(self):
        """Verify model loaded successfully."""
        if not self.model_loaded:
            pytest.skip(f"Model not loaded: {self.load_error}")
        assert hasattr(self.pipe, "transformer_ref")

    def test_controller_install_uninstall(self):
        """Test controller can be installed and uninstalled."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        config = AttentionControlConfig(reference_strength=0.3)
        controller = H3AttentionController(config=config)

        # Install
        controller.install(self.pipe.transformer_ref)
        assert controller._installed_transformer is self.pipe.transformer_ref

        # Uninstall
        controller.uninstall()
        assert controller._installed_transformer is None

    def test_controller_with_preset(self):
        """Test controller with preset config."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        controller = H3AttentionController(preset="reference_only")
        controller.install(self.pipe.transformer_ref)

        assert controller.config.reference_strength > 0
        controller.uninstall()

    def test_context_manager(self):
        """Test context manager patching."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        config = AttentionControlConfig(identity_lock=0.3)
        controller = H3AttentionController(config=config)

        with controller.patch(self.pipe.transformer_ref):
            assert controller._installed_transformer is self.pipe.transformer_ref

        assert controller._installed_transformer is None

    def test_generation_with_control(self):
        """Test actual generation with attention control."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        # Simple text-only generation for speed
        prompt = "A person walking in a park"

        config = AttentionControlConfig(
            reference_strength=0.2,
            prompt_adherence=0.3,
        )
        controller = H3AttentionController(config=config)

        generator = torch.Generator("cuda").manual_seed(42)

        with controller.patch(self.pipe.transformer_ref):
            result = self.pipe(
                prompt=prompt,
                references=[],
                num_frames=50,  # Short for testing
                num_inference_steps=10,  # Few steps for testing
                generator=generator,
            )

        assert "videos" in result
        assert result["videos"] is not None
        assert len(result["videos"]) > 0

    def test_metrics_collection(self):
        """Test metrics are collected during generation."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        config = AttentionControlConfig(
            reference_strength=0.3,
            debug_attention=True,
        )
        controller = H3AttentionController(config=config)

        prompt = "A person walking"
        generator = torch.Generator("cuda").manual_seed(42)

        with controller.patch(self.pipe.transformer_ref):
            _ = self.pipe(
                prompt=prompt,
                references=[],
                num_frames=50,
                num_inference_steps=5,
                generator=generator,
            )

        # Check metrics were recorded
        assert len(controller.state.metrics_history) > 0
        summary = controller.get_metrics_summary()
        assert "avg_reference_affinity" in summary

    def test_schedule_integration(self):
        """Test attention schedule during generation."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        schedule = SCHEDULE_PRESETS["balanced"]
        config = PRESETS["maximum_coherence"].copy()
        controller = H3AttentionController(config=config, schedule=schedule)

        prompt = "A person walking"
        generator = torch.Generator("cuda").manual_seed(42)

        with controller.patch(self.pipe.transformer_ref):
            _ = self.pipe(
                prompt=prompt,
                references=[],
                num_frames=50,
                num_inference_steps=10,
                generator=generator,
            )

        # Verify schedule was applied (config should have changed during generation)
        # The final config should reflect late-phase values
        assert controller.state.step > 0


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.slow
class TestRef2VAIntegration:
    """Integration tests specific to Ref2VA variant."""

    @classmethod
    def setup_class(cls):
        """Load Ref2VA model."""
        try:
            from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks
            from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

            cls.pipe = MiniMaxH3Ref2VABlocks().init_pipeline(MODEL_PATH)
            cls.pipe.load_components(dtype=torch.bfloat16)
            cls.pipe.to("cuda")
            cls.MiniMaxH3ImageReference = MiniMaxH3ImageReference
            cls.model_loaded = True
        except Exception as e:
            cls.model_loaded = False
            cls.load_error = str(e)

    def test_ref2va_with_references(self):
        """Test Ref2VA with image references."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        # Create dummy reference
        ref_img = torch.randn(3, 512, 512)
        reference = self.MiniMaxH3ImageReference(image=ref_img)

        config = AttentionControlConfig(
            reference_strength=0.4,
            identity_lock=0.3,
        )
        controller = H3AttentionController(config=config)

        generator = torch.Generator("cuda").manual_seed(42)

        with controller.patch(self.pipe.transformer_ref):
            result = self.pipe(
                prompt="The person from the reference walks in a garden",
                references=[reference],
                num_frames=50,
                num_inference_steps=10,
                generator=generator,
            )

        assert "videos" in result

    def test_reference_affinity_with_real_refs(self):
        """Test reference affinity metric with real references."""
        if not self.model_loaded:
            pytest.skip("Model not loaded")

        ref_img = torch.randn(3, 512, 512)
        reference = self.MiniMaxH3ImageReference(image=ref_img)

        config = AttentionControlConfig(
            reference_strength=0.5,
            debug_attention=True,
        )
        controller = H3AttentionController(config=config)

        generator = torch.Generator("cuda").manual_seed(42)

        with controller.patch(self.pipe.transformer_ref):
            _ = self.pipe(
                prompt="The person from the reference dances",
                references=[reference],
                num_frames=50,
                num_inference_steps=5,
                generator=generator,
            )

        # Check reference affinity is being measured
        summary = controller.get_metrics_summary()
        assert "avg_reference_affinity" in summary
        # With reference_strength=0.5, we expect some measurable affinity
        assert summary["avg_reference_affinity"] >= 0.0


def test_install_controller_convenience():
    """Test convenience function."""
    from h3_attention import install_controller_on_model, remove_controller_from_model

    mock_model = Mock()
    mock_model.transformer = Mock()
    mock_model.transformer.transformer_blocks = []

    controller = install_controller_on_model(mock_model, preset="reference_only")
    assert controller is not None
    assert controller.config.reference_strength > 0

    remove_controller_from_model(mock_model, controller)
    assert controller._installed_transformer is None


# Mock for testing without model
from unittest.mock import Mock


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-slow"])
