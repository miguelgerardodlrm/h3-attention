"""
Unit tests for H3 Attention Control configuration.
"""

import pytest
import torch

from h3_attention import (
    AttentionControlConfig,
    PRESETS,
    AttentionSchedule,
    SchedulePhase,
    PhaseConfig,
    SCHEDULE_PRESETS,
)


class TestAttentionControlConfig:
    """Tests for AttentionControlConfig."""

    def test_default_config(self):
        """Test default config has all zeros."""
        config = AttentionControlConfig()
        assert config.reference_strength == 0.0
        assert config.identity_lock == 0.0
        assert config.prompt_adherence == 0.0
        assert config.temporal_lock == 0.0
        assert config.subject_separation == 0.0
        assert config.audio_strength == 0.0
        assert config.adaptive_mode is False
        assert config.debug_attention is False
        assert config.is_active is False

    def test_preset_configs(self):
        """Test all presets are valid."""
        for name, preset in PRESETS.items():
            assert isinstance(preset, AttentionControlConfig)
            # All strengths should be in [0, 1]
            assert 0.0 <= preset.reference_strength <= 1.0
            assert 0.0 <= preset.identity_lock <= 1.0
            assert 0.0 <= preset.prompt_adherence <= 1.0
            assert 0.0 <= preset.temporal_lock <= 1.0
            assert 0.0 <= preset.subject_separation <= 1.0
            assert 0.0 <= preset.audio_strength <= 1.0

    def test_config_validation(self):
        """Test config validation clamps values."""
        # Should raise for out-of-range values
        with pytest.raises(ValueError):
            AttentionControlConfig(reference_strength=1.5)

        with pytest.raises(ValueError):
            AttentionControlConfig(identity_lock=-0.1)

        with pytest.raises(ValueError):
            AttentionControlConfig(reference_boost_mode="invalid")

        with pytest.raises(ValueError):
            AttentionControlConfig(identity_spatial_scope="invalid")

        with pytest.raises(ValueError):
            AttentionControlConfig(temporal_mode="invalid")

    def test_config_copy(self):
        """Test config copying."""
        config = PRESETS["maximum_coherence"].copy()
        config.reference_strength = 0.9
        assert config.reference_strength == 0.9
        assert PRESETS["maximum_coherence"].reference_strength != 0.9

    def test_config_with_overrides(self):
        """Test creating new config with overrides."""
        base = PRESETS["reference_only"]
        modified = base.with_overrides(reference_strength=0.5, identity_lock=0.3)

        assert modified.reference_strength == 0.5
        assert modified.identity_lock == 0.3
        assert modified.prompt_adherence == base.prompt_adherence

    def test_active_mechanisms(self):
        """Test active_mechanisms property."""
        config = AttentionControlConfig()
        assert config.active_mechanisms == []

        config.reference_strength = 0.3
        config.identity_lock = 0.4
        assert "reference" in config.active_mechanisms
        assert "identity" in config.active_mechanisms
        assert len(config.active_mechanisms) == 2

    def test_to_from_dict(self):
        """Test serialization round-trip."""
        config = PRESETS["maximum_coherence"]
        data = config.to_dict()
        restored = AttentionControlConfig.from_dict(data)

        assert restored.reference_strength == config.reference_strength
        assert restored.identity_lock == config.identity_lock
        assert restored.adaptive_mode == config.adaptive_mode


class TestAttentionSchedule:
    """Tests for AttentionSchedule."""

    def test_default_schedule(self):
        """Test default three-phase schedule."""
        schedule = AttentionSchedule()
        assert SchedulePhase.EARLY in schedule.phases
        assert SchedulePhase.MIDDLE in schedule.phases
        assert SchedulePhase.LATE in schedule.phases

    def test_get_config_at_step(self):
        """Test config retrieval at specific steps."""
        schedule = SCHEDULE_PRESETS["balanced"]

        # Early phase (step 0 of 30)
        early_config = schedule.get_config_at_step(0, 30)
        assert early_config["reference_strength"] > 0

        # Middle phase (step 15 of 30)
        mid_config = schedule.get_config_at_step(15, 30)
        assert mid_config["temporal_lock"] > 0

        # Late phase (step 25 of 30)
        late_config = schedule.get_config_at_step(25, 30)
        assert late_config["temporal_lock"] > 0

    def test_get_config_at_ratio(self):
        """Test config retrieval at ratio."""
        schedule = SCHEDULE_PRESETS["balanced"]

        early = schedule.get_config_at_ratio(0.1)
        assert early["reference_strength"] > 0

        late = schedule.get_config_at_ratio(0.9)
        assert late["temporal_lock"] > 0

    def test_schedule_presets(self):
        """Test all schedule presets are valid."""
        for name, schedule in SCHEDULE_PRESETS.items():
            assert isinstance(schedule, AttentionSchedule)
            for phase, config in schedule.phases.items():
                assert isinstance(config, PhaseConfig)
                assert 0.0 <= config.start_ratio <= 1.0
                assert 0.0 <= config.end_ratio <= 1.0
                assert config.start_ratio < config.end_ratio

    def test_three_phase_factory(self):
        """Test three_phase factory method."""
        schedule = AttentionSchedule.three_phase(
            early={"reference_strength": 0.5},
            middle={"identity_lock": 0.5},
            late={"temporal_lock": 0.5},
        )

        assert schedule.phases[SchedulePhase.EARLY].reference_strength == 0.5
        assert schedule.phases[SchedulePhase.MIDDLE].identity_lock == 0.5
        assert schedule.phases[SchedulePhase.LATE].temporal_lock == 0.5

    def test_linear_decay_factory(self):
        """Test linear_decay factory method."""
        schedule = AttentionSchedule.linear_decay(
            start={"reference_strength": 0.5},
            end={"reference_strength": 0.1},
        )

        # Only early and late phases should exist
        assert SchedulePhase.EARLY in schedule.phases
        assert SchedulePhase.LATE in schedule.phases
        assert schedule.phases[SchedulePhase.EARLY].reference_strength == 0.5
        assert schedule.phases[SchedulePhase.LATE].reference_strength == 0.1

    def test_serialization(self):
        """Test schedule serialization."""
        schedule = SCHEDULE_PRESETS["balanced"]
        data = schedule.to_dict()
        restored = AttentionSchedule.from_dict(data)

        assert (
            restored.phases[SchedulePhase.EARLY].reference_strength
            == schedule.phases[SchedulePhase.EARLY].reference_strength
        )


class TestPhaseConfig:
    """Tests for PhaseConfig."""

    def test_to_dict(self):
        """Test phase config serialization."""
        phase = PhaseConfig(
            reference_strength=0.5,
            identity_lock=0.3,
            start_ratio=0.0,
            end_ratio=0.33,
        )

        data = phase.to_dict()
        assert data["reference_strength"] == 0.5
        assert data["identity_lock"] == 0.3
        assert "start_ratio" not in data  # Not included in to_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
