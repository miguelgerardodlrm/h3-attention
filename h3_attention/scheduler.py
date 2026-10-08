"""
Attention Schedule for timestep-dependent control.

Allows configuring different control strengths at different denoising phases.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import torch


class SchedulePhase(str, Enum):
    """Denoising phases."""

    EARLY = "early"  # High noise, structure formation
    MIDDLE = "middle"  # Mid noise, detail refinement
    LATE = "late"  # Low noise, final polish


@dataclass
class PhaseConfig:
    """Configuration for a single schedule phase."""

    reference_strength: float = 0.0
    identity_lock: float = 0.0
    prompt_adherence: float = 0.0
    temporal_lock: float = 0.0
    subject_separation: float = 0.0
    audio_strength: float = 0.0

    # Phase boundaries (0.0 to 1.0, where 0 = start, 1 = end)
    start_ratio: float = 0.0
    end_ratio: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reference_strength": self.reference_strength,
            "identity_lock": self.identity_lock,
            "prompt_adherence": self.prompt_adherence,
            "temporal_lock": self.temporal_lock,
            "subject_separation": self.subject_separation,
            "audio_strength": self.audio_strength,
        }


# Default phase configurations
DEFAULT_PHASES = {
    SchedulePhase.EARLY: PhaseConfig(
        reference_strength=0.5,
        identity_lock=0.4,
        prompt_adherence=0.3,
        temporal_lock=0.1,
        subject_separation=0.2,
        audio_strength=0.2,
        start_ratio=0.0,
        end_ratio=0.33,
    ),
    SchedulePhase.MIDDLE: PhaseConfig(
        reference_strength=0.3,
        identity_lock=0.5,
        prompt_adherence=0.2,
        temporal_lock=0.4,
        subject_separation=0.3,
        audio_strength=0.3,
        start_ratio=0.33,
        end_ratio=0.66,
    ),
    SchedulePhase.LATE: PhaseConfig(
        reference_strength=0.2,
        identity_lock=0.3,
        prompt_adherence=0.1,
        temporal_lock=0.5,
        subject_separation=0.2,
        audio_strength=0.4,
        start_ratio=0.66,
        end_ratio=1.0,
    ),
}


@dataclass
class AttentionSchedule:
    """
    Schedule for attention control parameters across denoising steps.

    Maps step ratios to parameter configurations, enabling policies like:
    - Early: strong reference + strong prompt
    - Middle: strong identity + strong temporal
    - Late: moderate reference + strong temporal/video consistency
    """

    phases: Dict[SchedulePhase, PhaseConfig] = field(default_factory=dict)
    custom_phases: List[PhaseConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.phases:
            self.phases = DEFAULT_PHASES.copy()

    def get_config_at_step(self, step: int, total_steps: int) -> Dict[str, float]:
        """Get interpolated config for current step."""
        if total_steps <= 0:
            return {}

        ratio = step / total_steps

        # Find active phase
        active_phase = None
        for phase, config in self.phases.items():
            if config.start_ratio <= ratio < config.end_ratio:
                active_phase = config
                break

        # Handle edge case at end
        if active_phase is None:
            for phase, config in self.phases.items():
                if ratio >= config.end_ratio - 1e-6:
                    active_phase = config

        if active_phase is None:
            return {}

        # Interpolate within phase
        phase_progress = (ratio - active_phase.start_ratio) / max(
            1e-6, active_phase.end_ratio - active_phase.start_ratio
        )
        phase_progress = max(0.0, min(1.0, phase_progress))

        return active_phase.to_dict()

    def get_config_at_ratio(self, ratio: float) -> Dict[str, float]:
        """Get config for a specific ratio (0.0 to 1.0)."""
        ratio = max(0.0, min(1.0, ratio))

        for phase, config in self.phases.items():
            if config.start_ratio <= ratio < config.end_ratio:
                return config.to_dict()

        # Last phase
        for phase, config in reversed(self.phases.items()):
            if ratio >= config.start_ratio:
                return config.to_dict()

        return {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttentionSchedule":
        """Create schedule from dictionary."""
        phases = {}
        for phase_name, config_data in data.get("phases", {}).items():
            phase = SchedulePhase(phase_name)
            phases[phase] = PhaseConfig(**config_data)

        custom = [PhaseConfig(**c) for c in data.get("custom_phases", [])]

        return cls(phases=phases, custom_phases=custom)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phases": {k.value: v.to_dict() for k, v in self.phases.items()},
            "custom_phases": [c.to_dict() for c in self.custom_phases],
        }

    @classmethod
    def linear_decay(cls, start: Dict[str, float], end: Dict[str, float]) -> "AttentionSchedule":
        """Create schedule with linear interpolation between start and end."""
        phases = {}
        for key in start:
            phases[SchedulePhase.EARLY] = PhaseConfig(**start)
            phases[SchedulePhase.LATE] = PhaseConfig(**end)
            break
        return cls(phases=phases)

    @classmethod
    def three_phase(
        cls, early: Dict[str, float], middle: Dict[str, float], late: Dict[str, float]
    ) -> "AttentionSchedule":
        """Create standard three-phase schedule."""
        phases = {
            SchedulePhase.EARLY: PhaseConfig(**early),
            SchedulePhase.MIDDLE: PhaseConfig(**middle),
            SchedulePhase.LATE: PhaseConfig(**late),
        }
        return cls(phases=phases)


# Preset schedules
SCHEDULE_PRESETS = {
    "disabled": AttentionSchedule(phases={}),
    "reference_heavy_early": AttentionSchedule.three_phase(
        early={"reference_strength": 0.5, "prompt_adherence": 0.4, "identity_lock": 0.3},
        middle={"reference_strength": 0.3, "identity_lock": 0.5, "temporal_lock": 0.4},
        late={"reference_strength": 0.15, "identity_lock": 0.3, "temporal_lock": 0.5},
    ),
    "identity_focused": AttentionSchedule.three_phase(
        early={"identity_lock": 0.5, "reference_strength": 0.3},
        middle={"identity_lock": 0.6, "temporal_lock": 0.3, "subject_separation": 0.3},
        late={"identity_lock": 0.4, "temporal_lock": 0.4},
    ),
    "temporal_coherence": AttentionSchedule.three_phase(
        early={"reference_strength": 0.2, "prompt_adherence": 0.3},
        middle={"temporal_lock": 0.4, "identity_lock": 0.3, "audio_strength": 0.2},
        late={"temporal_lock": 0.6, "audio_strength": 0.4},
    ),
    "prompt_adherence": AttentionSchedule.three_phase(
        early={"prompt_adherence": 0.5, "reference_strength": 0.3},
        middle={"prompt_adherence": 0.3, "identity_lock": 0.3, "temporal_lock": 0.2},
        late={"prompt_adherence": 0.2, "temporal_lock": 0.4},
    ),
    "balanced": AttentionSchedule.three_phase(
        early={"reference_strength": 0.3, "identity_lock": 0.3, "prompt_adherence": 0.2},
        middle={
            "reference_strength": 0.25,
            "identity_lock": 0.4,
            "temporal_lock": 0.3,
            "subject_separation": 0.2,
        },
        late={
            "reference_strength": 0.15,
            "identity_lock": 0.25,
            "temporal_lock": 0.4,
            "audio_strength": 0.2,
        },
    ),
}
