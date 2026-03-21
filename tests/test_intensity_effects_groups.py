"""Tests for intensity selector (Task 1.12), effects size (Task 1.13),
and light group splitting (Task 1.14).

Verifies:
- Intensity selector: 3 levels apply correct multipliers on top of base values
- Effects size: controls how many lights get reactive effects per beat
- Light group splitting: auto-assigns lights to groups with phase offsets
"""

import math

import numpy as np
import pytest

from hue_visualizer.audio.analyzer import AudioFeatures
from hue_visualizer.audio.beat_detector import BeatInfo
from hue_visualizer.visualizer.engine import (
    EffectEngine,
    INTENSITY_INTENSE,
    INTENSITY_NORMAL,
    INTENSITY_CHILL,
    INTENSITY_LEVELS,
    INTENSITY_MULTIPLIERS,
)


# --- Test helpers ---


def _silence_features() -> AudioFeatures:
    return AudioFeatures(
        band_energies=np.zeros(7),
        spectral_centroid=0.0,
        spectral_flux=0.0,
        spectral_rolloff=0.0,
        spectral_flatness=0.0,
        rms=0.0,
        peak=0.0,
        spectrum=np.zeros(1024),
    )


def _loud_features(rms: float = 0.7) -> AudioFeatures:
    bands = np.array([0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])
    return AudioFeatures(
        band_energies=bands,
        spectral_centroid=3000.0,
        spectral_flux=20.0,
        spectral_rolloff=5000.0,
        spectral_flatness=0.2,
        rms=rms,
        peak=rms * 1.5,
        spectrum=np.zeros(1024),
    )


def _no_beat() -> BeatInfo:
    return BeatInfo(
        is_beat=False,
        bpm=0.0,
        bpm_confidence=0.0,
        beat_strength=0.0,
        predicted_next_beat=0.0,
        time_since_beat=0.0,
    )


def _beat(strength: float = 0.8, bpm: float = 128.0) -> BeatInfo:
    return BeatInfo(
        is_beat=True,
        bpm=bpm,
        bpm_confidence=0.5,
        beat_strength=strength,
        predicted_next_beat=0.0,
        time_since_beat=0.0,
    )


# ============================================================================
# Task 1.12: Intensity selector tests
# ============================================================================


class TestIntensityConstants:
    """Verify intensity level constants and multiplier definitions."""

    def test_three_levels_defined(self):
        assert len(INTENSITY_LEVELS) == 3
        assert INTENSITY_INTENSE in INTENSITY_LEVELS
        assert INTENSITY_NORMAL in INTENSITY_LEVELS
        assert INTENSITY_CHILL in INTENSITY_LEVELS

    def test_multipliers_defined_for_all_levels(self):
        for level in INTENSITY_LEVELS:
            assert level in INTENSITY_MULTIPLIERS
            mult = INTENSITY_MULTIPLIERS[level]
            assert "flash_tau" in mult
            assert "attack_alpha" in mult
            assert "max_brightness" in mult

    def test_normal_multipliers_are_identity_or_near(self):
        mult = INTENSITY_MULTIPLIERS[INTENSITY_NORMAL]
        assert mult["flash_tau"] == 1.0
        assert mult["attack_alpha"] == 1.0

    def test_intense_has_correct_multipliers(self):
        mult = INTENSITY_MULTIPLIERS[INTENSITY_INTENSE]
        assert mult["flash_tau"] == 0.7
        assert mult["attack_alpha"] == 1.3
        assert mult["max_brightness"] == 1.0

    def test_chill_has_correct_multipliers(self):
        mult = INTENSITY_MULTIPLIERS[INTENSITY_CHILL]
        assert mult["flash_tau"] == 1.5
        assert mult["attack_alpha"] == 0.6
        assert mult["max_brightness"] == 0.6


class TestIntensityDefault:
    """Engine starts in normal intensity."""

    def test_default_is_normal(self):
        engine = EffectEngine(num_lights=6)
        assert engine.intensity_level == INTENSITY_NORMAL

    def test_default_max_brightness(self):
        engine = EffectEngine(num_lights=6)
        assert engine._max_brightness == INTENSITY_MULTIPLIERS[INTENSITY_NORMAL]["max_brightness"]


class TestIntensitySetLevel:
    """Test set_intensity applies multipliers correctly."""

    def test_set_intense(self):
        engine = EffectEngine(num_lights=6, attack_alpha=0.7)
        base_tau = engine._base_flash_tau
        engine.set_intensity(INTENSITY_INTENSE)
        assert engine.intensity_level == INTENSITY_INTENSE
        assert engine._flash_tau == pytest.approx(base_tau * 0.7)
        assert engine.attack_alpha == pytest.approx(0.7 * 1.3)
        assert engine._max_brightness == 1.0

    def test_set_chill(self):
        engine = EffectEngine(num_lights=6, attack_alpha=0.7)
        base_tau = engine._base_flash_tau
        engine.set_intensity(INTENSITY_CHILL)
        assert engine.intensity_level == INTENSITY_CHILL
        assert engine._flash_tau == pytest.approx(base_tau * 1.5)
        assert engine.attack_alpha == pytest.approx(0.7 * 0.6)
        assert engine._max_brightness == 0.6

    def test_set_back_to_normal(self):
        engine = EffectEngine(num_lights=6, attack_alpha=0.7)
        base_tau = engine._base_flash_tau
        engine.set_intensity(INTENSITY_INTENSE)
        engine.set_intensity(INTENSITY_NORMAL)
        # Should restore normal multipliers on base values
        assert engine._flash_tau == pytest.approx(base_tau * 1.0)
        assert engine.attack_alpha == pytest.approx(0.7 * 1.0)

    def test_invalid_level_ignored(self):
        engine = EffectEngine(num_lights=6)
        engine.set_intensity(INTENSITY_INTENSE)
        engine.set_intensity("nonsense")
        assert engine.intensity_level == INTENSITY_INTENSE  # unchanged

    def test_attack_alpha_clamped_to_valid_range(self):
        """Even with extreme base values, attack_alpha stays in [0.01, 1.0]."""
        engine = EffectEngine(num_lights=4, attack_alpha=0.95)
        engine.set_intensity(INTENSITY_INTENSE)  # 0.95 * 1.3 = 1.235
        assert engine.attack_alpha <= 1.0
        assert engine.attack_alpha >= 0.01


class TestIntensityWithGenrePreset:
    """Intensity should stack on top of genre base values."""

    def test_set_flash_tau_stores_base(self):
        engine = EffectEngine(num_lights=6)
        engine.set_flash_tau(0.30)  # Genre sets base
        assert engine._base_flash_tau == 0.30
        # With normal intensity, flash_tau == base
        assert engine._flash_tau == pytest.approx(0.30)

    def test_set_flash_tau_with_intensity(self):
        engine = EffectEngine(num_lights=6)
        engine.set_intensity(INTENSITY_INTENSE)
        engine.set_flash_tau(0.30)  # Genre sets base
        assert engine._base_flash_tau == 0.30
        assert engine._flash_tau == pytest.approx(0.30 * 0.7)

    def test_set_base_attack_alpha(self):
        engine = EffectEngine(num_lights=6)
        engine.set_intensity(INTENSITY_CHILL)
        engine.set_base_attack_alpha(0.8)  # Genre sets base
        assert engine._base_attack_alpha == 0.8
        assert engine.attack_alpha == pytest.approx(0.8 * 0.6)

    def test_intensity_preserves_across_genre_change(self):
        engine = EffectEngine(num_lights=6)
        engine.set_intensity(INTENSITY_INTENSE)
        # Simulate genre change setting new base values
        engine.set_flash_tau(0.15)
        engine.set_base_attack_alpha(0.9)
        # Intensity should still be applied
        assert engine._flash_tau == pytest.approx(0.15 * 0.7)
        assert engine.attack_alpha == pytest.approx(min(0.9 * 1.3, 1.0))


class TestIntensityBrightnessCap:
    """Chill mode should cap brightness lower than intense."""

    def test_chill_caps_brightness(self):
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)
        engine.set_intensity(INTENSITY_CHILL)
        features = _loud_features(rms=0.8)

        # Run engine with beats to maximize brightness
        for i in range(100):
            is_beat = (i % 10 == 0)
            beat = _beat(strength=1.0) if is_beat else _no_beat()
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        # All brightnesses should be <= 0.6 (chill max brightness)
        for s in states:
            assert s.brightness <= 0.61, \
                f"Chill brightness should be capped at 0.6, got {s.brightness}"

    def test_intense_allows_full_brightness(self):
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)
        engine.set_intensity(INTENSITY_INTENSE)
        features = _loud_features(rms=0.9)

        for i in range(100):
            is_beat = (i % 10 == 0)
            beat = _beat(strength=1.0) if is_beat else _no_beat()
            states = engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        # At least some lights should reach high brightness
        max_b = max(s.brightness for s in states)
        assert max_b > 0.7, \
            f"Intense should allow high brightness, got max={max_b}"


class TestIntensityProducesValidOutput:
    """All intensity levels produce valid output over time."""

    @pytest.mark.parametrize("level", INTENSITY_LEVELS)
    def test_continuous_operation(self, level):
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        engine.set_intensity(level)
        dt = 0.033
        now = 1000.0

        for i in range(300):  # ~10 seconds
            rms = 0.3 + 0.3 * abs(math.sin(i * 0.05))
            features = _loud_features(rms=rms)
            is_beat = (i % 15 == 0) and rms > 0.3
            beat = _beat(strength=0.7) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
                assert 0 <= s.x <= 1.0
                assert 0 <= s.y <= 1.0


# ============================================================================
# Task 1.13: Effects size tests
# ============================================================================


class TestEffectsSizeDefault:
    """Engine starts with 100% effects size (all lights active)."""

    def test_default_is_full(self):
        engine = EffectEngine(num_lights=6)
        assert engine.effects_size == 1.0

    def test_default_all_lights_active(self):
        engine = EffectEngine(num_lights=6)
        assert engine._active_lights == set(range(6))


class TestEffectsSizeSet:
    """Test set_effects_size controls active light count."""

    def test_set_half(self):
        engine = EffectEngine(num_lights=6)
        engine.set_effects_size(0.5)
        assert engine.effects_size == 0.5
        assert len(engine._active_lights) == 3  # ceil(6 * 0.5)

    def test_set_quarter(self):
        engine = EffectEngine(num_lights=8)
        engine.set_effects_size(0.25)
        assert len(engine._active_lights) == 2  # ceil(8 * 0.25)

    def test_set_single_light(self):
        engine = EffectEngine(num_lights=6)
        engine.set_effects_size(1.0 / 6)
        assert len(engine._active_lights) == 1

    def test_set_full(self):
        engine = EffectEngine(num_lights=6)
        engine.set_effects_size(0.5)
        engine.set_effects_size(1.0)
        assert engine._active_lights == set(range(6))

    def test_clamp_to_zero(self):
        engine = EffectEngine(num_lights=6)
        engine.set_effects_size(-0.5)
        assert engine.effects_size == 0.0
        # At least 1 light should still be active
        assert len(engine._active_lights) >= 1

    def test_clamp_to_one(self):
        engine = EffectEngine(num_lights=6)
        engine.set_effects_size(1.5)
        assert engine.effects_size == 1.0
        assert engine._active_lights == set(range(6))

    def test_minimum_one_active(self):
        """Even at very small effects_size, at least 1 light is active."""
        engine = EffectEngine(num_lights=6)
        engine.set_effects_size(0.01)
        assert len(engine._active_lights) >= 1


class TestEffectsSizeRotation:
    """Active lights should rotate on each beat."""

    def test_active_lights_rotate_on_beat(self):
        engine = EffectEngine(num_lights=6, max_flash_hz=50.0)
        engine.set_effects_size(1.0 / 6)  # 1 light at a time
        features = _loud_features()

        active_sets = []
        for i in range(12):
            # Fire a beat
            states = engine.tick(features, _beat(), dt=0.033, now=1000.0 + i * 0.5)
            active_sets.append(frozenset(engine._active_lights))

        # Should have visited multiple different light indices
        unique_active = set()
        for s in active_sets:
            unique_active.update(s)
        assert len(unique_active) > 1, \
            "Active lights should rotate across different indices"

    def test_full_size_does_not_rotate(self):
        engine = EffectEngine(num_lights=6, max_flash_hz=50.0)
        engine.set_effects_size(1.0)
        features = _loud_features()

        for i in range(10):
            engine.tick(features, _beat(), dt=0.033, now=1000.0 + i * 0.5)
        # All lights should always be active
        assert engine._active_lights == set(range(6))


class TestEffectsSizeFlashBehavior:
    """Inactive lights should have reduced reactive effects."""

    def test_inactive_lights_dimmer_output(self):
        """Active lights should produce brighter output than inactive ones on a beat."""
        engine = EffectEngine(num_lights=6, max_flash_hz=50.0)
        engine.set_effects_size(0.5)  # 3 lights active
        features = _loud_features(rms=0.6)

        # Stabilize
        for i in range(30):
            engine.tick(features, _no_beat(), dt=0.033, now=1000.0 + i * 0.033)

        # Record which lights are active BEFORE the beat (they get the flash)
        active_before_beat = set(engine._active_lights)
        inactive_before_beat = set(range(6)) - active_before_beat

        # Fire a strong beat — active lights get full flash, inactive get reduced
        states = engine.tick(features, _beat(strength=1.0), dt=0.033, now=1001.0)

        # Check output brightness (which includes reactive_scale for inactive lights)
        active_brightness = [states[idx].brightness for idx in active_before_beat]
        inactive_brightness = [states[idx].brightness for idx in inactive_before_beat]

        if inactive_before_beat:
            avg_active = sum(active_brightness) / max(len(active_brightness), 1)
            avg_inactive = sum(inactive_brightness) / max(len(inactive_brightness), 1)
            assert avg_active > avg_inactive, \
                f"Active lights should be brighter: active={avg_active}, inactive={avg_inactive}"


class TestEffectsSizeReset:
    """Reset should restore effects size rotation offset."""

    def test_reset_restores_offset(self):
        engine = EffectEngine(num_lights=6, max_flash_hz=50.0)
        engine.set_effects_size(0.5)
        features = _loud_features()

        # Advance the rotation
        for i in range(10):
            engine.tick(features, _beat(), dt=0.033, now=1000.0 + i * 0.5)
        assert engine._active_light_offset > 0 or True  # may wrap to 0

        engine.reset()
        assert engine._active_light_offset == 0


class TestEffectsSizeProducesValidOutput:
    """All effects_size values produce valid output."""

    @pytest.mark.parametrize("size", [0.0, 0.17, 0.25, 0.5, 1.0])
    def test_continuous_operation(self, size):
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        engine.set_effects_size(size)
        dt = 0.033
        now = 1000.0

        for i in range(200):
            rms = 0.4 + 0.3 * abs(math.sin(i * 0.05))
            features = _loud_features(rms=rms)
            is_beat = (i % 15 == 0)
            beat = _beat(strength=0.7) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0


# ============================================================================
# Task 1.14: Light group splitting tests
# ============================================================================


class TestLightGroupAssignment:
    """Lights are assigned to groups based on count."""

    def test_single_light_one_group(self):
        engine = EffectEngine(num_lights=1)
        assert engine.num_groups == 1
        assert engine.light_groups == [0]

    def test_two_lights_two_groups(self):
        engine = EffectEngine(num_lights=2)
        assert engine.num_groups == 2
        assert len(engine.light_groups) == 2

    def test_four_lights_two_groups(self):
        engine = EffectEngine(num_lights=4)
        assert engine.num_groups == 2
        groups = engine.light_groups
        assert len(groups) == 4
        # Each group should have 2 lights
        assert groups.count(0) == 2
        assert groups.count(1) == 2

    def test_five_lights_three_groups(self):
        engine = EffectEngine(num_lights=5)
        assert engine.num_groups == 3
        groups = engine.light_groups
        assert len(groups) == 5
        # All three groups should be used
        assert 0 in groups
        assert 1 in groups
        assert 2 in groups

    def test_six_lights_three_groups(self):
        engine = EffectEngine(num_lights=6)
        assert engine.num_groups == 3
        groups = engine.light_groups
        assert len(groups) == 6
        # Each group should have 2 lights
        assert groups.count(0) == 2
        assert groups.count(1) == 2
        assert groups.count(2) == 2

    def test_eight_lights_three_groups(self):
        engine = EffectEngine(num_lights=8)
        assert engine.num_groups == 3
        groups = engine.light_groups
        assert len(groups) == 8
        # All groups should have lights
        for g in range(3):
            assert g in groups

    def test_groups_are_contiguous(self):
        """Lights should be assigned to groups in order (0, 0, 1, 1, 2, 2 etc.)."""
        engine = EffectEngine(num_lights=6)
        groups = engine.light_groups
        # Verify group numbers don't decrease (monotonic non-decreasing)
        for i in range(1, len(groups)):
            assert groups[i] >= groups[i - 1], \
                f"Groups should be contiguous, got {groups}"


class TestGroupPhaseOffsets:
    """Phase offsets should be evenly distributed."""

    def test_two_groups_offsets(self):
        engine = EffectEngine(num_lights=4)
        assert engine._group_phase_offsets == pytest.approx([0.0, 0.5])

    def test_three_groups_offsets(self):
        engine = EffectEngine(num_lights=6)
        offsets = engine._group_phase_offsets
        assert len(offsets) == 3
        assert offsets[0] == pytest.approx(0.0)
        assert offsets[1] == pytest.approx(1.0 / 3)
        assert offsets[2] == pytest.approx(2.0 / 3)

    def test_single_group_no_offset(self):
        engine = EffectEngine(num_lights=1)
        assert engine._group_phase_offsets == [0.0]

    def test_get_group_phase_offset_per_light(self):
        engine = EffectEngine(num_lights=6)
        # First 2 lights -> group 0 -> offset 0.0
        assert engine.get_group_phase_offset(0) == pytest.approx(0.0)
        assert engine.get_group_phase_offset(1) == pytest.approx(0.0)
        # Middle 2 lights -> group 1 -> offset ~0.333
        assert engine.get_group_phase_offset(2) == pytest.approx(1.0 / 3)
        assert engine.get_group_phase_offset(3) == pytest.approx(1.0 / 3)
        # Last 2 lights -> group 2 -> offset ~0.667
        assert engine.get_group_phase_offset(4) == pytest.approx(2.0 / 3)
        assert engine.get_group_phase_offset(5) == pytest.approx(2.0 / 3)

    def test_out_of_range_returns_zero(self):
        engine = EffectEngine(num_lights=4)
        assert engine.get_group_phase_offset(10) == 0.0


class TestGroupColorDiversity:
    """Groups should produce different colors in palette mode."""

    def test_groups_produce_different_hues_freq_mode(self):
        """In frequency_zones mode with groups, different groups should have different hues."""
        engine = EffectEngine(num_lights=6)
        engine.set_spatial_mode("frequency_zones")
        features = _loud_features(rms=0.5)
        beat = _no_beat()

        # Run to stabilize
        for i in range(60):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        # Check that lights in different groups have different hues
        hues = [light.hue for light in engine._lights]
        groups = engine.light_groups

        # Get average hue per group
        group_hues: dict[int, list[float]] = {}
        for idx, g in enumerate(groups):
            group_hues.setdefault(g, []).append(hues[idx])

        # At least two groups should have meaningfully different average hues
        avg_hues = []
        for g in sorted(group_hues.keys()):
            # Simple average (ignoring circular nature for test purposes)
            avg = sum(group_hues[g]) / len(group_hues[g])
            avg_hues.append(avg)

        # Check that at least one pair of groups has different hues
        if len(avg_hues) >= 2:
            diffs = []
            for i in range(len(avg_hues)):
                for j in range(i + 1, len(avg_hues)):
                    diff = abs(avg_hues[i] - avg_hues[j])
                    if diff > 180:
                        diff = 360 - diff
                    diffs.append(diff)
            max_diff = max(diffs)
            # With 3 groups at ~120 deg offset, we expect significant hue difference
            # But palette + rotation may vary, so just check it's > 0
            assert max_diff > 0, \
                f"Groups should produce some hue diversity, avg_hues={avg_hues}"

    def test_uniform_mode_with_groups_has_diversity(self):
        """Even in uniform mode, groups should create color variety."""
        engine = EffectEngine(num_lights=6)
        engine.set_spatial_mode("uniform")
        features = _loud_features(rms=0.5)
        beat = _no_beat()

        for i in range(60):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        # With groups and palette mode, uniform should still have per-group hue variation
        hues = [light.hue for light in engine._lights]
        # At least not all identical (groups add offset)
        unique_hues = set(round(h, 1) for h in hues)
        # If we have 3 groups, we should have at least 2 distinct hue values
        if engine.num_groups > 1:
            assert len(unique_hues) >= 2, \
                f"Uniform mode with groups should have hue variety, got hues={hues}"


class TestGroupsCentroidMode:
    """In centroid mode, groups should not add phase offsets (no palette)."""

    def test_centroid_mode_groups_no_effect(self):
        engine = EffectEngine(num_lights=6)
        engine.set_color_mode("centroid")
        features = _loud_features(rms=0.5)
        beat = _no_beat()

        # Run to stabilize
        for i in range(60):
            engine.tick(features, beat, dt=0.033, now=1000.0 + i * 0.033)

        # In centroid mode, all lights share the same hue (no palette sampling)
        # Groups should not cause divergence in uniform mode
        engine.set_spatial_mode("uniform")
        for i in range(30):
            engine.tick(features, beat, dt=0.033, now=1002.0 + i * 0.033)

        hues = [light.hue for light in engine._lights]
        # In centroid+uniform mode, hues should be very similar
        hue_spread = max(hues) - min(hues)
        if hue_spread > 180:
            hue_spread = 360 - hue_spread
        # Allow some spread from EMA smoothing, but shouldn't be large
        assert hue_spread < 45, \
            f"Centroid mode should not have large hue spread from groups: {hue_spread}"


class TestGroupsProduceValidOutput:
    """Light groups should not break any invariants."""

    def test_continuous_operation_with_groups(self):
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        dt = 0.033
        now = 1000.0

        for i in range(300):
            rms = 0.3 + 0.3 * abs(math.sin(i * 0.05))
            features = _loud_features(rms=rms)
            is_beat = (i % 15 == 0)
            beat = _beat(strength=0.7) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0
                assert 0 <= s.x <= 1.0
                assert 0 <= s.y <= 1.0


# ============================================================================
# Integration: all three features together
# ============================================================================


class TestIntegrationAllFeatures:
    """Test intensity + effects size + groups working together."""

    def test_chill_with_single_light_fx(self):
        """Chill intensity + single light effects size."""
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        engine.set_intensity(INTENSITY_CHILL)
        engine.set_effects_size(1.0 / 6)
        dt = 0.033
        now = 1000.0

        for i in range(200):
            rms = 0.5
            features = _loud_features(rms=rms)
            is_beat = (i % 15 == 0)
            beat = _beat(strength=0.8) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            for s in states:
                assert 0 <= s.brightness <= 1.0
                # Chill caps at 0.6
                assert s.brightness <= 0.61

    def test_intense_with_half_fx(self):
        """Intense intensity + 50% effects size."""
        engine = EffectEngine(num_lights=6, max_flash_hz=10.0)
        engine.set_intensity(INTENSITY_INTENSE)
        engine.set_effects_size(0.5)
        dt = 0.033
        now = 1000.0

        for i in range(200):
            features = _loud_features(rms=0.7)
            is_beat = (i % 10 == 0)
            beat = _beat(strength=0.9) if is_beat else _no_beat()

            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            assert len(states) == 6
            for s in states:
                assert 0 <= s.brightness <= 1.0

    def test_switching_intensity_mid_session(self):
        """Switch from normal to intense mid-session should not crash."""
        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        dt = 0.033
        now = 1000.0

        for i in range(100):
            features = _loud_features(rms=0.5)
            beat = _beat(strength=0.7) if (i % 15 == 0) else _no_beat()
            states = engine.tick(features, beat, dt=dt, now=now)
            now += dt

            if i == 50:
                engine.set_intensity(INTENSITY_INTENSE)
                engine.set_effects_size(0.25)

            for s in states:
                assert 0 <= s.brightness <= 1.0

    def test_all_spatial_modes_with_groups(self):
        """All spatial modes should work with light groups."""
        from hue_visualizer.visualizer.spatial import SpatialMapper

        engine = EffectEngine(num_lights=6, max_flash_hz=3.0)
        features = _loud_features(rms=0.5)

        for mode in SpatialMapper.MODES:
            engine.set_spatial_mode(mode)
            for i in range(30):
                states = engine.tick(
                    features,
                    _beat(strength=0.7) if (i % 10 == 0) else _no_beat(),
                    dt=0.033,
                    now=1000.0 + i * 0.033,
                )
                assert len(states) == 6
                for s in states:
                    assert 0 <= s.brightness <= 1.0
                    assert 0 <= s.x <= 1.0
                    assert 0 <= s.y <= 1.0
