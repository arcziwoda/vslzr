"""Tests for genre presets and color palettes."""

from hue_visualizer.visualizer.presets import PRESETS, PALETTES, GenrePreset, DEFAULT_GENRE, DEFAULT_PALETTE


class TestPresets:
    def test_all_presets_exist(self):
        assert "techno" in PRESETS
        assert "house" in PRESETS
        assert "dnb" in PRESETS
        assert "ambient" in PRESETS
        assert "trap" in PRESETS

    def test_default_genre_exists(self):
        assert DEFAULT_GENRE in PRESETS

    def test_preset_values_in_range(self):
        for name, p in PRESETS.items():
            assert 50 <= p.beat_cooldown_ms <= 1000, f"{name}: bad cooldown"
            assert 0.5 <= p.bass_boost <= 5.0, f"{name}: bad bass_boost"
            assert 0.0 < p.attack_alpha <= 1.0, f"{name}: bad attack"
            assert 0.0 < p.release_alpha <= 1.0, f"{name}: bad release"
            assert 0.0 <= p.strobe_frequency <= 3.0, f"{name}: strobe exceeds 3 Hz safety"

    def test_preset_is_frozen(self):
        p = PRESETS["techno"]
        try:
            p.beat_cooldown_ms = 999
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_palettes_valid(self):
        for name, palette in PALETTES.items():
            assert len(palette) >= 3, f"{name}: palette needs at least 3 colors"
            for h in palette:
                assert 0 <= h <= 360, f"{name}: hue {h} out of range"

    def test_default_palette_exists(self):
        assert DEFAULT_PALETTE in PALETTES


class TestGenrePaletteLink:
    """Tests for Task 1.11: palettes linked to genre presets."""

    def test_all_presets_have_default_palette(self):
        """Every genre preset must specify a default_palette."""
        for name, p in PRESETS.items():
            assert hasattr(p, "default_palette"), f"{name}: missing default_palette"
            assert isinstance(p.default_palette, str), f"{name}: default_palette not str"

    def test_default_palette_exists_in_palettes(self):
        """Every preset's default_palette must be a valid key in PALETTES."""
        for name, p in PRESETS.items():
            assert p.default_palette in PALETTES, (
                f"{name}: default_palette '{p.default_palette}' not in PALETTES dict"
            )

    def test_genre_specific_palettes_exist(self):
        """Genre-specific palettes should be in PALETTES."""
        for genre_name in ("techno", "house", "dnb", "ambient", "trap"):
            assert genre_name in PALETTES, f"Missing genre palette: {genre_name}"

    def test_techno_palette_has_correct_hues(self):
        """Techno: deep azure (~220), blue (~240), blue-violet (~260), red accent (~0)."""
        palette = PALETTES["techno"]
        assert any(abs(h - 220.0) < 25 for h in palette), "Missing deep azure"
        assert any(abs(h - 240.0) < 25 for h in palette), "Missing blue"
        assert any(h < 20 or h > 340 for h in palette), "Missing red accent"

    def test_house_palette_has_correct_hues(self):
        """House: magenta (~300), rose/pink (~330), amber/gold (~30), purple (~270)."""
        palette = PALETTES["house"]
        assert any(abs(h - 300.0) < 20 for h in palette), "Missing magenta"
        assert any(abs(h - 330.0) < 20 for h in palette), "Missing rose/pink"
        assert any(abs(h - 30.0) < 20 for h in palette), "Missing amber/gold"

    def test_dnb_palette_has_correct_hues(self):
        """DnB: neon green (~120), cyan (~180), blue (~240), violet (~280)."""
        palette = PALETTES["dnb"]
        assert any(abs(h - 120.0) < 20 for h in palette), "Missing neon green"
        assert any(abs(h - 180.0) < 20 for h in palette), "Missing cyan"
        assert any(abs(h - 240.0) < 20 for h in palette), "Missing blue"

    def test_ambient_palette_has_correct_hues(self):
        """Ambient: deep blue (~240), purple (~270), teal (~200), warm amber (~30)."""
        palette = PALETTES["ambient"]
        assert any(abs(h - 240.0) < 25 for h in palette), "Missing deep blue"
        assert any(abs(h - 270.0) < 25 for h in palette), "Missing purple"
        assert any(abs(h - 200.0) < 25 for h in palette), "Missing teal"

    def test_trap_palette_has_correct_hues(self):
        """Trap: red (~0), purple (~270), orange/gold (~30), pink/rose (~330)."""
        palette = PALETTES["trap"]
        assert any(h < 20 or h > 340 for h in palette), "Missing red"
        assert any(abs(h - 270.0) < 25 for h in palette), "Missing purple"
        assert any(abs(h - 30.0) < 20 for h in palette), "Missing orange/gold"

    def test_each_genre_has_unique_palette(self):
        """Each genre should map to a different palette."""
        palette_names = set()
        for name, p in PRESETS.items():
            palette_names.add(p.default_palette)
        assert len(palette_names) == len(PRESETS), "Some genres share the same default palette"

    def test_preset_default_palette_field_is_frozen(self):
        """default_palette should be frozen like other fields."""
        p = PRESETS["techno"]
        try:
            p.default_palette = "fire"
            assert False, "Should be frozen"
        except AttributeError:
            pass
