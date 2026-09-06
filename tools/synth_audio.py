"""Deterministic synthetic tracks with exact ground-truth beat times.

Used by tools/synth_benchmark.py to measure beat-detector changes against a
known beat grid. Audio realism is deliberately modest — what matters is that

  * the ground-truth beat times are exact, and
  * every instrument lands in the frequency band the detector expects
    (kick/808 in sub-bass + bass, snare/clap in low-mid..upper-mid,
    hats/shakers above 6 kHz).

Everything is driven by a seeded numpy Generator, so a given
(scenario, seed, humanize_ms) triple always produces bit-identical audio.

Usage:
    uv run python tools/synth_audio.py --save-dir /tmp/synth        # all scenarios
    uv run python tools/synth_audio.py --scenario dnb_174 --save-dir /tmp/synth
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

SAMPLE_RATE = 44100

# --- Filters -------------------------------------------------------------


@lru_cache(maxsize=32)
def _sos_band(sr: int, lo: float, hi: float, order: int) -> np.ndarray:
    nyq = sr / 2.0
    lo_n = max(lo, 10.0) / nyq
    hi_n = min(hi, nyq * 0.99) / nyq
    return signal.butter(order, [lo_n, hi_n], btype="band", output="sos")


@lru_cache(maxsize=32)
def _sos_low(sr: int, cutoff: float, order: int) -> np.ndarray:
    nyq = sr / 2.0
    return signal.butter(order, min(cutoff, nyq * 0.99) / nyq, btype="low", output="sos")


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    return signal.sosfilt(_sos_band(sr, lo, hi, order), x)


def lowpass(x: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    return signal.sosfilt(_sos_low(sr, cutoff, order), x)


def _saw(freq: float, n: int, sr: int, phase0: float = 0.0) -> np.ndarray:
    t = np.arange(n) / sr
    return 2.0 * ((freq * t + phase0) % 1.0) - 1.0


def _attack(env: np.ndarray, sr: int, attack_ms: float = 2.0) -> np.ndarray:
    """Apply a short linear fade-in so a hit does not start with a step."""
    a = max(1, int(sr * attack_ms / 1000.0))
    a = min(a, len(env))
    env[:a] *= np.linspace(0.0, 1.0, a)
    return env


# --- Instruments ---------------------------------------------------------


def kick(
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
    duration: float = 0.25,
    f_start: float = 150.0,
    f_end: float = 50.0,
    click: float = 0.30,
) -> np.ndarray:
    """Short click plus a decaying sine with a 150 -> 50 Hz pitch sweep."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    freq = f_end + (f_start - f_end) * np.exp(-t / 0.040)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    env = _attack(np.exp(-t / 0.090), sr, 1.0)
    out = np.sin(phase) * env

    if click > 0.0:
        n_click = int(sr * 0.004)
        noise = rng.standard_normal(n_click)
        click_env = np.exp(-np.arange(n_click) / sr / 0.0012)
        out[:n_click] += click * noise * click_env

    return out


def hihat(
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
    duration: float = 0.012,
    f_lo: float = 6000.0,
    f_hi: float = 12000.0,
) -> np.ndarray:
    """Bandpassed noise burst — closed hat at ~12 ms, open hat at ~120 ms."""
    n = int(sr * duration)
    y = bandpass(rng.standard_normal(n), sr, f_lo, f_hi)
    env = _attack(np.exp(-np.arange(n) / sr / (duration * 0.35)), sr, 0.5)
    return y * env


def clap(
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
    duration: float = 0.12,
) -> np.ndarray:
    """Clap / snare: 200-4000 Hz noise with three smeared taps plus a 180 Hz tone."""
    n = int(sr * duration)
    idx = np.arange(n)
    noise = bandpass(rng.standard_normal(n), sr, 200.0, 4000.0)
    body = noise * _attack(np.exp(-idx / sr / 0.035), sr, 0.5)

    out = np.zeros(n)
    for delay_ms, gain in ((0.0, 1.0), (9.0, 0.65), (18.0, 0.45)):
        d = int(sr * delay_ms / 1000.0)
        out[d:] += gain * body[: n - d]

    tone = np.sin(2.0 * np.pi * 180.0 * idx / sr) * np.exp(-idx / sr / 0.030)
    return out * 0.75 + tone * 0.25


def sub808(
    sr: int = SAMPLE_RATE,
    duration: float = 0.6,
    f_start: float = 68.0,
    f_end: float = 40.0,
) -> np.ndarray:
    """Long sub sine with a slow pitch drop — sits almost entirely in sub-bass."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    freq = f_end + (f_start - f_end) * np.exp(-t / 0.250)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    env = _attack(np.exp(-t / 0.280), sr, 5.0)
    return np.sin(phase) * env


def perc(
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
    duration: float = 0.06,
) -> np.ndarray:
    """Short mid-band percussion hit (rim / tight tom) for syncopated patterns."""
    n = int(sr * duration)
    idx = np.arange(n)
    noise = bandpass(rng.standard_normal(n), sr, 400.0, 3500.0)
    env = _attack(np.exp(-idx / sr / 0.018), sr, 0.5)
    tone = np.sin(2.0 * np.pi * 320.0 * idx / sr) * np.exp(-idx / sr / 0.020)
    return noise * env * 0.8 + tone * 0.3


def bass_note(
    sr: int = SAMPLE_RATE,
    duration: float = 0.09,
    freq: float = 55.0,
) -> np.ndarray:
    """Short plucked bass note (lowpassed saw) for rolling 16th basslines."""
    n = int(sr * duration)
    y = lowpass(_saw(freq, n, sr), sr, 350.0)
    env = _attack(np.exp(-np.arange(n) / sr / 0.035), sr, 2.0)
    return y * env


def reese_bass(n: int, sr: int = SAMPLE_RATE, freq: float = 46.0) -> np.ndarray:
    """Continuous detuned-saw bass with audible beating — DnB Reese stand-in."""
    y = (
        _saw(freq, n, sr)
        + _saw(freq + 0.7, n, sr, phase0=0.31)
        + 0.6 * _saw(freq * 2.0 + 1.1, n, sr, phase0=0.67)
    )
    return lowpass(y, sr, 400.0) / 2.6


def pad(n: int, sr: int = SAMPLE_RATE, root: float = 110.0) -> np.ndarray:
    """Sustained pad: three detuned saw+sine layers (root, fifth, octave)."""
    out = np.zeros(n)
    for freq, gain in ((root, 1.0), (root * 1.5, 0.7), (root * 2.0, 0.5)):
        out += gain * (
            0.5 * _saw(freq, n, sr)
            + 0.5 * _saw(freq * 1.003, n, sr, phase0=0.5)
            + 0.6 * np.sin(2.0 * np.pi * freq * 0.997 * np.arange(n) / sr)
        )
    return lowpass(out, sr, 2500.0) / 3.0


def sidechain_envelope(
    n: int,
    sr: int,
    beat_times: np.ndarray,
    beat_period: float,
    depth: float = 0.85,
    release_frac: float = 0.6,
) -> np.ndarray:
    """Pumping envelope keyed to a beat grid.

    A fast dip at every beat followed by an exponential release whose 95%
    recovery lands at `release_frac` of the beat period.
    """
    env = np.ones(n)
    tau = max(release_frac * beat_period / 3.0, 0.02)
    t = np.arange(n) / sr

    for beat_time in beat_times:
        i0 = int(beat_time * sr)
        if i0 >= n:
            break
        i0 = max(i0, 0)
        curve = 1.0 - depth * np.exp(-(t[i0:] - beat_time) / tau)
        env[i0:] = np.minimum(env[i0:], curve)

    # Smooth the instantaneous drop into a ~4 ms ramp so it does not click.
    smooth_len = max(3, int(sr * 0.004))
    kernel = np.hanning(smooth_len)
    kernel /= kernel.sum()
    return np.convolve(env, kernel, mode="same")


# --- Timing grid ---------------------------------------------------------


@dataclass
class Grid:
    """Timing grid of 32nd-note steps (8 per beat), optionally humanized."""

    bpm: float
    duration: float
    times: np.ndarray
    steps_per_beat: int = 8

    @classmethod
    def build(
        cls,
        bpm: float,
        duration: float,
        rng: np.random.Generator,
        humanize_ms: float = 0.0,
        steps_per_beat: int = 8,
    ) -> "Grid":
        period = 60.0 / bpm
        step = period / steps_per_beat
        n_steps = int(np.ceil(duration / step)) + 4 * steps_per_beat
        times = np.arange(n_steps) * step
        if humanize_ms > 0.0:
            times = times + rng.normal(0.0, humanize_ms / 1000.0, n_steps)
        return cls(bpm=bpm, duration=duration, times=np.maximum(times, 0.0),
                   steps_per_beat=steps_per_beat)

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm

    def at(self, beats: float) -> float:
        """Time (seconds) of the grid position `beats` quarter notes from the start."""
        idx = min(int(round(beats * self.steps_per_beat)), len(self.times) - 1)
        return float(self.times[idx])

    @property
    def n_beats(self) -> int:
        return int(self.duration / self.beat_period)

    def beat_times(self) -> np.ndarray:
        beats = self.times[:: self.steps_per_beat][: self.n_beats]
        return beats[beats < self.duration]


# --- Track buffer --------------------------------------------------------


class Track:
    """Mono mix buffer with one-shot placement at absolute times."""

    def __init__(self, duration: float, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.duration = duration
        # One extra second of tail room so hits near the end are not clipped mid-decay.
        self.buf = np.zeros(int((duration + 1.0) * sr))

    def add(self, sample: np.ndarray, time_sec: float, gain: float = 1.0) -> None:
        i0 = int(round(time_sec * self.sr))
        if i0 < 0 or i0 >= len(self.buf):
            return
        i1 = min(len(self.buf), i0 + len(sample))
        self.buf[i0:i1] += gain * sample[: i1 - i0]

    def add_signal(self, sig: np.ndarray, gain: float = 1.0, offset_sec: float = 0.0) -> None:
        i0 = int(round(offset_sec * self.sr))
        i1 = min(len(self.buf), i0 + len(sig))
        self.buf[i0:i1] += gain * sig[: i1 - i0]

    def finish(self, peak: float = 0.9) -> np.ndarray:
        out = self.buf[: int(self.duration * self.sr)]
        max_abs = float(np.max(np.abs(out))) if len(out) else 0.0
        if max_abs > 1e-9:
            out = out * (peak / max_abs)
        return out.astype(np.float32)


def _fade(sig: np.ndarray, sr: int, fade_sec: float = 0.5) -> np.ndarray:
    n = min(int(sr * fade_sec), len(sig) // 2)
    if n <= 0:
        return sig
    sig = sig.copy()
    sig[:n] *= np.linspace(0.0, 1.0, n)
    sig[-n:] *= np.linspace(1.0, 0.0, n)
    return sig


# --- Pattern builders ----------------------------------------------------

# Rolling 16th bassline: semitone offsets from A1 (55 Hz), one bar of 16ths.
_BASS_SEQUENCE = (0, 0, -2, 0, 0, 3, 0, -2, 0, 0, -2, 0, 5, 0, 3, 0)


def _render_techno_beat(
    track: Track,
    grid: Grid,
    rng: np.random.Generator,
    beat: int,
    synco: bool,
) -> None:
    """One quarter note of the techno pattern (kick, hats, clap, bass, opt. perc)."""
    t_beat = grid.at(beat)

    track.add(kick(rng), t_beat, 1.0)
    track.add(hihat(rng, duration=0.012), t_beat, 0.32)
    # Open hat on the offbeat 8th.
    track.add(hihat(rng, duration=0.120), grid.at(beat + 0.5), 0.28)
    # Clap on 2 and 4.
    if beat % 4 in (1, 3):
        track.add(clap(rng), t_beat, 0.55)
    # Rolling bass on 16ths.
    for step in range(4):
        idx = (beat * 4 + step) % len(_BASS_SEQUENCE)
        freq = 55.0 * 2.0 ** (_BASS_SEQUENCE[idx] / 12.0)
        track.add(bass_note(freq=freq), grid.at(beat + step * 0.25), 0.26)
    # Percussion on the "and-a" (0.75 of a beat after the kick).
    if synco:
        track.add(perc(rng), grid.at(beat + 0.75), 0.45)


def _make_techno(
    duration: float,
    seed: int,
    humanize_ms: float,
    synco: bool = False,
    bpm: float = 130.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grid = Grid.build(bpm, duration, rng, humanize_ms)
    track = Track(duration)
    for beat in range(grid.n_beats):
        _render_techno_beat(track, grid, rng, beat, synco)
    return track.finish(), grid.beat_times()


# --- Scenarios -----------------------------------------------------------


def techno_130(seed: int = 0, humanize_ms: float = 0.0, duration: float = 60.0):
    """130 BPM techno: 4/4 kick, 8th hats + offbeat open hat, clap on 2 and 4, 16th bass."""
    audio, beats = _make_techno(duration, seed, humanize_ms, synco=False)
    meta = _meta("techno_130", 130.0, duration, seed, humanize_ms, "techno",
                 "Straight 4/4 techno with rolling 16th bass")
    return audio, beats, meta


def techno_130_synco(seed: int = 0, humanize_ms: float = 0.0, duration: float = 60.0):
    """techno_130 plus a percussion hit on every 'and-a' (0.75 beat after each kick)."""
    audio, beats = _make_techno(duration, seed, humanize_ms, synco=True)
    meta = _meta("techno_130_synco", 130.0, duration, seed, humanize_ms, "techno",
                 "techno_130 + syncopated perc on the 16th before each kick")
    return audio, beats, meta


def house_125(seed: int = 0, humanize_ms: float = 0.0, duration: float = 60.0):
    """125 BPM house: 4/4 kick, offbeat open hats, 16th shaker, clap on 2/4, sidechained pad."""
    rng = np.random.default_rng(seed)
    grid = Grid.build(125.0, duration, rng, humanize_ms)
    track = Track(duration)

    for beat in range(grid.n_beats):
        t_beat = grid.at(beat)
        track.add(kick(rng, f_start=140.0, f_end=48.0), t_beat, 1.0)
        track.add(hihat(rng, duration=0.130), grid.at(beat + 0.5), 0.30)
        if beat % 4 in (1, 3):
            track.add(clap(rng), t_beat, 0.50)
        for step in range(4):
            track.add(
                hihat(rng, duration=0.020, f_lo=8000.0, f_hi=14000.0),
                grid.at(beat + step * 0.25),
                0.14,
            )

    n = len(track.buf)
    env = sidechain_envelope(n, SAMPLE_RATE, grid.beat_times(), grid.beat_period)
    track.add_signal(pad(n, root=110.0) * env, gain=0.45)

    meta = _meta("house_125", 125.0, duration, seed, humanize_ms, "house",
                 "House groove with a sidechained pad")
    return track.finish(), grid.beat_times(), meta


def dnb_174(seed: int = 0, humanize_ms: float = 0.0, duration: float = 60.0):
    """174 BPM two-step: kick on 1 and 2.5, snare on 2 and 4, 8th hats, Reese bass.

    Ground truth is the 174 BPM quarter-note grid, not the kick pattern.
    """
    rng = np.random.default_rng(seed)
    grid = Grid.build(174.0, duration, rng, humanize_ms)
    track = Track(duration)

    for bar_start in range(0, grid.n_beats, 4):
        # Canonical two-step: kick on beat 1 and on the "and" of 3 (2.5 in
        # zero-based beats), snare on beats 2 and 4.
        for offset in (0.0, 2.5):
            track.add(kick(rng, duration=0.20, f_start=170.0, f_end=52.0),
                      grid.at(bar_start + offset), 1.0)
        for offset in (1.0, 3.0):
            track.add(clap(rng, duration=0.10), grid.at(bar_start + offset), 0.65)
        for step in range(8):
            track.add(
                hihat(rng, duration=0.010 if step % 2 else 0.014),
                grid.at(bar_start + step * 0.5),
                0.22 if step % 2 else 0.30,
            )

    n = len(track.buf)
    track.add_signal(_fade(reese_bass(n, freq=46.0), SAMPLE_RATE, 0.2), gain=0.40)

    meta = _meta("dnb_174", 174.0, duration, seed, humanize_ms, "dnb",
                 "Two-step DnB with Reese bass; ground truth is the quarter grid")
    return track.finish(), grid.beat_times(), meta


def trap_70(seed: int = 0, humanize_ms: float = 0.0, duration: float = 60.0):
    """70 BPM half-time trap: 808 on 1 and 2.5, snare on 3, 16th hats with 32nd rolls.

    Ground truth is the 70 BPM quarter-note grid (the felt half-time pulse).
    """
    rng = np.random.default_rng(seed)
    grid = Grid.build(70.0, duration, rng, humanize_ms)
    track = Track(duration)

    for bar_start in range(0, grid.n_beats, 4):
        for offset in (0.0, 2.5):
            track.add(sub808(), grid.at(bar_start + offset), 1.0)
            track.add(kick(rng, duration=0.10, f_start=160.0, f_end=60.0, click=0.4),
                      grid.at(bar_start + offset), 0.30)
        track.add(clap(rng), grid.at(bar_start + 2.0), 0.60)

        for beat in range(4):
            roll = rng.random() < 0.18
            n_sub = 8 if roll else 4
            for step in range(n_sub):
                track.add(
                    hihat(rng, duration=0.008, f_lo=7000.0, f_hi=13000.0),
                    grid.at(bar_start + beat + step / n_sub),
                    0.22,
                )

    meta = _meta("trap_70", 70.0, duration, seed, humanize_ms, "trap",
                 "Half-time trap; ground truth is the 70 BPM quarter grid")
    return track.finish(), grid.beat_times(), meta


def techno_breakdown(seed: int = 0, humanize_ms: float = 0.0, duration: float = 60.0):
    """130 BPM techno with a 20-40 s breakdown (pad + sidechain only).

    The ground-truth grid runs unbroken through the breakdown.
    """
    rng = np.random.default_rng(seed)
    grid = Grid.build(130.0, duration, rng, humanize_ms)
    track = Track(duration)

    break_start, break_end = 20.0, 40.0
    for beat in range(grid.n_beats):
        if break_start <= grid.at(beat) < break_end:
            continue
        _render_techno_beat(track, grid, rng, beat, synco=False)

    seg_len = int((break_end - break_start) * SAMPLE_RATE)
    seg_beats = grid.beat_times()
    seg_beats = seg_beats[(seg_beats >= break_start) & (seg_beats < break_end)] - break_start
    env = sidechain_envelope(seg_len, SAMPLE_RATE, seg_beats, grid.beat_period)
    track.add_signal(_fade(pad(seg_len, root=98.0) * env, SAMPLE_RATE, 0.5),
                     gain=0.75, offset_sec=break_start)

    meta = _meta("techno_breakdown", 130.0, duration, seed, humanize_ms, "techno",
                 "Full / breakdown / full; grid continues through the breakdown")
    return track.finish(), grid.beat_times(), meta


def silence_then_start(seed: int = 0, humanize_ms: float = 0.0, duration: float = 45.0):
    """5 s of silence then 40 s of techno_130 — cold-start measurement."""
    lead_in = 5.0
    music_dur = duration - lead_in
    audio, beats = _make_techno(music_dur, seed, humanize_ms, synco=False)
    silence = np.zeros(int(lead_in * SAMPLE_RATE), dtype=np.float32)
    meta = _meta("silence_then_start", 130.0, duration, seed, humanize_ms, "techno",
                 "5 s silence then techno_130 (cold start)")
    return np.concatenate([silence, audio]), beats + lead_in, meta


def _meta(name, bpm, duration, seed, humanize_ms, preset, description) -> dict:
    return {
        "name": name,
        "true_bpm": bpm,
        "duration_sec": duration,
        "sample_rate": SAMPLE_RATE,
        "seed": seed,
        "humanize_ms": humanize_ms,
        "preset": preset,
        "description": description,
    }


SCENARIOS = {
    "techno_130": techno_130,
    "techno_130_synco": techno_130_synco,
    "house_125": house_125,
    "dnb_174": dnb_174,
    "trap_70": trap_70,
    "techno_breakdown": techno_breakdown,
    "silence_then_start": silence_then_start,
}


def generate(name: str, seed: int = 0, humanize_ms: float = 0.0):
    """Build one scenario. Returns (audio float32 mono @44100, beat_times_sec, meta)."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
    return SCENARIOS[name](seed=seed, humanize_ms=humanize_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark tracks")
    parser.add_argument("--scenario", default=None, help="Scenario name (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--humanize-ms", type=float, default=0.0,
                        help="Std-dev of per-step timing jitter in ms (default: 0)")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to write <scenario>.wav and <scenario>_beats.txt")
    args = parser.parse_args()

    names = [args.scenario] if args.scenario else list(SCENARIOS)

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        audio, beats, meta = generate(name, seed=args.seed, humanize_ms=args.humanize_ms)
        print(f"{name:20s} {len(audio)/SAMPLE_RATE:6.1f}s  {len(beats):4d} beats  "
              f"{meta['true_bpm']:6.1f} BPM  peak={np.max(np.abs(audio)):.2f}")
        if save_dir is not None:
            import soundfile as sf

            sf.write(save_dir / f"{name}.wav", audio, SAMPLE_RATE)
            np.savetxt(save_dir / f"{name}_beats.txt", beats, fmt="%.6f")
    if save_dir is not None:
        print(f"\nWritten to: {save_dir}")


if __name__ == "__main__":
    main()
