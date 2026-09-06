"""End-to-end light-trigger benchmark on synthetic scenarios.

Runs the real server glue (AudioPipeline.process_all + consume_*), rebuilds the
per-tick BeatInfo the way server/app.py:audio_loop does, ticks EffectEngine at
FPS_TARGET and scores the moments a beat flash starts (flash_onset_this_tick)
against the scenario's ground-truth beats.

Usage:
    uv run python tools/engine_benchmark.py [--scenario name] [--tag name]

Two engine configurations per scenario, both with the matching genre preset:
    reactive  - metric filter OFF (raw onsets + snare flashes), predictive ON
    metric    - metric filter ON, predictive ON
Latency compensation is LATENCY_COMP_MS (the engine fires that much before the
predicted beat so the bulb lights on the beat); flash times are shifted by the
same amount before scoring. With 0 ms the predictive path could never fire: the
prediction jumps to the next beat exactly when it would have become due.

Note: the BeatInfo rebuild below mirrors audio_loop and must be kept in sync
with it (that duplication is exactly where the is_metric_beat bug hid).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hue_visualizer.audio.beat_detector import BeatInfo  # noqa: E402
from hue_visualizer.core.config import Settings  # noqa: E402
from hue_visualizer.server import app as server_app  # noqa: E402
from hue_visualizer.visualizer.engine import EffectEngine  # noqa: E402
from hue_visualizer.visualizer.presets import PRESETS  # noqa: E402
from synth_audio import SAMPLE_RATE, SCENARIOS, generate  # noqa: E402
from synth_benchmark import COLD_START_SEC, score_stream  # noqa: E402

HOP = 1024
FPS = 50
LATENCY_COMP_MS = 60.0


class _Clock:
    """Replaces time.monotonic inside server.app for a deterministic run."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def run_engine(
    audio: np.ndarray,
    preset_name: str,
    metric_filter: bool,
    num_lights: int = 6,
    band_effects: bool = True,
    onset_source: str = "rnn",
) -> dict:
    settings = Settings(beat_onset_source=onset_source)
    pipeline = server_app.AudioPipeline(settings)
    preset = PRESETS[preset_name]
    pipeline.beat_detector.set_cooldown(preset.beat_cooldown_ms)
    pipeline.beat_detector.set_bpm_range(preset.bpm_min, preset.bpm_max)
    pipeline.analyzer.bass_boost = preset.bass_boost

    engine = EffectEngine(
        num_lights=num_lights,
        attack_alpha=preset.attack_alpha,
        release_alpha=preset.release_alpha,
        spatial_mode=preset.spatial_mode,
        latency_compensation_ms=LATENCY_COMP_MS,
        predictive_confidence_threshold=settings.predictive_confidence_threshold,
    )
    engine.set_flash_tau(preset.flash_tau)
    engine.set_metric_filter(metric_filter)

    clock = _Clock()
    original_monotonic = server_app.time.monotonic
    server_app.time.monotonic = clock

    flashes: list[float] = []
    triggers: list[float] = []
    original_resolve = engine._resolve_beat_trigger

    def resolve(beat_info, now):
        fired, strength = original_resolve(beat_info, now)
        if fired:
            triggers.append(now)
        return fired, strength

    engine._resolve_beat_trigger = resolve
    predictive_active_ticks = 0
    ticks = 0
    frame_dur = HOP / SAMPLE_RATE
    tick_dt = 1.0 / FPS
    n_frames = len(audio) // HOP
    next_frame = 0
    try:
        t = 0.0
        while next_frame < n_frames:
            clock.now = t
            # Frames whose capture completed by now become available, like the
            # capture thread appending to the ring buffer
            while next_frame < n_frames and (next_frame + 1) * frame_dur <= t:
                frame = audio[next_frame * HOP:(next_frame + 1) * HOP]
                with pipeline.capture._lock:
                    pipeline.capture._frames.append(frame)
                next_frame += 1

            pipeline.process_all()
            had_beat, had_metric, strength = pipeline.consume_beat()
            kick, snare, hihat, kick_e, snare_e, hihat_e = pipeline.consume_band_onsets()
            features = pipeline.consume_features()

            # Mirror of server/app.py:audio_loop
            beat_for_engine = BeatInfo(
                is_beat=had_beat,
                is_metric_beat=had_metric,
                bpm=pipeline.beat_info.bpm,
                bpm_confidence=pipeline.beat_info.bpm_confidence,
                beat_strength=strength,
                predicted_next_beat=pipeline.beat_info.predicted_next_beat,
                time_since_beat=pipeline.beat_info.time_since_beat,
                kick_onset=kick and band_effects,
                snare_onset=snare and band_effects,
                hihat_onset=hihat and band_effects,
                kick_energy=kick_e,
                snare_energy=snare_e,
                hihat_energy=hihat_e,
            )
            engine.tick(
                features, beat_for_engine, tick_dt, now=t,
                section_info=pipeline.section_info,
            )
            if any(light.flash_onset_this_tick for light in engine._lights):
                flashes.append(t)
            ticks += 1
            if (
                pipeline.beat_info.bpm_confidence
                >= settings.predictive_confidence_threshold
            ):
                predictive_active_ticks += 1
            t += tick_dt
    finally:
        server_app.time.monotonic = original_monotonic

    return {
        "flashes": flashes,
        "triggers": triggers,
        "predictive_active_fraction": round(predictive_active_ticks / max(ticks, 1), 3),
    }


def run_scenario(name: str, seed: int, band_effects: bool = True,
                 onset_source: str = "rnn") -> dict:
    audio, beats, meta = generate(name, seed=seed)
    duration = len(audio) / SAMPLE_RATE
    preset_name = meta["preset"]
    configs = {}
    for label, metric in (("reactive", False), ("metric", True)):
        out = run_engine(audio, preset_name, metric, band_effects=band_effects,
                         onset_source=onset_source)
        intended = np.asarray(out["flashes"]) + LATENCY_COMP_MS / 1000.0
        scores = score_stream(np.asarray(beats), intended, duration)
        scores["predictive_active_fraction"] = out["predictive_active_fraction"]
        scores["trigger_count"] = len([t for t in out["triggers"] if t >= COLD_START_SEC])
        scores["flash_count"] = len([t for t in out["flashes"] if t >= COLD_START_SEC])
        configs[label] = scores
    return {
        "scenario": name,
        "description": meta.get("description", ""),
        "true_bpm": meta.get("bpm"),
        "preset": preset_name,
        "configs": configs,
    }


def print_table(results: list[dict]) -> None:
    print(f"\n{'scenario':<20}{'engine':<10}{'F':>7}{'CMLc':>7}{'CMLt':>7}{'FP/min':>8}"
          f"{'Miss/min':>10}{'err_ms':>8}{'pred%':>7}{'trig':>6}{'flash':>6}")
    print("-" * 96)
    for r in results:
        for label, s in r["configs"].items():
            err = s["mean_signed_error_ms"]
            print(f"{r['scenario']:<20}{label:<10}{s['f_measure']:>7.3f}{s['cmlc']:>7.3f}"
                  f"{s['cmlt']:>7.3f}{s['false_positives_per_min']:>8.1f}"
                  f"{s['misses_per_min']:>10.1f}{(err if err is not None else float('nan')):>8.1f}"
                  f"{s['predictive_active_fraction'] * 100:>6.0f}%"
                  f"{s['trigger_count']:>6}{s['flash_count']:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end light flash benchmark")
    parser.add_argument("--scenario", default=None, help="Run one scenario (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--no-band-effects", action="store_true",
                        help="Suppress kick/snare/hihat onsets reaching the engine (diagnostic)")
    parser.add_argument("--onset-source", choices=("rnn", "spectral"), default="rnn")
    args = parser.parse_args()

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    results = [run_scenario(n, args.seed, band_effects=not args.no_band_effects,
                            onset_source=args.onset_source) for n in names]
    print_table(results)

    out_dir = Path(__file__).parent / "benchmark_results" / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        path = out_dir / f"{r['scenario']}_engine_{args.tag}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
    print(f"\nResults written to: {out_dir} (engine_{args.tag})")
    print(f"Scoring skips the first {COLD_START_SEC:.0f} s.")


if __name__ == "__main__":
    main()
