"""Score the beat detector against synthetic tracks with known beat grids.

Runs every scenario from tools/synth_audio.py through
AudioAnalyzer -> BeatDetector frame by frame (hop 1024, fft 2048, timestamp =
offset / sample_rate), exactly the way tools/benchmark_beats.py drives a real
file, and scores three estimated beat streams against the ground truth:

  raw        — every BeatInfo.is_beat (unfiltered onset stream)
  metric     — is_beat and is_metric_beat (PLL-validated onsets)
  predicted  — distinct BeatInfo.predicted_next_beat values (the stream the
               engine actually uses for predictive triggering)

Scoring uses mir_eval.beat.evaluate, which drops everything before 5.0 s
(the cold-start convention from docs/research/beat-detection-benchmarking-research.md).

Each scenario is run twice: once with the detector defaults (or whatever
--bpm-range / --cooldown-ms says) and once with the matching genre preset
from src/hue_visualizer/visualizer/presets.py.

Usage:
    uv run python tools/synth_benchmark.py --tag baseline
    uv run python tools/synth_benchmark.py --scenario dnb_174 --tag my_change
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import mir_eval  # noqa: E402

from hue_visualizer.audio.analyzer import AudioAnalyzer  # noqa: E402
from hue_visualizer.audio.beat_detector import BeatDetector  # noqa: E402
from hue_visualizer.visualizer.presets import PRESETS  # noqa: E402
from synth_audio import SAMPLE_RATE, SCENARIOS, generate  # noqa: E402

HOP_SIZE = 1024
FFT_SIZE = 2048
COLD_START_SEC = 5.0
MATCH_TOLERANCE_SEC = 0.070
PREDICTED_DEDUPE_SEC = 0.030

# Detector constructor defaults (BeatDetector.__init__).
DEFAULT_BPM_MIN = 80.0
DEFAULT_BPM_MAX = 180.0
DEFAULT_COOLDOWN_MS = 300.0
DEFAULT_BASS_BOOST = 2.0


def _dedupe(times: list[float], min_gap: float = PREDICTED_DEDUPE_SEC) -> np.ndarray:
    """Collapse event times that are within `min_gap` of the previous kept one."""
    kept: list[float] = []
    for t in times:
        if not kept or t - kept[-1] > min_gap:
            kept.append(t)
    return np.array(kept)


def run_detector(
    audio: np.ndarray,
    sr: int,
    bpm_min: float,
    bpm_max: float,
    cooldown_ms: float,
    bass_boost: float,
) -> dict:
    """Drive the pipeline frame by frame and collect the three beat streams."""
    analyzer = AudioAnalyzer(
        sample_rate=sr, fft_size=FFT_SIZE, hop_size=HOP_SIZE, bass_boost=bass_boost
    )
    detector = BeatDetector(
        sample_rate=sr,
        hop_size=HOP_SIZE,
        cooldown_ms=cooldown_ms,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
    )

    frame_dur = HOP_SIZE / sr
    n_frames = (len(audio) - FFT_SIZE) // HOP_SIZE + 1
    trace_every = max(1, int(0.25 / frame_dur))

    raw_beats: list[float] = []
    metric_beats: list[float] = []
    predicted_beats: list[float] = []
    bpm_trace: list[dict] = []

    t_start = time.perf_counter()
    for i in range(n_frames):
        offset = i * HOP_SIZE
        frame = audio[offset:offset + HOP_SIZE]
        if len(frame) < HOP_SIZE:
            break

        t = offset / sr
        features = analyzer.analyze(frame)
        info = detector.detect(features, timestamp=t)

        if i % trace_every == 0:
            bpm_trace.append({
                "time": round(t, 3),
                "bpm": round(info.bpm, 1),
                "confidence": round(info.bpm_confidence, 3),
            })

        if info.is_beat:
            raw_beats.append(t)
            if info.is_metric_beat:
                metric_beats.append(t)

        # Predicted beats: distinct predicted_next_beat values. The value only
        # moves when the PLL re-anchors, so dedupe by a 30 ms difference.
        pred = info.predicted_next_beat
        if pred > 0.0 and (
            not predicted_beats or abs(pred - predicted_beats[-1]) > PREDICTED_DEDUPE_SEC
        ):
            predicted_beats.append(pred)

    elapsed = time.perf_counter() - t_start

    final_bpm = 0.0
    for entry in reversed(bpm_trace):
        if entry["bpm"] > 0:
            final_bpm = entry["bpm"]
            break

    lock_on = None
    for entry in bpm_trace:
        if entry["confidence"] > 0.6 and entry["bpm"] > 0:
            lock_on = entry["time"]
            break

    conf_after = [e["confidence"] for e in bpm_trace if e["time"] >= COLD_START_SEC]

    return {
        "streams": {
            "raw": np.array(raw_beats),
            "metric": np.array(metric_beats),
            # predicted_next_beat can jump backwards when the PLL period
            # shrinks, so sort before the final dedupe — mir_eval requires
            # strictly increasing event times.
            "predicted": _dedupe(sorted(predicted_beats)),
        },
        "bpm_trace": bpm_trace,
        "final_bpm": final_bpm,
        "lock_on_time_sec": lock_on,
        "mean_confidence_after_5s": float(np.mean(conf_after)) if conf_after else 0.0,
        "realtime_ratio": (len(audio) / sr) / elapsed if elapsed > 0 else 0.0,
        "frames_processed": n_frames,
    }


def _match_stats(reference: np.ndarray, estimated: np.ndarray, duration: float) -> dict:
    """False positives, misses and mean signed timing error past the cold start."""
    ref = reference[reference >= COLD_START_SEC]
    est = estimated[estimated >= COLD_START_SEC]
    scored_minutes = max(duration - COLD_START_SEC, 1e-6) / 60.0

    if len(ref) == 0 or len(est) == 0:
        return {
            "estimated_count": int(len(est)),
            "reference_count": int(len(ref)),
            "false_positives_per_min": round(len(est) / scored_minutes, 2),
            "misses_per_min": round(len(ref) / scored_minutes, 2),
            "mean_signed_error_ms": None,
        }

    errors = []
    false_positives = 0
    for e in est:
        delta = e - ref[int(np.argmin(np.abs(ref - e)))]
        if abs(delta) <= MATCH_TOLERANCE_SEC:
            errors.append(delta)
        else:
            false_positives += 1

    misses = int(np.sum(
        np.array([np.min(np.abs(est - r)) for r in ref]) > MATCH_TOLERANCE_SEC
    ))

    return {
        "estimated_count": int(len(est)),
        "reference_count": int(len(ref)),
        "false_positives_per_min": round(false_positives / scored_minutes, 2),
        "misses_per_min": round(misses / scored_minutes, 2),
        "mean_signed_error_ms": round(float(np.mean(errors)) * 1000.0, 1) if errors else None,
    }


def score_stream(reference: np.ndarray, estimated: np.ndarray, duration: float) -> dict:
    """mir_eval metric suite plus FP/miss/timing stats for one beat stream."""
    scores = mir_eval.beat.evaluate(np.asarray(reference, dtype=float),
                                    np.asarray(estimated, dtype=float))
    out = {
        "f_measure": round(float(scores["F-measure"]), 4),
        "cemgil": round(float(scores["Cemgil"]), 4),
        "p_score": round(float(scores["P-score"]), 4),
        "goto": round(float(scores["Goto"]), 4),
        "cmlc": round(float(scores["Correct Metric Level Continuous"]), 4),
        "cmlt": round(float(scores["Correct Metric Level Total"]), 4),
        "amlc": round(float(scores["Any Metric Level Continuous"]), 4),
        "amlt": round(float(scores["Any Metric Level Total"]), 4),
        "information_gain": round(float(scores["Information gain"]), 4),
    }
    out.update(_match_stats(reference, estimated, duration))
    return out


def run_scenario(name: str, seed: int, humanize_ms: float, configs: list[dict]) -> dict:
    audio, reference, meta = generate(name, seed=seed, humanize_ms=humanize_ms)
    duration = len(audio) / SAMPLE_RATE
    runs = []

    for config in configs:
        result = run_detector(
            audio,
            SAMPLE_RATE,
            bpm_min=config["bpm_min"],
            bpm_max=config["bpm_max"],
            cooldown_ms=config["cooldown_ms"],
            bass_boost=config["bass_boost"],
        )
        runs.append({
            "config": config,
            "detector": {
                "final_bpm": result["final_bpm"],
                "true_bpm": meta["true_bpm"],
                "bpm_error": round(result["final_bpm"] - meta["true_bpm"], 1),
                "lock_on_time_sec": result["lock_on_time_sec"],
                "mean_confidence_after_5s": round(result["mean_confidence_after_5s"], 3),
                "realtime_ratio": round(result["realtime_ratio"], 1),
                "frames_processed": result["frames_processed"],
            },
            "streams": {
                stream: score_stream(reference, times, duration)
                for stream, times in result["streams"].items()
            },
            "bpm_trace": result["bpm_trace"],
        })

    return {
        "scenario": name,
        "meta": meta,
        "duration_sec": round(duration, 2),
        "reference_beat_count": int(len(reference)),
        "runs": runs,
    }


HEADER = (
    f"{'config':<16}{'stream':<11}{'F':>7}{'Cemgil':>8}{'CMLc':>7}{'CMLt':>7}"
    f"{'AMLc':>7}{'AMLt':>7}{'InfGain':>9}{'FP/min':>8}{'Miss/min':>10}"
    f"{'err_ms':>8}{'BPM':>8}{'lock_s':>8}{'conf':>7}"
)


def print_scenario_table(result: dict) -> None:
    meta = result["meta"]
    print()
    print(f"=== {result['scenario']} — {meta['description']}")
    print(f"    true BPM {meta['true_bpm']}, {result['duration_sec']}s, "
          f"{result['reference_beat_count']} reference beats, "
          f"humanize {meta['humanize_ms']} ms, seed {meta['seed']}")
    print(HEADER)
    print("-" * len(HEADER))
    for run in result["runs"]:
        cfg = run["config"]
        det = run["detector"]
        lock = det["lock_on_time_sec"]
        lock_str = f"{lock:.1f}" if lock is not None else "--"
        for stream in ("raw", "metric", "predicted"):
            s = run["streams"][stream]
            err = s["mean_signed_error_ms"]
            err_str = f"{err:.1f}" if err is not None else "--"
            print(
                f"{cfg['name']:<16}{stream:<11}"
                f"{s['f_measure']:>7.3f}{s['cemgil']:>8.3f}"
                f"{s['cmlc']:>7.3f}{s['cmlt']:>7.3f}{s['amlc']:>7.3f}{s['amlt']:>7.3f}"
                f"{s['information_gain']:>9.3f}"
                f"{s['false_positives_per_min']:>8.1f}{s['misses_per_min']:>10.1f}"
                f"{err_str:>8}{det['final_bpm']:>8.1f}{lock_str:>8}"
                f"{det['mean_confidence_after_5s']:>7.3f}"
            )


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 92)
    print("SUMMARY — metric stream (is_beat and is_metric_beat)")
    print("=" * 92)
    head = (f"{'scenario':<20}{'config':<16}{'F':>7}{'CMLc':>7}{'AMLc':>7}"
            f"{'AMLc-CMLc':>11}{'FP/min':>8}{'BPM':>8}{'true':>7}")
    print(head)
    print("-" * len(head))
    for result in results:
        for run in result["runs"]:
            s = run["streams"]["metric"]
            print(
                f"{result['scenario']:<20}{run['config']['name']:<16}"
                f"{s['f_measure']:>7.3f}{s['cmlc']:>7.3f}{s['amlc']:>7.3f}"
                f"{s['amlc'] - s['cmlc']:>11.3f}"
                f"{s['false_positives_per_min']:>8.1f}"
                f"{run['detector']['final_bpm']:>8.1f}"
                f"{result['meta']['true_bpm']:>7.1f}"
            )


def build_configs(args, preset_name: str) -> list[dict]:
    configs = [{
        "name": "default" if args.bpm_range is None and args.cooldown_ms is None else "custom",
        "bpm_min": args.bpm_range[0] if args.bpm_range else DEFAULT_BPM_MIN,
        "bpm_max": args.bpm_range[1] if args.bpm_range else DEFAULT_BPM_MAX,
        "cooldown_ms": args.cooldown_ms if args.cooldown_ms is not None else DEFAULT_COOLDOWN_MS,
        "bass_boost": DEFAULT_BASS_BOOST,
    }]

    if not args.only_default and preset_name in PRESETS:
        preset = PRESETS[preset_name]
        configs.append({
            "name": f"preset:{preset.name}",
            "bpm_min": preset.bpm_min,
            "bpm_max": preset.bpm_max,
            "cooldown_ms": preset.beat_cooldown_ms,
            "bass_boost": preset.bass_boost,
        })

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic ground-truth beat benchmark")
    parser.add_argument("--scenario", default=None, help="Scenario name (default: all)")
    parser.add_argument("--tag", default="baseline", help="Tag for output JSON files")
    parser.add_argument("--bpm-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                        default=None, help="BPM range for the first pass (default: 80 180)")
    parser.add_argument("--cooldown-ms", type=float, default=None,
                        help="Beat cooldown for the first pass in ms (default: 300)")
    parser.add_argument("--only-default", action="store_true",
                        help="Skip the genre-preset pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--humanize-ms", type=float, default=0.0)
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: tools/benchmark_results/synth)")
    args = parser.parse_args()

    names = [args.scenario] if args.scenario else list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            print(f"Error: unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
            sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).parent / "benchmark_results" / "synth"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in names:
        _, _, meta = generate(name, seed=args.seed, humanize_ms=args.humanize_ms)
        configs = build_configs(args, meta["preset"])
        result = run_scenario(name, args.seed, args.humanize_ms, configs)
        result["tag"] = args.tag
        print_scenario_table(result)
        results.append(result)

        out_path = out_dir / f"{name}_{args.tag}.json"
        with open(out_path, "w") as handle:
            json.dump(result, handle, indent=2)

    print_summary(results)
    print(f"\nResults written to: {out_dir}")


if __name__ == "__main__":
    main()
