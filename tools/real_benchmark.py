"""Beat benchmark on real tracks against offline-tracker annotations.

Ground truth comes from tools/annotate_tracks.py (Beat This! beats, optionally
cross-checked with madmom) stored as audio/annotations/<slug>.json. Audio and
annotations live under audio/ (gitignored); only the result JSONs are kept.

Usage:
    uv run python tools/real_benchmark.py [--track slug] [--tag name] [--no-engine]
                                          [--preset name] [--window 30]

Per track and configuration this reports the detector streams (raw / metric /
predicted, as in synth_benchmark.py) and the end-to-end flashes (engine metric
config, as in engine_benchmark.py), plus an F-measure timeline in fixed windows
so a failure can be located in the track.

The genre preset defaults to the one whose BPM range contains the annotation
tempo (first match in PRESET_ORDER); --preset overrides it. Tracks whose
annotation tempo is outside every range fall back to the detector defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hue_visualizer.visualizer.presets import PRESETS  # noqa: E402
from engine_benchmark import LATENCY_COMP_MS, run_engine  # noqa: E402
from synth_benchmark import (  # noqa: E402
    COLD_START_SEC,
    DEFAULT_BASS_BOOST,
    DEFAULT_BPM_MAX,
    DEFAULT_BPM_MIN,
    DEFAULT_COOLDOWN_MS,
    run_detector,
    score_stream,
)

TRACK_DIR = ROOT / "audio" / "tracks"
ANNOTATION_DIR = ROOT / "audio" / "annotations"
RESULT_DIR = ROOT / "tools" / "benchmark_results" / "real"
PRESET_ORDER = ("techno", "house", "dnb", "trap")
MATCH_TOLERANCE_SEC = 0.070


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(sr)


def load_annotation(slug: str) -> dict:
    path = ANNOTATION_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python tools/annotate_tracks.py` first"
        )
    with open(path) as f:
        return json.load(f)


def choose_preset(bpm: float) -> str | None:
    for name in PRESET_ORDER:
        p = PRESETS[name]
        if p.bpm_min <= bpm <= p.bpm_max:
            return name
    return None


def timeline(reference: np.ndarray, estimated: np.ndarray, duration: float,
             window: float) -> list[dict]:
    """F-measure per fixed window, so failures can be located in the track."""
    out = []
    t = COLD_START_SEC
    while t < duration:
        t1 = min(t + window, duration)
        ref = reference[(reference >= t) & (reference < t1)]
        est = estimated[(estimated >= t) & (estimated < t1)]
        if len(ref) == 0:
            out.append({"start": round(t, 1), "f": None, "ref": 0, "est": int(len(est))})
        else:
            hits = 0
            used = np.zeros(len(est), dtype=bool)
            for r in ref:
                d = np.abs(est - r)
                d[used] = np.inf
                if len(d) and d.min() <= MATCH_TOLERANCE_SEC:
                    used[int(np.argmin(d))] = True
                    hits += 1
            precision = hits / len(est) if len(est) else 0.0
            recall = hits / len(ref)
            f = 2 * precision * recall / (precision + recall) if hits else 0.0
            out.append({"start": round(t, 1), "f": round(f, 3),
                        "ref": int(len(ref)), "est": int(len(est))})
        t = t1
    return out


def run_track(slug: str, preset_override: str | None, with_engine: bool,
              window: float, onset_source: str = "rnn") -> dict:
    annotation = load_annotation(slug)
    audio_path = TRACK_DIR / annotation["file"]
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    reference = np.asarray(annotation["beats"], dtype=float)
    reference = reference[reference < duration]
    gt_bpm = float(annotation["bpm"])

    preset_name = preset_override or choose_preset(gt_bpm)
    if preset_name:
        p = PRESETS[preset_name]
        config = {"name": f"preset:{preset_name}", "bpm_min": p.bpm_min, "bpm_max": p.bpm_max,
                  "cooldown_ms": p.beat_cooldown_ms, "bass_boost": p.bass_boost}
    else:
        config = {"name": "default", "bpm_min": DEFAULT_BPM_MIN, "bpm_max": DEFAULT_BPM_MAX,
                  "cooldown_ms": DEFAULT_COOLDOWN_MS, "bass_boost": DEFAULT_BASS_BOOST}

    det = run_detector(audio, sr, config["bpm_min"], config["bpm_max"],
                       config["cooldown_ms"], config["bass_boost"], onset_source=onset_source)
    config["onset_source"] = onset_source
    streams = {name: score_stream(reference, times, duration)
               for name, times in det["streams"].items()}
    streams["metric"]["timeline"] = timeline(reference, det["streams"]["metric"], duration, window)
    streams["predicted"]["timeline"] = timeline(
        reference, det["streams"]["predicted"], duration, window)

    result = {
        "track": slug,
        "file": annotation["file"],
        "duration_sec": round(duration, 1),
        "sample_rate": sr,
        "annotation": {
            "bpm": gt_bpm,
            "beats": int(len(reference)),
            "source": annotation.get("source"),
            "agreement": annotation.get("agreement"),
        },
        "config": config,
        "detector": {
            "final_bpm": det["final_bpm"],
            "lock_on_time_sec": det["lock_on_time_sec"],
            "mean_confidence_after_5s": round(det["mean_confidence_after_5s"], 3),
            "realtime_ratio": round(det["realtime_ratio"], 1),
        },
        "streams": streams,
        "bpm_trace": det["bpm_trace"],
    }

    if with_engine and preset_name:
        out = run_engine(audio, preset_name, metric_filter=True, onset_source=onset_source)
        intended = np.asarray(out["flashes"]) + LATENCY_COMP_MS / 1000.0
        engine = score_stream(reference, intended, duration)
        engine["predictive_active_fraction"] = out["predictive_active_fraction"]
        engine["flash_count"] = int(len([t for t in out["flashes"] if t >= COLD_START_SEC]))
        engine["timeline"] = timeline(reference, intended, duration, window)
        result["engine_metric"] = engine
    return result


def print_result(r: dict) -> None:
    a = r["annotation"]
    agreement = a.get("agreement")
    agreement_txt = f" agree={agreement:.2f}" if isinstance(agreement, (int, float)) else ""
    print(f"\n{r['track']}  {r['duration_sec']:.0f} s  GT {a['bpm']:.1f} BPM "
          f"({a['beats']} beats{agreement_txt})  config={r['config']['name']}  "
          f"detector BPM {r['detector']['final_bpm']:.1f}  "
          f"conf {r['detector']['mean_confidence_after_5s']:.2f}  "
          f"lock {r['detector']['lock_on_time_sec']}")
    print(f"  {'stream':<12}{'F':>7}{'CMLc':>7}{'CMLt':>7}{'AMLc':>7}{'FP/min':>8}{'Miss/min':>10}"
          f"{'err_ms':>8}")
    rows = list(r["streams"].items())
    if "engine_metric" in r:
        rows.append(("engine", r["engine_metric"]))
    for name, s in rows:
        err = s["mean_signed_error_ms"]
        print(f"  {name:<12}{s['f_measure']:>7.3f}{s['cmlc']:>7.3f}{s['cmlt']:>7.3f}"
              f"{s['amlc']:>7.3f}{s['false_positives_per_min']:>8.1f}"
              f"{s['misses_per_min']:>10.1f}{(err if err is not None else float('nan')):>8.1f}")
    key = "engine_metric" if "engine_metric" in r else "metric"
    source = r[key] if key == "engine_metric" else r["streams"]["metric"]
    cells = []
    for w in source["timeline"]:
        cells.append("  ---" if w["f"] is None else f"{w['f']:5.2f}")
    print(f"  {key} F per window:")
    for i in range(0, len(cells), 12):
        start = source["timeline"][i]["start"]
        print(f"    {start:6.0f}s |" + "".join(cells[i:i + 12]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Beat benchmark on real annotated tracks")
    parser.add_argument("--track", default=None, help="Track slug (default: all annotated)")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--preset", default=None, help="Force a genre preset")
    parser.add_argument("--no-engine", action="store_true")
    parser.add_argument("--window", type=float, default=30.0, help="Timeline window (s)")
    parser.add_argument("--onset-source", choices=("rnn", "spectral"), default="rnn")
    args = parser.parse_args()

    if args.track:
        slugs = [args.track]
    else:
        slugs = sorted(p.stem for p in ANNOTATION_DIR.glob("*.json"))
    if not slugs:
        print("No annotations found in", ANNOTATION_DIR)
        sys.exit(1)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for slug in slugs:
        r = run_track(slug, args.preset, not args.no_engine, args.window, args.onset_source)
        print_result(r)
        with open(RESULT_DIR / f"{slug}_{args.tag}.json", "w") as f:
            json.dump(r, f, indent=1)
        summary.append(r)

    print(f"\n{'track':<42}{'GT BPM':>8}{'det BPM':>8}{'metric F':>10}{'pred F':>8}"
          f"{'engine F':>10}{'FP/min':>8}")
    print("-" * 94)
    for r in summary:
        eng = r.get("engine_metric")
        print(f"{r['track']:<42}{r['annotation']['bpm']:>8.1f}{r['detector']['final_bpm']:>8.1f}"
              f"{r['streams']['metric']['f_measure']:>10.3f}"
              f"{r['streams']['predicted']['f_measure']:>8.3f}"
              f"{(eng['f_measure'] if eng else float('nan')):>10.3f}"
              f"{(eng['false_positives_per_min'] if eng else float('nan')):>8.1f}")
    print(f"\nResults written to {RESULT_DIR} (tag {args.tag}); first {COLD_START_SEC:.0f} s skipped.")


if __name__ == "__main__":
    main()
