"""Benchmark beat detector against an audio file.

Usage:
    uv run python tools/benchmark_beats.py <audio_file> [--output results.json]

Processes the file offline (simulating real-time frame-by-frame) through
AudioAnalyzer → BeatDetector, records all beat events, BPM trajectory,
confidence, and per-band onsets. Saves results as JSON for comparison.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hue_visualizer.audio.analyzer import AudioAnalyzer
from hue_visualizer.audio.beat_detector import BeatDetector


def process_file(audio_path: str) -> dict:
    """Process audio file frame-by-frame through the beat detection pipeline."""
    # Load audio
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # Mono mix

    print(f"Loaded: {audio_path}")
    print(f"  Sample rate: {sr} Hz, Duration: {len(data)/sr:.1f}s, Samples: {len(data)}")

    # Pipeline setup (matching production config)
    hop_size = 1024
    fft_size = 2048
    analyzer = AudioAnalyzer(sample_rate=sr, fft_size=fft_size, hop_size=hop_size)
    detector = BeatDetector(sample_rate=sr, hop_size=hop_size)

    frame_dur = hop_size / sr
    n_frames = (len(data) - fft_size) // hop_size + 1

    # Results storage
    beats = []           # (time, strength)
    bpm_trace = []       # (time, bpm, confidence)
    kick_onsets = []     # (time, energy)
    snare_onsets = []    # (time, energy)
    hihat_onsets = []    # (time, energy)

    # Process frame by frame
    t_start = time.perf_counter()
    for i in range(n_frames):
        offset = i * hop_size
        frame = data[offset:offset + hop_size]
        if len(frame) < hop_size:
            break

        t = offset / sr  # Current time in seconds
        # Simulate monotonic timestamp (starting from 0)
        sim_time = t

        features = analyzer.analyze(frame)
        beat_info = detector.detect(features, timestamp=sim_time)

        # Record BPM every ~0.25s
        if i % max(1, int(0.25 / frame_dur)) == 0:
            bpm_trace.append({
                "time": round(t, 3),
                "bpm": round(beat_info.bpm, 1),
                "confidence": round(beat_info.bpm_confidence, 3),
            })

        if beat_info.is_beat:
            beats.append({
                "time": round(t, 4),
                "strength": round(beat_info.beat_strength, 3),
            })

        if beat_info.kick_onset:
            kick_onsets.append({"time": round(t, 4), "energy": round(beat_info.kick_energy, 3)})
        if beat_info.snare_onset:
            snare_onsets.append({"time": round(t, 4), "energy": round(beat_info.snare_energy, 3)})
        if beat_info.hihat_onset:
            hihat_onsets.append({"time": round(t, 4), "energy": round(beat_info.hihat_energy, 3)})

    elapsed = time.perf_counter() - t_start
    realtime_ratio = (len(data) / sr) / elapsed

    # Compute summary stats
    beat_times = [b["time"] for b in beats]
    ibis = np.diff(beat_times) if len(beat_times) > 1 else np.array([])

    # BPM from median IBI
    median_ibi = float(np.median(ibis)) if len(ibis) > 0 else 0
    ibi_bpm = 60.0 / median_ibi if median_ibi > 0 else 0

    # IBI consistency (lower = more consistent)
    ibi_std = float(np.std(ibis)) if len(ibis) > 0 else 0
    ibi_cv = ibi_std / median_ibi if median_ibi > 0 else 0  # Coefficient of variation

    # Final reported BPM (from detector)
    final_bpms = [t["bpm"] for t in bpm_trace if t["bpm"] > 0]
    final_bpm = final_bpms[-1] if final_bpms else 0

    # BPM lock-on time (first time confidence > 0.6)
    lock_on_time = None
    for t in bpm_trace:
        if t["confidence"] > 0.6 and t["bpm"] > 0:
            lock_on_time = t["time"]
            break

    # Confidence stats
    confidences = [t["confidence"] for t in bpm_trace if t["time"] > 5.0]  # Skip warmup
    avg_confidence = float(np.mean(confidences)) if confidences else 0

    summary = {
        "total_beats": len(beats),
        "duration_sec": round(len(data) / sr, 2),
        "final_bpm": round(final_bpm, 1),
        "ibi_median_bpm": round(ibi_bpm, 1),
        "ibi_median_sec": round(median_ibi, 4) if median_ibi > 0 else None,
        "ibi_std_sec": round(ibi_std, 4) if len(ibis) > 0 else None,
        "ibi_cv": round(ibi_cv, 4) if ibi_cv > 0 else None,
        "lock_on_time_sec": round(lock_on_time, 2) if lock_on_time else None,
        "avg_confidence": round(avg_confidence, 3),
        "kick_onsets_count": len(kick_onsets),
        "snare_onsets_count": len(snare_onsets),
        "hihat_onsets_count": len(hihat_onsets),
        "processing_realtime_ratio": round(realtime_ratio, 1),
        "frames_processed": n_frames,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"BEAT DETECTION BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Duration:         {summary['duration_sec']}s")
    print(f"  Total beats:      {summary['total_beats']}")
    print(f"  Final BPM:        {summary['final_bpm']}")
    print(f"  IBI median BPM:   {summary['ibi_median_bpm']}")
    print(f"  IBI std:          {summary['ibi_std_sec']}s")
    print(f"  IBI CV:           {summary['ibi_cv']} (lower=more consistent)")
    print(f"  BPM lock-on:      {summary['lock_on_time_sec']}s")
    print(f"  Avg confidence:   {summary['avg_confidence']}")
    print(f"  Kick onsets:      {summary['kick_onsets_count']}")
    print(f"  Snare onsets:     {summary['snare_onsets_count']}")
    print(f"  Hi-hat onsets:    {summary['hihat_onsets_count']}")
    print(f"  Realtime ratio:   {summary['processing_realtime_ratio']}x")
    print(f"{'='*60}")

    # IBI histogram (text-based)
    if len(ibis) > 5:
        print(f"\nIBI Distribution (inter-beat intervals):")
        hist, edges = np.histogram(ibis, bins=20, range=(0, max(1.5, np.percentile(ibis, 99))))
        max_count = max(hist) if max(hist) > 0 else 1
        for count, left, right in zip(hist, edges[:-1], edges[1:]):
            bar = "#" * int(40 * count / max_count)
            bpm_left = 60.0 / right if right > 0 else 0
            bpm_right = 60.0 / left if left > 0 else 999
            print(f"  {left:.3f}-{right:.3f}s ({bpm_left:.0f}-{bpm_right:.0f} BPM): {bar} ({count})")

    return {
        "file": str(audio_path),
        "summary": summary,
        "beats": beats,
        "bpm_trace": bpm_trace,
        "kick_onsets": kick_onsets,
        "snare_onsets": snare_onsets,
        "hihat_onsets": hihat_onsets,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark beat detector on audio file")
    parser.add_argument("audio_file", help="Path to audio file (FLAC, WAV, etc.)")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: tools/benchmark_results/<filename>_baseline.json)",
    )
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Tag for the output file (default: baseline)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}")
        sys.exit(1)

    results = process_file(str(audio_path))

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(__file__).parent / "benchmark_results"
        out_dir.mkdir(exist_ok=True)
        stem = audio_path.stem.replace(" ", "_")
        out_path = out_dir / f"{stem}_{args.tag}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
