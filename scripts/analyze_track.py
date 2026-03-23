#!/usr/bin/env python3
"""Offline audio analysis — run the real pipeline on an audio file.

Usage:
    uv run python scripts/analyze_track.py path/to/song.wav
    uv run python scripts/analyze_track.py path/to/song.flac --csv output.csv

Outputs a CSV with per-frame metrics including the new drop detection
diagnostics: drop_score, bass_exertion, rms_exertion, adaptive_threshold,
energy_slope, and state machine transitions.

Supports: wav, flac, ogg (via soundfile). For mp3: convert first with ffmpeg:
    ffmpeg -i song.mp3 song.wav
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hue_visualizer.audio.analyzer import AudioAnalyzer  # noqa: E402
from hue_visualizer.audio.beat_detector import BeatDetector  # noqa: E402
from hue_visualizer.audio.section_detector import Section, SectionDetector  # noqa: E402


def analyze_track(
    audio_path: str,
    csv_path: str | None = None,
    start_time: float = 0.0,
    sample_rate: int = 44100,
    buffer_size: int = 1024,
    fft_size: int = 2048,
) -> None:
    """Analyze an audio file through the full pipeline and output diagnostics."""

    # Load audio
    print(f"Loading {audio_path}...")
    data, file_sr = sf.read(audio_path, dtype="float32")

    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed (simple decimation — good enough for analysis)
    if file_sr != sample_rate:
        print(f"Resampling {file_sr} -> {sample_rate} Hz...")
        ratio = sample_rate / file_sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len).astype(int)
        data = data[indices]

    # Skip to start time if specified
    if start_time > 0:
        skip_samples = int(start_time * sample_rate)
        if skip_samples < len(data):
            data = data[skip_samples:]
            print(f"Starting from {start_time:.1f}s (skipped {skip_samples} samples)")
        else:
            print(f"Start time {start_time:.1f}s exceeds duration, nothing to analyze")
            return

    duration = len(data) / sample_rate
    print(f"Duration: {duration:.1f}s, {len(data)} samples (from {start_time:.1f}s)")

    # Initialize pipeline
    analyzer = AudioAnalyzer(
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_size=buffer_size,
    )
    beat_detector = BeatDetector(
        sample_rate=sample_rate,
        hop_size=buffer_size,
    )
    section_detector = SectionDetector(
        sample_rate_hz=float(sample_rate) / buffer_size,
    )

    # Process frames
    frames_per_sec = sample_rate / buffer_size
    total_frames = len(data) // buffer_size
    print(f"Processing {total_frames} frames ({frames_per_sec:.1f} fps)...")

    rows = []
    prev_section = Section.UNKNOWN

    for i in range(total_frames):
        start = i * buffer_size
        frame = data[start : start + buffer_size]
        timestamp = start / sample_rate + start_time

        # Run pipeline
        features = analyzer.analyze(frame)
        beat_info = beat_detector.detect(features, timestamp=timestamp)
        section_info = section_detector.update(
            bass_energy=features.bass_energy,
            rms=features.rms,
            centroid=features.spectral_centroid,
            is_beat=beat_info.is_beat,
            bpm=beat_info.bpm,
            now=timestamp,
            rms_raw=features.rms_raw,
            spectral_flux=features.spectral_flux,
            spectral_flatness=features.spectral_flatness,
            band_energies=features.band_energies_unnorm,
        )

        # Section transition marker
        transition = ""
        if section_info.section != prev_section:
            transition = f"{prev_section.value}->{section_info.section.value}"
            prev_section = section_info.section

        rows.append({
            "time": f"{timestamp:.3f}",
            "section": section_info.section.value,
            "confidence": f"{section_info.confidence:.3f}",
            "intensity": f"{section_info.intensity:.3f}",
            "drop_score": f"{section_info.drop_score:.4f}",
            "bass_exertion": f"{section_info.bass_exertion:.4f}",
            "rms_exertion": f"{section_info.rms_exertion:.4f}",
            "adaptive_threshold": f"{section_info.adaptive_threshold:.4f}",
            "energy_slope": f"{section_detector._energy_slope_ema:.6f}",
            "bass": f"{features.bass_energy:.4f}",
            "rms": f"{features.rms:.4f}",
            "rms_raw": f"{features.rms_raw:.6f}",
            "centroid": f"{features.spectral_centroid:.1f}",
            "flux": f"{features.spectral_flux:.2f}",
            "flatness": f"{features.spectral_flatness:.4f}",
            "beat": "1" if beat_info.is_beat else "",
            "bpm": f"{beat_info.bpm:.1f}",
            "transition": transition,
        })

        # Print transitions to console in real-time
        if transition:
            print(
                f"  [{timestamp:7.2f}s] {transition:25s}  "
                f"drop_score={section_info.drop_score:.3f}  "
                f"threshold={section_info.adaptive_threshold:.3f}  "
                f"bass_ex={section_info.bass_exertion:.2f}  "
                f"rms_ex={section_info.rms_exertion:.2f}  "
                f"rms_raw={features.rms_raw:.5f}  "
                f"bpm={beat_info.bpm:.0f}"
            )

    # Write CSV
    if csv_path is None:
        csv_path = str(Path(audio_path).with_suffix(".analysis.csv"))

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")
    else:
        print("\nNo frames processed!")
        return

    # Summary
    sections: dict[str, int] = {}
    for row in rows:
        s = row["section"]
        sections[s] = sections.get(s, 0) + 1
    print("\nSection summary:")
    for s, count in sorted(sections.items()):
        pct = count / len(rows) * 100
        print(f"  {s:12s}: {count:5d} frames ({pct:.1f}%)")

    transitions = [r for r in rows if r["transition"]]
    print(f"\n{len(transitions)} section transitions:")
    for t in transitions:
        print(f"  [{t['time']:>8s}s] {t['transition']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze audio track for section detection diagnostics"
    )
    parser.add_argument("audio", help="Path to audio file (wav, flac, ogg)")
    parser.add_argument("--csv", help="Output CSV path (default: <audio>.analysis.csv)")
    parser.add_argument(
        "--start", type=float, default=0.0,
        help="Start time in seconds (skip audio before this)",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--buffer-size", type=int, default=1024)
    parser.add_argument("--fft-size", type=int, default=2048)
    args = parser.parse_args()

    analyze_track(
        args.audio,
        csv_path=args.csv,
        start_time=args.start,
        sample_rate=args.sample_rate,
        buffer_size=args.buffer_size,
        fft_size=args.fft_size,
    )
