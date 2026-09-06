"""Pseudo ground truth for real tracks from offline beat trackers.

Runs in the separate tool environment (torch, beat_this, madmom), NOT the
project venv:

    uv venv .venv-gt --python 3.12
    uv pip install --python .venv-gt/bin/python beat_this soundfile \
        "madmom @ git+https://github.com/CPJKU/madmom"
    .venv-gt/bin/python tools/annotate_tracks.py [--track slug] [--force]

For every audio file in audio/tracks/ this writes audio/annotations/<slug>.json:

    beats        Beat This! (final0 checkpoint) with madmom DBN post-processing
    downbeats    same model
    madmom_beats madmom RNNBeatProcessor + DBNBeatTrackingProcessor (cross-check)
    agreement    fraction of reference beats with a madmom beat within 70 ms
    bpm          median inter-beat interval of the reference beats
    windows      per 30 s: reference beat count, local BPM, agreement — to spot
                 sections where the trackers themselves disagree (unreliable GT)

Beats where the two trackers disagree are kept (the reference is Beat This!);
the agreement numbers tell the consumer how much to trust a track or a window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TRACK_DIR = ROOT / "audio" / "tracks"
ANNOTATION_DIR = ROOT / "audio" / "annotations"
AUDIO_EXT = {".flac", ".wav", ".mp3", ".aiff", ".aif", ".ogg"}
TOLERANCE = 0.070
WINDOW = 30.0


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1).astype(np.float32), int(sr)


def run_beat_this(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import File2Beats

    file2beats = File2Beats(checkpoint_path="final0", device="cpu", dbn=True)
    beats, downbeats = file2beats(str(path))
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def run_madmom(audio: np.ndarray, sr: int) -> np.ndarray:
    from madmom.audio.signal import Signal
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

    signal = Signal(audio, sample_rate=sr)
    activations = RNNBeatProcessor()(signal)
    tracker = DBNBeatTrackingProcessor(fps=100, min_bpm=55.0, max_bpm=215.0)
    return np.asarray(tracker(activations), dtype=float)


def agreement(reference: np.ndarray, other: np.ndarray) -> float:
    if len(reference) == 0 or len(other) == 0:
        return 0.0
    idx = np.searchsorted(other, reference)
    hits = 0
    for r, k in zip(reference, idx):
        candidates = [other[j] for j in (k - 1, k) if 0 <= j < len(other)]
        if candidates and min(abs(r - c) for c in candidates) <= TOLERANCE:
            hits += 1
    return hits / len(reference)


def windows(reference: np.ndarray, other: np.ndarray, duration: float) -> list[dict]:
    out = []
    t = 0.0
    while t < duration:
        t1 = min(t + WINDOW, duration)
        ref = reference[(reference >= t) & (reference < t1)]
        oth = other[(other >= t) & (other < t1)]
        local_bpm = float(60.0 / np.median(np.diff(ref))) if len(ref) > 2 else None
        out.append({
            "start": round(t, 1),
            "beats": int(len(ref)),
            "bpm": None if local_bpm is None else round(local_bpm, 1),
            "agreement": round(agreement(ref, oth), 3) if len(ref) else None,
        })
        t = t1
    return out


def annotate(path: Path) -> dict:
    audio, sr = load_mono(path)
    duration = len(audio) / sr
    beats, downbeats = run_beat_this(path)
    madmom_beats = run_madmom(audio, sr)
    ibi = np.diff(beats)
    return {
        "file": path.name,
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "source": "beat_this final0 + madmom DBN; cross-check madmom RNN+DBN",
        "bpm": round(float(60.0 / np.median(ibi)), 2) if len(ibi) else 0.0,
        "ibi_std_ms": round(float(np.std(ibi) * 1000.0), 1) if len(ibi) else 0.0,
        "agreement": round(agreement(beats, madmom_beats), 3),
        "agreement_madmom_vs_ref": round(agreement(madmom_beats, beats), 3),
        "beats": [round(float(b), 4) for b in beats],
        "downbeats": [round(float(b), 4) for b in downbeats],
        "madmom_beats": [round(float(b), 4) for b in madmom_beats],
        "windows": windows(beats, madmom_beats, duration),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate real tracks with offline beat trackers")
    parser.add_argument("--track", default=None, help="Slug (file stem) to annotate")
    parser.add_argument("--force", action="store_true", help="Re-annotate existing")
    args = parser.parse_args()

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in TRACK_DIR.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if args.track:
        paths = [p for p in paths if p.stem == args.track]
    for path in paths:
        out = ANNOTATION_DIR / f"{path.stem}.json"
        if out.exists() and not args.force:
            print(f"skip {path.stem} (exists)")
            continue
        print(f"annotating {path.name} ...", flush=True)
        result = annotate(path)
        with open(out, "w") as f:
            json.dump(result, f)
        print(f"  {result['bpm']:.1f} BPM, {len(result['beats'])} beats, "
              f"IBI std {result['ibi_std_ms']:.0f} ms, agreement {result['agreement']:.2f}")


if __name__ == "__main__":
    main()
