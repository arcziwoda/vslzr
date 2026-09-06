# Real-audio baseline (2026-09-06)

First measurement of the beat path on real tracks instead of synthetic ground truth.
Audio and annotations live in `audio/` (gitignored); result JSONs are in
`tools/benchmark_results/real/`.

## Pipeline

1. `audio/tracks/<slug>.{flac,mp3}` — the user's tracks (6, 3-8 min each, 44.1 kHz stereo).
2. `.venv-gt/bin/python tools/annotate_tracks.py` — pseudo ground truth from two offline
   trackers in a separate environment (torch, `beat_this` 1.1.0, `madmom` git master):
   reference beats = Beat This! `final0` with madmom DBN post-processing, cross-check =
   madmom RNN + DBN. `agreement` = fraction of reference beats with a madmom beat within
   70 ms, per track and per 30 s window. Without the DBN post-processing Beat This! flips
   metrical level mid-track (Prodigy came out at 187 BPM with 809 ms IBI std).
3. `uv run python tools/real_benchmark.py --tag <name>` — detector streams
   (raw / metric / predicted) and end-to-end flashes (engine, metric filter on, 60 ms
   compensation) scored with mir_eval against the reference, plus an F timeline per 30 s.
   The genre preset is chosen from the annotation tempo (techno for 118-150, dnb for
   155-185).

## Ground truth quality

| track | BPM | beats | agreement | note |
|---|---|---|---|---|
| bours_blind_pick | 142.9 | 913 | 0.96 | |
| dimension_devotion | 171.4 | 433 | 0.99 | DnB; trackers annotate 88 BPM in intro/outro, 176 in drops |
| discobitch_cest_beau_la_bourgeoisie | 130.4 | 740 | 0.99 | |
| dj_gigola_siente_el_ritmo | 142.9 | 709 | 0.95 | |
| fred_again_turn_on_the_lights_again | 130.4 | 587 | 0.70 | trackers disagree on phase in 4 of 9 windows; treat as unreliable |
| prodigy_firestarter_empirion_mix | 142.9 | 1078 | 0.90 | first 60 s agreement 0.36-0.44 |

## Baseline (code at commit 356a60b, tag `step14`)

| track | det BPM | metric F | pred F | engine F | engine FP/min |
|---|---|---|---|---|---|
| bours_blind_pick | 143.4 | 0.516 | 0.776 | 0.293 | 60.5 |
| dimension_devotion | 174.2 | 0.792 | 0.827 | 0.718 | 41.4 |
| discobitch | 128.3 | 0.652 | 0.632 | 0.712 | 21.4 |
| dj_gigola | 145.8 | 0.360 | 0.202 | 0.118 | 93.3 |
| fred_again | 131.9 | 0.232 | 0.358 | 0.159 | 95.6 |
| prodigy | 139.3 | 0.585 | 0.569 | 0.563 | 54.6 |

Synthetic scenarios give 0.996 on the same code. The tempo is right on every track; the
phase is not.

## Attribution

Per-window diagnostics (scratch scripts, not in the repo) on the two worst tracks:

- Strong onsets (`strong_val >= 0.4` against the 8-minute peak) fire 8-12 times per minute
  against 143 beats per minute. Everything gated on them (agent scoring weight, evidence
  window, seeding, capture) starves: on dj_gigola no agent exists for the first 90 s.
- Onset recall at reference beats is 0.4-0.7; onset precision 0.2-0.5 (200-250 onsets/min).
- The onset detection function does not separate beats from the midpoints between them:
  median `onset_val` 0.39 at beats vs 0.42 at midpoints on bours_blind_pick. Offbeat hats
  and offbeat bass produce as much bass increase + SuperFlux as the kick.

Separability of candidate hand-crafted features, AUC of the feature maximum within 70 ms of
a reference beat vs at the midpoint between beats:

| feature | bours | devotion | discobitch | gigola | fred | prodigy |
|---|---|---|---|---|---|---|
| SuperFlux full band (current) | 0.36 | 0.82 | 0.50 | 0.71 | 0.27 | 0.53 |
| SuperFlux < 250 Hz | 0.85 | 0.60 | 0.84 | 0.50 | 0.57 | 0.62 |
| SuperFlux 250-2000 Hz | 0.53 | 0.67 | 0.76 | 0.78 | 0.60 | 0.58 |
| energy increase 60-250 Hz | 0.90 | 0.60 | 0.88 | 0.75 | 0.53 | 0.87 |
| current mix (bass increase + SuperFlux) | 0.60 | 0.75 | 0.76 | 0.73 | 0.37 | 0.71 |
| madmom causal RNN beat activation | 0.99 | 0.90 | 0.99 | 0.95 | 0.69 | 0.93 |

No hand-crafted band works across tracks (the best band differs per track and tops out
around 0.85-0.90); the learned activation does.

## Existing PLL on the learned activation

The causal (online, 100 fps, unidirectional LSTM) madmom activation was fed into the
unchanged `BeatDetector` as its onset function (bass increase and SuperFlux both replaced
by the activation), detector level only:

| track | metric F | pred F | madmom online DBN F (reference tracker) |
|---|---|---|---|
| bours_blind_pick | 0.955 | 0.956 | 0.976 |
| dimension_devotion | 0.789 | 0.849 | 0.875 |
| discobitch | 0.963 | 0.987 | 0.963 |
| dj_gigola | 0.875 | 0.942 | 0.865 |
| fred_again | 0.678 | 0.781 | 0.684 |
| prodigy | 0.793 | 0.795 | 0.820 |

The tracker is not the bottleneck; the onset function is. Devotion is capped by the
reference switching metrical level; fred_again by the reference itself.

## Options (decision pending)

1. Port madmom's online beat RNN to numpy inside the app (feature pipeline: three STFT
   sizes at 100 fps, log-filtered spectrogram + first differences; model: small LSTM
   stack; weights are .pkl files in the madmom package). No new runtime dependency.
   The madmom **model weights are CC BY-NC-SA 4.0** (non-commercial); the code is BSD.
2. Train a small causal model on EDM with Beat This! pseudo-labels (licence-clean, needs
   data and time).
3. Stay hand-crafted with per-track adaptive band selection; ceiling ~0.85-0.90 AUC,
   clearly below option 1.
