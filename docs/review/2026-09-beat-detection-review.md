# Beat detection review — September 2026

Scope: the path from audio frame to light trigger:
`audio/analyzer.py` → `audio/beat_detector.py` → `server/app.py` (`AudioPipeline`, `audio_loop`) → `visualizer/engine.py` (`_resolve_beat_trigger`, flash overlays).
Reference spec: `docs/beat_detection_research_2026_03.md`. Failure-mode catalogue: `docs/research/beat-detection-benchmarking-research.md` §4.

Method: full read of the four files, then feature-level simulations against `BeatDetector`
(synthetic `AudioFeatures` with ideal kick pulses at a known BPM). "Confirmed" findings were
reproduced in a simulation; "plausible" findings are from code reading only.

Reported symptom: lights sometimes flash on the beat, sometimes on extra non-beat moments.

## Summary

| ID | Severity | Status | Where | One line |
|---|---|---|---|---|
| F1 | critical | confirmed | `beat_detector.py:_seed_agents_from_autocorrelation` | New PLL agents start at phase 0, never aligned to onsets; correct-tempo agent is pruned, a lucky wrong-tempo agent wins. Reported BPM is wrong on every tempo tested (130→173, 125→163, 174→130, 100→84). |
| F2 | critical | confirmed | `server/app.py:audio_loop` | `is_metric_beat` is not copied into the `BeatInfo` handed to the engine, so the metric-filter toggle is a no-op (filter ON = reactive triggers never fire). |
| F3 | high | confirmed | `beat_detector.py:detect` | `is_beat` is a raw onset (bass OR SuperFlux) with a cooldown. A percussive hit at the "and-a" (0.75 beat) passes cooldown, fires, and then the cooldown suppresses the real kick: every trigger is 100 ms early. |
| F4 | high | confirmed | `beat_detector.py:detect` (auto-cooldown) | Preset cooldown is overwritten by `max(0.30, 0.75·period)`. DnB preset 150 ms becomes 300 ms; ambient 600 ms becomes ~0.36–0.9 s depending on BPM. |
| F5 | high | plausible | `beat_detector.py:detect` | `predicted_next_beat = last_beat_time + pll_period` is anchored to the last raw onset (hats included), not to the PLL phase. Predictive triggers inherit onset jitter and jump on every false onset. |
| F6 | medium | confirmed | `beat_detector.py:_seed_agents_from_autocorrelation` | Prune runs in the same call as seeding: new agents (score 1.0) are killed instantly once any agent scores > 1.25. After warm-up no new tempo hypothesis can ever enter; steady-state scores are ~19. |
| F7 | medium | confirmed | `beat_detector.py` | Output BPM comes from the best agent even when it contradicts the autocorrelation estimate (raw 129 BPM at 0.8 confidence vs agent 174 BPM). No consistency check. |
| F8 | medium | plausible | `engine.py:tick` | `snare_onset` (mid-band energy threshold, 120 ms cooldown) triggers the white flash independently of any beat logic. Bypasses both metric filter and predictive path. |
| F9 | medium | plausible | `beat_detector.py:_check_metric_beat_alignment` | Passthrough while confidence < 0.4, but confidence sits around 0.4 in practice (SNR-based confidence on a p=0.5 compressed autocorrelation is structurally low). The filter would be off half the time even after F2. |
| F10 | medium | plausible | `beat_detector.py:detect` | Time base mixes `time.monotonic()` at processing time (bursty, 50 Hz tick) with frame-duration phase advance. Frames processed in one tick share a timestamp; ±20 ms jitter on every onset time and on the ±50 ms confirmation windows. |
| F11 | low | plausible | `beat_detector.py:detect` | Adaptive threshold `1.55 − variance/0.02·0.30` is clipped at 1.25 for any real music (variance ≫ 0.02); the adaptive term is dead. Same for the per-band thresholds. |
| F12 | low | plausible | `analyzer.py:analyze` | Bass boost clips `band_energies[0:2]` at 1.5; with boost 2.0 anything above 0.75 saturates. In sustained-bass sections `median·1.25 > 1.5` is impossible, so the energy detector goes silent and only SuperFlux remains. |
| F13 | low | plausible | `analyzer.py:analyze` | STFT overlap is correct only when `hop == fft_size/2`; other `buffer_size` values silently produce a zero-padded shorter window. |
| F14 | low | confirmed | `beat_detector.py:_seed_agents_from_autocorrelation` | Recomputes the generalized autocorrelation that `_estimate_bpm_autocorrelation` just computed (same buffer, same padding). Wasted work, and two code paths to keep in sync. |
| F15 | low | plausible | `beat_detector.py:set_bpm_range` | `_prediction_window` maxlen is derived from `bpm_max` at construction and not updated on range change. |

## Details

### F1 — agents are seeded with an arbitrary phase (critical, confirmed)

`_seed_agents_from_autocorrelation` appends `BeatAgent(period=period)`; the dataclass default is
`phase = 0.0`, which means "a beat is happening right now" at whatever frame the periodic
autocorrelation happened to run. Agents only ever move their phase when an onset lands within
±50 ms of their prediction (`_correct_agents_on_beat`). A correct-tempo agent seeded 200 ms off
the grid predicts beats 200 ms off the grid forever, never gets confirmed, accumulates misses
and is pruned. Meanwhile agents at unrelated tempi get coincidental confirmations, score +1,
and the 0.8 kill ratio removes the correct agent.

Trace on an ideal 130 BPM kick train (autocorrelation says 129.2 BPM at 0.8 confidence
throughout):

```
seed@1.16  after=[(129.2, 0.95), (83.4, 0.95), (117.5, 0.95), (92.3, 0.95), (103.4, 0.95)]
seed@2.14  before=[(129.2, 0.9, misses=2), (83.7, 1.9), ...]   after=[(83.7, 1.81), (92.1, 1.81)]
seed@7.50  after=[(171.5, 1.85)]
seed@13.35 after=[(174.3, 4.28)]      # reported BPM 173.9, locked, one agent
```

The surviving 171→174 BPM agent is 4/3 of the true tempo: its every 4th prediction coincides
with every 3rd kick, so it never reaches 8 consecutive misses, and the PI term walks it to
exactly 4/3 × 130.

Results with default range 80–180 over 30 s of ideal kicks:

| true BPM | reported (current) | reported (phase-aligned seeding) |
|---|---|---|
| 130 | 172.9 | 130.1 |
| 125 | 163.2 | 125.1 |
| 174 | 130.0 | 172.4 |
| 100 | 83.7 | 99.4 |

The fix column is a two-line change: after seeding, set
`agent.phase = ((now − last_beat_time) / agent.period) % 1.0` when `last_beat_time > 0`.
IBT seeds children from the parent's current beat position for exactly this reason.

Genre presets narrow the BPM range (techno 118–150), which bounds how wrong the winner can be
but does not fix the mechanism; the winner is still whichever in-range hypothesis had lucky
phase, and its period then drifts under PI correction.

Also affected: `predicted_next_beat`, `bpm` displayed in the UI, `is_metric_beat`, cooldown
(F4), generative breathing rate and palette rotation (both take `beat_info.bpm`).

### F2 — metric filter never reaches the engine (critical, confirmed)

`audio_loop` (`app.py` ~line 407) builds `BeatInfo(is_beat=had_beat, bpm=..., ...)` from latched
flags. `is_metric_beat` is not among them and `AudioPipeline.consume_beat` does not latch it, so
the engine always sees `is_metric_beat=False`. With the toggle ON, `_resolve_beat_trigger` gates
on `False` and reactive flashes stop entirely; with it OFF nothing changes. The A/B comparison
this toggle was added for (commit 1926462) measured nothing.

Fix: latch `is_metric_beat` in `AudioPipeline` alongside `_pending_beat` (any frame in the tick
with `is_beat and is_metric_beat` → `True`) and copy it into `beat_for_engine`. Consider making
`consume_beat` return the full latched `BeatInfo` so this class of bug cannot recur.

### F3 — raw onset + cooldown = the wrong onset wins (high, confirmed)

Simulation: 130 BPM kicks plus a percussive hit at 0.75 of each beat (very common in techno).
All 54 triggers after warm-up land in the 0.70–0.85 phase bin, none at the kick;
mean timing error −103.6 ms. Mechanism: the 0.75 hit passes the 0.75·period cooldown by a hair,
fires, resets `_last_beat_time`, and the real kick 0.25 period later is inside the cooldown.

Hats on the offbeat (0.5) are blocked by cooldown, so this is pattern-dependent, which matches
"sometimes right, sometimes not". With `is_metric_beat` working (F2) the 0.75 hit would be
filtered, but the kick would still be swallowed by the cooldown, so the light would not flash
at all on that beat. The cooldown must not be reset by onsets that fail the metric test, or,
better, the reactive trigger should be "onset within the PLL window", with the cooldown as a
safety net only.

### F4 — auto-cooldown overrides preset cooldown (high, confirmed)

`set_cooldown()` sets `_manual_cooldown_sec` and `auto_cooldown = True`; `detect()` then does
`cooldown_sec = max(0.30, 0.75 · period)` whenever BPM is known. The manual value is never used
again. DnB preset 150 ms → effective 300 ms (beat period 345 ms, kick-snare two-step gap 172 ms).
Ambient 600 ms → 0.75·period. Presets document intent that the code does not honor.

### F5 — prediction anchored to last raw onset (high, plausible)

`predicted_next_beat = _last_beat_time + _pll_period`. The PLL keeps a phase; the prediction
should be `now + (1 − best.phase) · best.period`. As written, every accepted onset (including
the false ones from F3) moves the prediction, `_resolve_beat_trigger` sees a "new prediction"
(>50 ms different), and can fire predictively again within the same beat.

### F6 — new agents pruned in the same call (medium, confirmed)

Seeding appends agents with score 1.0, then pruning removes everything below
`0.8 · best_score`. Steady-state best score is ~19 (+1 per confirmed beat, ×0.95 every 0.5 s),
so any new hypothesis dies the frame it is born. A tempo change (new track, DJ transition) can
only be tracked after the old agent dies by 8 consecutive misses, which coincidental
confirmations keep resetting. IBT: children inherit 0.9 × parent score and are protected for a
few beats.

### F7 — no agreement check between autocorrelation and agents (medium, confirmed)

`_raw_bpm` was 129.2 with `_raw_confidence` 0.8 for the whole F1 run while the output was 174.
The research doc's confidence recipe (§7, upgrade 5) includes "agent agreement" for this
reason; the current 50/50 blend of raw confidence and prediction ratio cannot see the
disagreement. At minimum: when the best agent's tempo is not within ~5% of `_raw_bpm` or a
2×/0.5× multiple of it, cap confidence and prefer re-seeding.

### F8 — snare onset flashes bypass all beat logic (medium, plausible)

`engine.py` ~line 686: `if trigger_beat or beat_info.snare_onset:` white flash. Mid-band
transients (vocal stabs, synth chords) flash at up to 8 Hz (120 ms cooldown), limited only by
the 3 Hz safety limiter. For the A/B test to mean anything this needs its own toggle or should
be folded into the metric-gated path.

### F9 — confidence floor keeps the metric filter in passthrough (medium, plausible)

`_check_metric_beat_alignment` returns `True` below `_confidence_gate = 0.4`. Confidence is
`0.5 · raw + 0.5 · prediction_ratio`, times coasting. `raw` is an SNR of the p=0.5 compressed
autocorrelation, which is flat by construction (peak 0.19 vs 0.78 for p=2 on the same buffer),
so `raw` rarely exceeds ~0.5 outside ideal signals. Benchmarks show `avg_confidence` 0.36–0.52.
Either compute SNR on a less compressed curve, or gate the metric filter on agent score/age
rather than on this blend.

### F10 — two clocks (medium, plausible)

`detect()` timestamps with `time.monotonic()` at processing time. `audio_loop` runs at 50 Hz and
processes all buffered frames (0–2 per tick at 43 Hz), so consecutive frames often share a
timestamp and onset times carry up to one tick of jitter. Agent phases advance by `_frame_dur`
per frame (audio clock) but confirmations, cooldown and predictions compare against wall clock.
Use a sample-count clock (`frames_processed · hop / sr`) plus one measured offset to wall time.

### F11–F15

See the summary table. F11 and F12 change detector sensitivity in dense sections; F13 only
matters for non-default `BUFFER_SIZE`; F14 is cleanup; F15 is cosmetic.

## What this means for the symptom

In production the predictive path is mostly inactive (confidence rarely ≥ 0.6, F9), so the
lights follow `is_beat` plus `snare_onset` flashes. `is_beat` is a raw onset detector whose
cooldown lets the wrong onset through on syncopated patterns (F3), and the BPM/phase that
everything else depends on comes from a mis-seeded agent (F1). "Sometimes right, sometimes
extra flashes" is the expected output of this pipeline, not a tuning problem.

## Recommended order

1. F2 (glue fix, 10 lines) and F8 (toggle) so the metric filter can actually be A/B tested.
2. F1 + F6 + F7 (agent seeding with phase alignment, grace period, agreement check). Measure
   with the synthetic harness before/after: CMLc/AMLc, FP per minute, BPM error.
3. F5 (prediction from PLL phase) and F3 (reactive trigger = metric onset; cooldown not reset
   by rejected onsets).
4. F4 (cooldown = `max(manual, k·period)` or drop auto-cooldown once the PLL gates onsets).
5. F9, F10, then the low-severity items.

Each step: run `tools/synth_benchmark.py` with a tag and compare against `baseline`.
Turn the F1/F3/F4 simulations in this document into unit tests (`tests/test_beat_detector.py`)
so they cannot regress.

## Baseline measurements (synthetic harness)

Tools: `tools/synth_audio.py` (deterministic scenarios with exact ground truth) and
`tools/synth_benchmark.py` (frame-by-frame run through analyzer + detector, `mir_eval.beat`
scoring, first 5 s skipped). Results: `tools/benchmark_results/synth/*_baseline.json`.
Harness sanity check: a bare 4/4 kick + 8th-hat control track scores F 1.000 / CMLc 1.000 on the
raw stream, so low scores below are the detector, not the truth alignment.

Metric stream (`is_beat and is_metric_beat`), baseline code:

```
scenario            config                F   CMLc   CMLt  FP/min  Miss/min     BPM   true  conf
techno_130          default           0.177  0.000  0.000    38.2     113.5   179.7  130.0  0.53
techno_130          preset:techno     0.436  0.013  0.096   104.7      64.4   142.5  130.0  0.32
techno_130_synco    default           0.161  0.000  0.000     7.6         -    89.1  130.0     -
techno_130_synco    preset:techno     0.445  0.012      -   117.8         -   149.5  130.0     -
house_125           default           0.229  0.000      -    36.0         -   160.5  125.0     -
house_125           preset:house      0.441  0.014      -    91.6         -   130.2  125.0     -
dnb_174             default           0.206  0.019      -    26.2         -   147.6  174.0     -
dnb_174             preset:dnb        0.592  0.082  0.478    67.6      72.0   173.2  174.0     -
trap_70             default           0.193  0.000      -    63.3         -   138.7   70.0     -
trap_70             preset:trap       0.262  0.030      -    53.5         -    70.6   70.0     -
techno_breakdown    default           0.259  0.008      -    24.0         -   102.9  130.0     -
techno_breakdown    preset:techno     0.500  0.028      -    82.9         -   122.9  130.0     -
silence_then_start  default           0.274  0.012      -    22.5         -   137.8  130.0     -
silence_then_start  preset:techno     0.511  0.039      -    81.0         -   129.7  130.0     -
```

Observations that add to the findings above:

- Wrong BPM on every default-range scenario (F1 mechanism). Presets get the BPM close only
  because the narrowed range leaves few wrong hypotheses to win.
- CMLt ≫ CMLc everywhere (dnb preset 0.478 vs 0.082): the tracker is briefly right often and
  never stays right. Continuity failure, not a clean octave error (AMLc − CMLc ≈ 0).
- Misses dominate: 48–150 misses/min against 70–174 reference beats/min, alongside high FP/min.
  The trigger is spent on the wrong event (F3), not merely late.
- Under presets mean confidence is 0.25–0.34 and never crosses 0.6 in 4 of 7 runs, so the
  predictive trigger (threshold 0.6) is never active exactly in the configurations that get the
  BPM right. Under presets `metric ≡ raw` because confidence never leaves the 0.4 passthrough (F9).
- `predicted_next_beat` moves backwards in time when the period shrinks (F5); the harness has to
  sort the stream before scoring.

Targets after fixes, on these same scenarios: CMLc > 0.8 on the straight patterns, FP/min < 5,
Miss/min < 5, BPM within 1% by 5 s, confidence ≥ 0.6 sustained under presets.

## Fixes applied (September 2026) and results

Commits, in order: glue fix for F2/F8; onset front-end + multi-agent rewrite (F1, F3, F4, F5,
F6, F7, F14, F15); comb-phase seeding, prior-weighted pruning, quality-based confidence (F9);
evidence-gated scoring, capture window, period anchoring; audio-clock timestamps (F10).

Additional findings made while fixing, not in the original list:

| ID | Where | One line |
|---|---|---|
| F16 | `beat_detector.py:detect` (old) | SuperFlux threshold was 1.5 x median over 0.2 s. Between hits the median is 0, so every positive flux value was an onset (gated only by cooldown). Replaced by the normalized ODF with `mean + 1.5 std` and a floor. |
| F17 | `beat_detector.py:detect` (old) | Bass detector was level-based (`bass > median x 1.25`); the decaying kick tail re-fired onsets 70-90 ms after the attack with full weight. Replaced by rectified bass increase. |
| F18 | `beat_detector.py:_seed_agents…` (old) | Agents were deduplicated on tempo only, so a wrong-phase agent (locked to a syncopated hit) blocked the correct phase at the same tempo forever. Now deduplicated on tempo and phase; phase from a comb search over the onset buffer. |
| F19 | `beat_detector.py:_prune_agents` | A half/double-tempo agent is confirmed on every one of its predictions and out-scored the true agent whenever some beats lacked onsets. Pruning and selection now use `score x tempo_prior(autocorrelation)`. |
| F20 | `beat_detector.py` | PI loop hunted +-1.5 BPM around the true tempo with phase errors of +-20 ms (period gains too high); autocorrelation lag quantization gave 129.2 for 130 BPM. Gains lowered, peaks parabolically interpolated, agents anchored to the autocorrelation period. |
| F21 | `beat_detector.py` | During the rewrite a stale duplicate `_sync_best_agent` (syncing from the raw top score) shadowed the new one; caught by the unit-test sim. |

| F22 | `engine.py:tick` | The 3 Hz flash limiter was shared by beat flashes and the per-band overlays (bass pulse, hi-hat sparkle). A sparkle 200 ms before the kick blocked the beat flash: end-to-end only 48 of 119 validated triggers produced a flash on techno. Beat/drop flashes are now limited against the previous beat flash only; overlays yield to the beat. |
| F23 | `engine.py`, `core/config.py` | The metric filter defaulted to OFF, i.e. the production path was raw onsets + snare flashes (100+ false flashes/min end-to-end on the techno scenario). New `Settings.metric_beat_filter` (default ON) is applied at startup; the UI toggle still overrides at runtime. |
| F24 | `beat_detector.py` | With no evidence the normalization reference decays to its floor and a flat, non-zero ODF creeps above `mean + k std` (std = 0), producing onsets every 60 ms that confirmed and dragged the agents. Threshold gets a `1.2 x mean` term; agents are only touched by onsets when the onset is strong or evidence is recent. |

Synthetic benchmark, metric stream (`is_metric_beat`), genre presets, baseline vs final:

```
scenario            F  base -> final   CMLc base -> final   FP/min base -> final   BPM final (true)
techno_130          0.436 -> 1.000     0.013 -> 1.000       104.7 -> 0.0           130.2 (130)
techno_130_synco    0.445 -> 1.000     0.012 -> 1.000       117.8 -> 0.0           130.3 (130)
house_125           0.441 -> 1.000     0.014 -> 1.000        91.6 -> 0.0           125.4 (125)
dnb_174             0.592 -> 0.907     0.082 -> 0.069        67.6 -> 0.0           173.9 (174)
trap_70             0.262 -> 0.634     0.030 -> 0.031        53.5 -> 5.5            70.6 (70)
techno_breakdown    0.500 -> 0.774     0.028 -> 0.328        82.9 -> 6.5           130.3 (130)
silence_then_start  0.511 -> 0.994     0.039 -> 0.988        81.0 -> 0.0           129.9 (130)
```

End-to-end (`tools/engine_benchmark.py`: real `AudioPipeline` glue at 50 Hz, `EffectEngine.tick`,
scored on the ticks where a beat flash starts, 60 ms latency compensation modelled), genre
presets, final code:

```
scenario            engine      F      CMLc   FP/min  Miss/min
techno_130          reactive    0.368  0.028  102.5   77.5      <- the old default path
techno_130          metric      0.996  0.992    1.1    0.0
techno_130_synco    metric      0.996  0.992    1.1    0.0
house_125           metric      0.996  0.991    1.1    0.0
dnb_174             metric      0.944  0.126    1.1   17.4
trap_70             metric      0.800  0.281   12.0   15.3
techno_breakdown    metric      0.929  0.542    9.8    8.7
silence_then_start  metric      0.902  0.897   13.5   12.0
```

Mean confidence after 5 s is 0.89-1.00 in every preset run (baseline 0.25-0.34, so the
predictive path was never active before). Flashes land ~40 ms after the beat on average in the
benchmark; in production the latency compensation and calibration slider absorb that offset.

Remaining, by design or deferred:

- DnB and half-time trap need their genre presets. With the default 80-180 range the
  autocorrelation picks 2/3 tempo for two-step (117.5 for 174) and 2x for trap (140 for 70).
  A metrical-level model (bar-level periodicity) would be needed to fix this without presets.
- The metric stream cannot fire on beats without an onset (DnB beat 3, trap beats 2 and 4,
  breakdowns). The predictive stream covers these.
- F12 (bass boost clip at 1.5) and F13 (STFT overlap only correct for `hop == fft/2`) are
  unchanged; both are outside the beat path's critical behaviour now that the ODF normalizes.
- The predictive path fires more than once per beat when the prediction shifts by > 50 ms
  within a beat (agent switch or capture); the flash limiter absorbs the duplicate (1.1 FP/min).
  Tightening `is_new_prediction` to a fraction of the period would remove it.
- No real annotated audio yet. The synthetic scenarios are exact but idealized; the next step
  is 3-5 real tracks with hand-corrected beat annotations on 30 s excerpts.
