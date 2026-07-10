# Beat Detection Benchmarking for Electronic Music — Research Report

*Generated 2026-05-05 · 5 sub-agents + 1 adversary · 50+ unique sources visited*

## TL;DR
- **No EDM-beat-annotated public dataset exists.** GiantSteps (664 Beatport tracks) is the only EDM corpus, but it ships **tempo only**, not beat times. For beat-level evaluation you'll need to either self-annotate a small EDM set or use the genre-mixed standards (Ballroom, SMC, Harmonix Set, RWC) and accept that they under-cover techno/house/DnB/trap. [GiantSteps repo](https://github.com/GiantSteps/giantsteps-tempo-dataset)
- **Benchmark with `mir_eval.beat`** — it's the canonical implementation of the MIREX metric suite (F-measure ±70 ms, Cemgil σ=40 ms, Goto, P-score, **CMLc/CMLt/AMLc/AMLt**, information gain D/Dg). For driving lights, **CMLc and AMLt** matter most: continuity-based metrics measure stable lock; AMLt exposes octave errors directly via the AMLt−CMLt gap. [mir_eval source](https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/beat.py)
- **Online evaluation = offline metrics + 5-second cold-start skip + RTF.** No latency-aware standard yet; one paper (Heydari TISMIR 2024) explicitly disregards beats < 5 s to skip PLL warm-up, and reports F1 + Real-Time-Factor + system latency separately. [TISMIR 2024](https://transactions.ismir.net/articles/10.5334/tismir.189)
- **Most likely diagnosis for your "3 mini-flashes between 2 kicks" symptom**: (a) **octave-up error** — PLL latched on 8th- or 16th-note period (snare-on-2-and-4, or hi-hat cascade), or (b) onset detector firing on non-beat events (hi-hats, snare/clap). The AMLc−CMLc gap on a benchmark run will diagnose this directly: if AMLc is much higher than CMLc, it's an octave error. [synthesis from worker findings]
- **Build the harness with `mir_eval` + `mirdata`** (Python lib that downloads/wraps Ballroom, GTZAN, RWC, GiantSteps, Harmonix). Self-annotate a small EDM test set (8–16 tracks × 30 s) covering techno, house, DnB, trap by tapping along to a click track or hand-correcting a strong tracker's output. [mirdata docs](https://mirdata.readthedocs.io)

---

## Key findings

### 1. Public datasets — what you can actually use

*Confidence: high — multiple corroborating sources; gap on EDM-with-beat-times is verified absent across all searches*

There are ~6 standard datasets used in MIREX/ISMIR beat tracking evaluation. Coverage of electronic music is sparse:

| Dataset | Size | Beat times? | EDM/electronic content? | Access |
|---|---|---|---|---|
| MIREX 2006 set | 160 × 30 s | yes | minimal | request-based, academic |
| Ballroom | 698 × ~30 s | yes (+ bars) | no — ballroom dance | via [mirdata](https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/ballroom.html) |
| SMC Mirum | 217 × ~40 s | yes | no — "hard cases" (Romantic, film, blues) | MIREX-restricted |
| Harmonix Set | 912 full pop tracks | yes (+ downbeats + segments) | no — 0% electronic | [Zenodo](https://zenodo.org/records/3527870) |
| RWC Popular/Jazz | ~350 tracks | yes (+ bars) | no | request-based ([AIST](https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/)) |
| **GiantSteps Tempo** | 664 × 120 s Beatport EDM | **NO — BPM only** | yes (techno, house, trance, DnB, dubstep) | [free GitHub](https://github.com/GiantSteps/giantsteps-tempo-dataset) |
| HJDB (Hardcore Jungle / DnB) | small, ISMIR 2012 paper context only | likely yes per paper | yes — DnB | not easily obtainable [interpretation] |

The standout gap: **no public dataset combines beat-time annotations with electronic music content at scale.** GiantSteps was annotated for tempo via Beatport user corrections and forum crowdsourcing — not beat grids. For a hue-light driver targeting techno/house/DnB/trap, the realistic option is a small self-annotated test set on top of Ballroom/Harmonix for sanity-check coverage. The Python library [mirdata](https://mirdata.readthedocs.io) abstracts download + parsing for most of the standard sets.

### 2. Metrics — what to compute and why

*Confidence: high — mir_eval is the de-facto MIREX reference and is open-source*

The MIREX standard metric suite, all implemented in [mir_eval.beat](https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/beat.py):

- **F-measure** (default ±70 ms tolerance) — TP/FP/FN within window; **blind to tempo stability** — a tracker that locks for 5 s, drifts for 5 s, and re-locks can still score high.
- **Cemgil** (Gaussian σ=40 ms) — same idea but smoother weighting; less informative than F.
- **Goto** — hard threshold; mostly historical.
- **P-score** (McKinney) — adaptive tolerance = 0.2 × median IBI; correlates well with subjective rating.
- **CMLc / CMLt** — Continuity at Correct Metric Level. Requires beats within 70 ms **AND** inter-beat interval within ±17.5% of reference tempo. CMLc = continuous longest-correct fraction; CMLt = total correct fraction (continuity not required).
- **AMLc / AMLt** — Allowed Metric Levels: same as CMLc/CMLt but **also accepts octave-up, octave-down, off-beat, and triple variants** of the reference. The **AMLc − CMLc gap is the canonical octave-error diagnostic**: large gap = tracker locks on wrong metrical level but consistently.
- **Information gain (D / Dg)** — entropy of the beat-error histogram in bits; captures error *distribution* shape rather than binary correctness. High D + low CMLc = systematic timing bias (e.g. always firing 30 ms early).

For lights specifically:

- The **CMLc/AMLc gap** is the most actionable single number for diagnosing your reported symptom — if AMLc is much higher than CMLc, you have an octave error.
- **F-measure alone is misleading** for a real-time light driver; you can get high F-measure on tracks where the tracker silently jumps periods. [domain inference from metric definitions]
- **Information gain** is good for post-hoc analysis: low-D failures are random (broken detector); high-D + low-F failures are systematic (fixable via calibration).

`mir_eval.beat.evaluate(reference, estimated)` returns the full dict in one call. Madmom has [equivalent functions](https://madmom.readthedocs.io/en/v0.16/modules/evaluation/beats.html) but does not quantize beats to a 100 Hz grid before scoring; numerical results between mir_eval and madmom won't match exactly. Pick one and stick with it; mir_eval is the safer choice for MIREX comparability.

### 3. Real-time / online evaluation — what's different from offline

*Confidence: medium — the field has no codified online standard; practice is documented in 4–5 papers from 2021–2025*

**MIREX has never had an online beat-tracking subtask.** All MIREX evaluation since 2006 is offline batch on full clips. [MIREX 2025 task page](https://music-ir.org/mirex/wiki/2025:Audio_Beat_Tracking) Standard mir_eval scores applied to a real-time tracker's emitted beat stream are valid, but you have to handle:

1. **Cold-start exclusion.** The most documented convention is **disregard beats with t < 5.0 s** during scoring, motivated by the inevitable PLL warm-up transient. ([Heydari TISMIR 2024](https://transactions.ismir.net/articles/10.5334/tismir.189) explicitly says "we disregard all beats occurring before 5 s of each music track.") This is a *common* convention but **not a codified universal standard** — BeatNet+ does not document whether it applies the same exclusion. [adversary-flagged: do not present as universal]

2. **Lookahead.** No agreed definition of "online" — practical usage:
   - **Strict causal**: Madmom `online=True` (forward-only, frame-by-frame). [madmom DBN docs](https://madmom.readthedocs.io/en/v0.16/modules/features/beats.html)
   - **~1 frame lookahead for zero-latency output** (Heydari TISMIR 2024)
   - **A few frames** ("latency-controlled") — LC-Beating ICME 2023 [link](https://ieeexplore.ieee.org/document/10219926/)
   - **<50 ms** — BEAST streaming Transformer [arXiv](https://arxiv.org/abs/2312.17156)

   For your system's predictive PLL firing 50–100 ms early: this is **within the standard ±70 ms F-measure tolerance**, so it's credited as correct. The Heydari paper measured F1 degradation as a function of beat lookahead: −0.04 % at 11.6 ms, −2.33 % at 580 ms — predictive firing in the 50–100 ms range is essentially free in standard scoring.

3. **Report RTF (Real-Time Factor) and latency separately**, alongside the standard metrics. RTF = total processing time ÷ audio duration; if RTF > 1.0 the system can't run live. [BeatNet+ TISMIR 2024](https://transactions.ismir.net/articles/10.5334/tismir.198)

4. **Online-vs-offline F1 cost is real but modest.** RNN+PLP measured 79.07 % offline → 74.72 % online on GTZAN with the same architecture (~4.4 pp). [Heydari TISMIR 2024](https://transactions.ismir.net/articles/10.5334/tismir.189)

5. **LAF1 (Latency-Adjusted F1)** is an emerging single-score metric introduced in a 2025 Springer paper that combines accuracy and latency penalties. **[unverified — single source, paywalled]** Not a community standard yet.

6. **L-correct** (Heydari TISMIR 2024) evaluates *consecutive* beat sequences rather than individual beats — better captures lock stability. Worth implementing alongside CMLc.

For **IBT** (the multi-agent system your detector descends from) the original evaluation used standard offline metrics in MIREX 2012, plus a "streaming scenario" with long audio streams to expose steady-state vs warm-up behavior. [IBT IEEE 2012](https://ieeexplore.ieee.org/document/6255768/)

### 4. Failure modes on electronic music — diagnosing your symptom

*Confidence: medium — qualitative mechanisms are well-established; some specific quantitative claims from worker 4 were [adversary-flagged] for unsupported citations and are downgraded to mechanistic inference*

Your reported symptom — **"sometimes 3 smaller flashes between 2 actual kicks"** — most plausibly maps to one of these documented failure modes:

**(A) Octave-up tempo error (×2 or ×4).** The dominant EDM failure mode in the literature. Mechanism: autocorrelation of the onset envelope produces peaks at the beat period **and its integer fractions**. Steady 16th-note hi-hats, 4-on-the-floor patterns with snare on 2 and 4, or sidechain pumping all create strong periodicity at sub-beat divisions. Without a strong tempo prior, the tracker selects the wrong harmonic. [TISMIR 2020 tempo survey](https://transactions.ismir.net/articles/10.5334/tismir.43) DnB (high BPM, syncopated) is reported as the worst case in industry practice (Rekordbox/Serato require manual BPM correction frequently). [Pioneer DJ forum](https://community.pioneerdj.com/hc/en-us/community/posts/22977088055961-Rekordbox-screwing-up-the-analyzed-BPM) [Serato forum](https://serato.com/forum/discussion/898313) [adversary-flagged: forum evidence is anecdotal — do not generalize quantitatively]

The **hi-hat cascade** specifically (4× rate lock-in producing 4 detected beats per actual beat — exactly your "3 mini-flashes between 2 kicks" symptom) is a mechanistic prediction from how SuperFlux's log-Mel filterbank captures hi-hat energy without metrical hierarchy logic. [interpretation, based on [librosa SuperFlux docs](https://librosa.org/doc/main/auto_examples/plot_superflux.html), [SuperFlux GitHub](https://github.com/CPJKU/SuperFlux/blob/master/README.md)]

**(B) Snare/clap on 2 and 4 → 8th-note lock.** In 4-on-the-floor with snare on 2 and 4, an onset detector that weighs all bands equally sees onsets every 8th note (kick + snare alternation). The PLL can lock onto the 8th-note period, producing 2 flashes per intended beat. [interpretation, mechanism per [Ellis 2007](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf)] Robust systems (Ellis, madmom) weight kick-band (60–130 Hz) onsets more heavily than snare-band (200–800 Hz) onsets to avoid this.

**(C) Onset detector firing on hi-hats / claps directly without metrical filtering.** If your calibrate-mode flash is wired to *onsets* rather than *beats* in a particular code path, every loud transient flashes — hi-hats, claps, percussion fills. The downstream metrical-level logic (PLL agent voting, period selection) is where beats should be filtered down to the quarter-note grid. **Worth checking that the calibrate path actually consumes the post-PLL beat stream and not the raw onset detection function.** [interpretation]

**(D) Multi-agent flip-flopping.** With 3–15 competing tempo hypotheses, two agents at similar score can oscillate between selections frame-to-frame, producing intermittent extra beats. The original IBT paper ([Oliveira et al. ISMIR 2010](https://archives.ismir.net/ismir2010/paper/000050.pdf)) does not document hysteresis or damping explicitly — meaning your implementation should add explicit hysteresis / minimum-dwell-time before agent switching. [interpretation, single-source IBT context]

**(E) Sub-bass 808 kicks.** Standard SuperFlux with `fmin=27.5 Hz` covers sub-bass in principle, but onset peaks for 808 kicks are typically *broader and weaker* than for tight transient kicks, since 808s have long pitch envelopes (sliding ~60 Hz → ~30 Hz over 200 ms). PLL weighting the beat at that position less can desync from the perceived kick. [librosa SuperFlux docs](https://librosa.org/doc/main/auto_examples/plot_superflux.html) [adversary-flagged: HFC linear-bin claim was unsupported by cited URL — keep this paragraph as mechanistic inference]

**(F) Sidechain ducking false onsets.** Sidechain compression in EDM ducks non-kick elements on each kick and releases between kicks. The pump-release transient (other elements lifting back up, ~100–300 ms post-kick) creates an additional energy increase in the onset function — a *false* onset between beats. **No peer-reviewed source confirms this directly**; mechanistically plausible from compressor behavior. [adversary-flagged: mechanistic inference, not citation-supported]

**(G) Drops/breakdowns/build-ups.** During breakdowns the kick disappears; the PLL coasts on its last estimate; if the breakdown is long, it drifts. Build-up risers and snare rolls produce non-beat-aligned onsets. The user's existing system already implements coasting + sidechain compression detection — these mitigations match documented practice (IBT's "automatic monitoring and state recovery"). [IBT IEEE 2012](https://ieeexplore.ieee.org/document/6255768/)

### 5. SuperFlux and PI-PLL — algorithm-internal weaknesses

*Confidence: medium — published critique is sparse; some claims rely on first-principles reasoning where literature is silent*

- **SuperFlux requires the right filterbank.** Madmom docs explicitly state: *"This method works only properly, if the spectrogram is filtered with a filterbank of the right frequency spacing. Filter banks with 24 bands per octave (i.e. quarter-tone resolution) usually yield good results."* [madmom onsets](https://madmom.readthedocs.io/en/v0.16/modules/features/onsets.html) **Verify your implementation's filterbank matches.** A Mel filterbank with arbitrary band count (e.g. 138 bands as in some librosa examples) is *not* equivalent to the 24-bands-per-octave configuration the algorithm was designed for. [adversary-flagged: this is a contested implementation choice — librosa example uses Mel, madmom production uses 24-per-octave; may explain genre-dependent inconsistency]

- **SuperFlux's vibrato suppression does NOT suppress steady percussion.** The maximum-filter is designed to ignore *sustained pitch vibrato in melodic instruments*, not to ignore steady periodic percussion like hi-hats. [librosa SuperFlux docs](https://librosa.org/doc/main/auto_examples/plot_superflux.html) [interpretation — librosa demonstrates vibrato suppression; the negative claim about percussion is inference]

- **PI-PLL gain tuning trade-off.** Lower loop-filter response time = faster tempo-change detection, but higher susceptibility to spurious-onset phase pulls. [IEEE PLL beat tracking](https://ieeexplore.ieee.org/document/4145989) Standard musical-tracking Kp/Ki ranges are not documented in the literature. Your "coasting" behavior during low-confidence periods is exactly the right mitigation.

- **Multi-agent oscillation is theoretically possible but not formally documented.** No IBT-derived paper describes hysteresis mechanisms between equally-scored agents. With 3–15 agents and similar fitness scores in dense passages, oscillation is a real risk; explicit minimum dwell time before switching, or score margin requirements, are not literature-standard but are good defensive engineering. [interpretation]

- **Generalized autocorrelation `p=0.5`** ([Percival & Tzanetakis 2014](https://dl.acm.org/doi/10.1109/TASLP.2014.2348916)) reduces frequency-domain compression vs `p=2` (standard autocorrelation). The original paper recommends `p=0.5` empirically; rationale and failure conditions for the choice are not documented in accessible summaries. [unverified — paper PDF could not be fetched]

- **CNN/RNN onset detectors outperform SuperFlux** on matched test sets (Schlüter et al. ICASSP 2014 [PDF](https://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf)) — but SuperFlux remains widely used for speed, interpretability, and zero training-data requirement. Replacing SuperFlux is not the right first step; benchmark first.

---

## Conflicts and disagreements

### GiantSteps quantitative figures

- **Position A** (worker 4): cited Acc1=77.0% / Acc2=90.2% on GiantSteps, with the GitHub README as source.
- **Position B** (TISMIR 2020 paper [tismir.43](https://transactions.ismir.net/articles/10.5334/tismir.43)): reports ACC1 = 58.9% → 64.8% and ACC2 = 86.4% → 94.0% (pre/post annotation correction).
- **Assessment**: The 77.0% / 90.2% figures could not be located in any fetched source. Adversary review flagged this as unsupported. Drop these specific numbers from any quantitative claim; the qualitative finding (octave errors are dominant on EDM, AMLc materially exceeds CMLc) is still correct.

### SuperFlux filterbank configuration

- **Position A** (worker 5, [madmom docs](https://madmom.readthedocs.io/en/v0.16/modules/features/onsets.html)): SuperFlux works properly with **24 bands per octave** (quarter-tone resolution).
- **Position B** (worker 4, librosa example): SuperFlux is demonstrated with `n_mels=138, fmin=27.5 Hz`.
- **Assessment**: These describe **different implementations**, not contradictory facts. Librosa's example is a demonstration with Mel-spaced bins; madmom's production implementation uses log-frequency 24-per-octave. The user's system should pick one configuration deliberately and document which.

### Is "5-second cold-start exclusion" a community standard?

- **Position A** (worker 3 main claim): documented community standard.
- **Position B** (worker 3 own caveat): only the Heydari TISMIR 2024 paper explicitly documents it; BeatNet/BeatNet+ does not.
- **Assessment**: It's a *common convention*, not a codified standard. Use it for your harness, but don't claim it's universal.

---

## Caveats

- **No EDM-with-beat-times public dataset exists.** This is the single biggest gap. Plan to self-annotate a small EDM evaluation set (8–16 tracks × 30 s) — it's cheap and pays off long-term.
- **Worker 4 had multiple adversary-flagged citation issues** on quantitative EDM failure-mode claims (specific Acc1/Acc2 numbers, DnB ambiguity percentages, HFC sub-bass weighting, sidechain ducking). The qualitative mechanisms remain plausible and consistent with the user's reported symptom; the specific numbers do not. Treat the failure-mode taxonomy as a hypothesis space, not a citable benchmark.
- **Several PDFs were unreadable** during research — Beat Critic ISMIR 2010, SMC 2015 EDM octave errors, Hockman 2012 DnB downbeat, Ellis 2007 — these would deepen the failure-mode analysis but don't change the recommended benchmark architecture.
- **LAF1 metric exists but is paywalled and single-source** — don't depend on it; mir_eval's standard suite + RTF + latency reporting is sufficient for actionable diagnosis.
- **Online beat tracking is a young evaluation discipline.** The "standard" for online evaluation is genuinely unsettled (2023–2025 papers each define their own latency metric). Adopting offline mir_eval + 5 s skip + RTF is current best practice but may shift in 18 months.
- **Benchmark harness should distinguish onsets from beats.** First debugging step before running formal metrics: log both the SuperFlux onset times and the post-PLL beat times to disk, and confirm the calibrate-mode flash consumes the latter, not the former.

---

## Recommended benchmark harness — concrete starting plan

[interpretation, synthesizing across all worker findings]

```
tools/benchmark_beats.py
├── Load audio + reference beats (from a dataset or self-annotated .beats file)
├── Run beat detector causally (frame-by-frame, no peek-ahead)
│   └── Capture: emitted beat times, onset times, processing time, RTF
├── Skip beats with t < 5.0 s in both reference and estimate
├── Compute mir_eval.beat.evaluate(ref, est) → full metric dict
├── Compute the AMLc/CMLc gap explicitly → octave error diagnostic
├── Per-track output: F-measure, CMLc, CMLt, AMLc, AMLt, P-score, info-gain D
├── Aggregate across dataset: mean, median, per-genre breakdown
└── Output: JSON results file + stdout summary
```

**Datasets to run against, in priority order:**
1. **Self-annotated EDM mini-set** (techno, house, DnB, trap × 2–4 tracks each, 30 s per track). Annotate with Audacity label track + click; export `.beats` plain text (one timestamp per line in seconds).
2. **Ballroom** (via `mirdata`) — sanity check on regular 4/4 with high-quality annotations.
3. **Harmonix Set** (Zenodo) — broader pop coverage with downbeats.
4. **GiantSteps Tempo** — *tempo-only* check; you can verify your detector's tempo estimate even without beat-time evaluation.

**Diagnostic procedure for "3 mini-flashes between 2 kicks":**
1. Log onset times and post-PLL beat times for one bad track. **First: confirm calibrate flash uses beats, not onsets.**
2. If it's beats: run mir_eval, check **AMLc − CMLc**. Big gap = octave error (likely 2× or 4×).
3. Look at the inter-beat interval distribution: if it's bimodal at IBI and IBI/2, you have hi-hat cascade or snare-on-2-and-4 lock-in.
4. Check multi-agent voting log: are agents oscillating frame-to-frame? Add hysteresis if so.
5. Verify SuperFlux filterbank configuration matches the algorithm's design assumption.

---

## Sources

- [audioXpress — Audio Processing Beat Tracking Explained](https://audioxpress.com/article/audio-processing-beat-tracking-explained)
- [BeatNet (Heydari et al., ISMIR 2021)](https://arxiv.org/abs/2108.03576)
- [BeatNet+ (TISMIR 2024)](https://transactions.ismir.net/articles/10.5334/tismir.198)
- [Beat-Tracking-Evaluation-Toolbox](https://github.com/adamstark/Beat-Tracking-Evaluation-Toolbox/blob/master/README.md)
- [BEAST: Streaming Transformer (arXiv 2023)](https://arxiv.org/abs/2312.17156)
- [Ballroom dataset via mirdata](https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/ballroom.html)
- [Beat & Tempo Tracking (Krzyzaniak)](https://github.com/michaelkrzyzaniak/Beat-and-Tempo-Tracking)
- [Davies, Degara, Plumbley — Evaluation Methods (2009, ResearchGate)](https://www.researchgate.net/publication/228724188_Evaluation_Methods_for_Musical_Audio_Beat_Tracking_Algorithms)
- [Don't Look Back: Online Beat Tracking (ICASSP 2021)](https://arxiv.org/abs/2011.02619)
- [Ellis 2007 — Beat Tracking by Dynamic Programming](https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf)
- [Evaluation of Beat Tracking Measures (ISMIR 2014)](https://archives.ismir.net/ismir2014/paper/000238.pdf)
- [GiantSteps Tempo dataset (GitHub)](https://github.com/GiantSteps/giantsteps-tempo-dataset)
- [GiantSteps+ Key dataset (Zenodo)](https://zenodo.org/records/1095691)
- [Harmonix Set (GitHub)](https://github.com/urinieto/harmonixset)
- [Harmonix Set (Zenodo)](https://zenodo.org/records/3527870)
- [Heydari — Real-Time Beat Tracking with Zero Latency (TISMIR 2024)](https://transactions.ismir.net/articles/10.5334/tismir.189)
- [IBT — Beat Tracking for Multiple Applications (IEEE TASLP 2012)](https://ieeexplore.ieee.org/document/6255768/)
- [IBT — Real-Time Tempo and Beat Tracking (ISMIR 2010)](https://archives.ismir.net/ismir2010/paper/000050.pdf)
- [LC-Beating (ICME 2023)](https://ieeexplore.ieee.org/document/10219926/)
- [librosa SuperFlux example](https://librosa.org/doc/main/auto_examples/plot_superflux.html)
- [Madmom — beats evaluation](https://madmom.readthedocs.io/en/v0.16/modules/evaluation/beats.html)
- [Madmom — DBNBeatTracking (online)](https://madmom.readthedocs.io/en/v0.16/modules/features/beats.html)
- [Madmom — onset detection](https://madmom.readthedocs.io/en/v0.16/modules/features/onsets.html)
- [Madmom GitHub Discussion #503](https://github.com/CPJKU/madmom/discussions/503)
- [MIREX 2025 Audio Beat Tracking task](https://music-ir.org/mirex/wiki/2025:Audio_Beat_Tracking)
- [MIREX 2006 Audio Beat Tracking](https://www.music-ir.org/mirex/wiki/2006:Audio_Beat_Tracking)
- [mir_eval beat.py source](https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/beat.py)
- [mir_eval documentation](https://mir-eval.readthedocs.io/latest/)
- [mir_eval ISMIR 2014 paper](https://brianmcfee.net/papers/ismir2014_mireval.pdf)
- [mirdata documentation](https://mirdata.readthedocs.io/en/stable/source/mirdata.html)
- [Online Joint Beat/Downbeat with Time Series Forecasting (Springer 2025) — LAF1](https://link.springer.com/chapter/10.1007/978-981-96-4783-5_2)
- [On-Line Musical Beat Tracking with PLL (IEEE)](https://ieeexplore.ieee.org/document/4145989)
- [Percival & Tzanetakis — Streamlined Tempo Estimation (TASLP 2014)](https://dl.acm.org/doi/10.1109/TASLP.2014.2348916)
- [Pioneer DJ community — Rekordbox BPM analysis](https://community.pioneerdj.com/hc/en-us/community/posts/22977088055961-Rekordbox-screwing-up-the-analyzed-BPM)
- [RWC AIST Annotations](https://staff.aist.go.jp/m.goto/RWC-MDB/AIST-Annotation/)
- [Schlüter et al. — Improved onset detection with CNNs (ICASSP 2014)](https://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf)
- [Serato beat-grid forum thread](https://serato.com/forum/discussion/898313)
- [SuperFlux GitHub README](https://github.com/CPJKU/SuperFlux/blob/master/README.md)
- [Tempo, Beat and Downbeat Tutorial — Evaluate](https://tempobeatdownbeat.github.io/tutorial/ch2_basics/evaluate.html)
- [TISMIR 2020 — Music Tempo Estimation: Are We Done Yet?](https://transactions.ismir.net/articles/10.5334/tismir.43)

## Methodology

- Decomposition: 5 sub-questions (datasets, metrics, online eval, EDM failure modes, SuperFlux+PLL gotchas)
- Workers spawned: 5 initial + 0 follow-up + 1 adversary = 6 total
- Models: 3 × Haiku 4.5 (standard depth) + 2 × Sonnet 4.6 (deep depth) + 1 × Sonnet 4.6 (adversary)
- Tool calls: ~31 searches + ~28 fetches across workers
- Sources visited: 50+ unique URLs across 5 worker files
- Adversary verdicts: 2 supported, 1 partially supported, 4 unsupported (mostly worker-4 quantitative claims), 1 unavailable (paywalled), 3 hidden contradictions surfaced

<!-- METRICS:{"sub_questions":5,"workers_initial":5,"workers_followup":0,"sources_unique":40,"depth_distribution":{"shallow":0,"standard":3,"deep":2},"conflicts_surfaced":3,"single_source_claims":4,"self_critique_findings":"adversary flagged 4 unsupported worker-4 quantitative claims (GiantSteps Acc1/Acc2, DnB ambiguity %, HFC sub-bass, sidechain ducking) — downgraded to mechanistic inference in synthesis","follow_up_spawned":false,"deep_fetcher_used":false,"adversary_used":true,"mode":"web","constraints_used":true} -->
