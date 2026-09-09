"""Causal beat activation from a small LSTM ensemble (numpy port of madmom's
online beat tracker front-end and RNN).

The hand-crafted onset detection function (bass increase + SuperFlux) does not
separate beats from offbeats on real electronic music (AUC 0.5-0.7 at the beat
vs the midpoint between beats, see docs/real_audio_baseline.md); this learned
activation does (AUC 0.90-0.99) and it is cheap: 8 nets x 3 LSTM layers x 25
units at 100 frames per second cost ~0.2 ms per frame.

Pipeline per 100 fps frame (Böck & Schedl 2011, madmom RNNBeatProcessor online):
    frame of 2048 samples centred on t = i * 441 / 44100, zero-padded at the start
    -> Hann window -> |rfft| (1024 bins, no Nyquist)
    -> logarithmic filterbank (12 bands/octave, 30-17000 Hz, 81 bands)
    -> log10(x + 1)
    -> positive first difference to the previous frame (zero on the first frame)
    -> [spec, diff] (162 inputs) -> 8 peephole LSTM nets -> sigmoid -> mean

The model weights (audio/models/beats_lstm_2016.npz) are madmom's BEATS_LSTM
models, CC BY-NC-SA 4.0 (non-commercial); see the LICENSE next to the file.
The input is expected as float in [-1, 1] at 44100 Hz; other rates are resampled
by a streaming polyphase FIR before framing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.special import expit

MODEL_PATH = Path(__file__).resolve().parent / "models" / "beats_lstm_2016.npz"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)


class StreamingResampler:
    """Rational-ratio polyphase FIR resampler that keeps its state between calls.

    y[k] = sum_m P[r, m] * x[q - m] with q, r = divmod(k * M, L), where P[r] is
    the r-th polyphase branch of a Kaiser low-pass designed at the lower Nyquist.
    Equivalent to scipy.signal.resample_poly up to the filter's group delay
    ((taps - 1) / 2 / L input samples, ~0.2 ms here).
    """

    def __init__(self, rate_in: int, rate_out: int, taps_per_phase: int = 20):
        g = math.gcd(rate_in, rate_out)
        self.up = rate_out // g
        self.down = rate_in // g
        # Odd tap count whose group delay is a whole number of output samples,
        # so the output is a pure integer delay of resample_poly's
        half_delay = max(1, round(taps_per_phase * self.up / (2 * self.down)))
        n_taps = 2 * half_delay * self.down + 1
        self.delay_out = half_delay  # output samples of delay
        cutoff = 1.0 / max(self.up, self.down)
        taps = signal.firwin(n_taps, cutoff, window=("kaiser", 5.0)) * self.up
        n_phase_taps = -(-n_taps // self.up)
        self._poly = np.zeros((self.up, n_phase_taps))
        for r in range(self.up):
            branch = taps[r::self.up]
            self._poly[r, :len(branch)] = branch
        self._history = np.zeros(n_phase_taps, dtype=np.float64)
        self._n_in = 0  # total input samples seen
        self._k = 0  # next output index

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float64)
        buf = np.concatenate([self._history, x])
        base = self._n_in - len(self._history)  # global input index of buf[0]
        self._n_in += len(x)
        n_phase_taps = self._poly.shape[1]
        out = []
        while True:
            q, r = divmod(self._k * self.down, self.up)
            if q >= self._n_in:
                break
            lo = max(q - n_phase_taps + 1 - base, 0)
            segment = buf[lo:q - base + 1][::-1]  # x[q], x[q-1], ...
            out.append(float(np.dot(self._poly[r, :len(segment)], segment)))
            self._k += 1
        tail = buf[-n_phase_taps:]
        if len(tail) < n_phase_taps:
            tail = np.concatenate([np.zeros(n_phase_taps - len(tail)), tail])
        self._history = tail.copy()
        return np.asarray(out, dtype=np.float32)


class BeatActivationRNN:
    """Streaming beat activation at 100 frames per second."""

    def __init__(self, sample_rate: int = 44100, model_path: Path = MODEL_PATH):
        data = np.load(model_path)
        self.meta = json.loads(str(data["meta"]))
        self.model_rate = int(self.meta["sample_rate"])
        self.frame_size = int(self.meta["frame_size"])
        self.hop = int(self.meta["hop"])
        self.fps = self.model_rate / self.hop
        self.num_nets = int(self.meta["num_nets"])
        self.num_layers = int(self.meta["num_layers"])
        self.filterbank = data["filterbank"].astype(np.float64)  # (bins, bands)
        self._window = np.hanning(self.frame_size)

        # Stack the ensemble: (nets, in, hidden) etc. so one frame runs all nets at once
        self._layers = []
        for layer_idx in range(self.num_layers):
            layer = {}
            for g in "ifco":
                layer[f"W_{g}"] = np.stack([data[f"net{n}_l{layer_idx}_W_{g}"] for n in range(self.num_nets)]).astype(np.float64)
                layer[f"R_{g}"] = np.stack([data[f"net{n}_l{layer_idx}_R_{g}"] for n in range(self.num_nets)]).astype(np.float64)
                layer[f"b_{g}"] = np.stack([data[f"net{n}_l{layer_idx}_b_{g}"] for n in range(self.num_nets)]).astype(np.float64)
            for g in "ifo":
                layer[f"p_{g}"] = np.stack([data[f"net{n}_l{layer_idx}_p_{g}"] for n in range(self.num_nets)]).astype(np.float64)
            self._layers.append(layer)
        self._ff_W = np.stack([data[f"net{n}_ff_W"] for n in range(self.num_nets)]).astype(np.float64)  # (nets, hidden, 1)
        self._ff_b = np.stack([data[f"net{n}_ff_b"] for n in range(self.num_nets)]).astype(np.float64)  # (nets, 1)

        self.input_rate = int(sample_rate)
        self._resampler = (
            None if self.input_rate == self.model_rate
            else StreamingResampler(self.input_rate, self.model_rate)
        )
        self.reset()

    def reset(self) -> None:
        hidden = self._layers[0]["b_i"].shape[1]
        self._h = [np.zeros((self.num_nets, hidden)) for _ in self._layers]
        self._c = [np.zeros((self.num_nets, hidden)) for _ in self._layers]
        self._prev_log_spec: np.ndarray | None = None
        # Sample buffer at the model rate; frame i is centred on sample i * hop
        half = self.frame_size // 2
        self._buffer = np.zeros(half, dtype=np.float64)  # zero padding before t = 0
        self._buffer_start = -half  # model-rate sample index of buffer[0]
        self._next_frame = 0
        if self._resampler is not None:
            self._resampler = StreamingResampler(self.input_rate, self.model_rate)

    # --- streaming -------------------------------------------------------

    def push(self, samples: np.ndarray) -> list[tuple[float, float]]:
        """Feed new mono samples; returns (time_sec, activation) for every frame
        that became computable. time_sec is the frame centre on the input clock
        (seconds since the first sample ever pushed)."""
        x = np.asarray(samples, dtype=np.float64)
        if self._resampler is not None:
            x = self._resampler.process(x).astype(np.float64)
        if len(x):
            self._buffer = np.concatenate([self._buffer, x])
        out = []
        half = self.frame_size // 2
        while True:
            centre = self._next_frame * self.hop
            end = centre + half  # exclusive
            if end > self._buffer_start + len(self._buffer):
                break
            lo = centre - half - self._buffer_start
            frame = self._buffer[lo:lo + self.frame_size]
            out.append((centre / self.model_rate, self._activate(frame)))
            self._next_frame += 1
        # Drop samples no future frame needs
        keep_from = self._next_frame * self.hop - half - self._buffer_start
        if keep_from > 0:
            self._buffer = self._buffer[keep_from:]
            self._buffer_start += keep_from
        return out

    def process_offline(self, audio: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Activation for a whole signal (benchmarks); same math as push()."""
        self.reset()
        values = []
        for i in range(0, len(audio), chunk):
            values.extend(a for _, a in self.push(audio[i:i + chunk]))
        return np.asarray(values, dtype=np.float32)

    # --- one frame ---------------------------------------------------------

    def _features(self, frame: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(frame * self._window))[: self.frame_size // 2]
        log_spec = np.log10(spectrum @ self.filterbank + 1.0)
        if self._prev_log_spec is None:
            diff = np.zeros_like(log_spec)
        else:
            diff = np.maximum(log_spec - self._prev_log_spec, 0.0)
        self._prev_log_spec = log_spec
        return np.concatenate([log_spec, diff])

    def _activate(self, frame: np.ndarray) -> float:
        x = self._features(frame)  # (in,)
        inp = np.broadcast_to(x, (self.num_nets, len(x)))
        for layer_idx, layer in enumerate(self._layers):
            h, c = self._h[layer_idx], self._c[layer_idx]
            # (nets, in) x (nets, in, hidden) -> (nets, hidden)
            ig = _sigmoid(np.einsum("ni,nij->nj", inp, layer["W_i"]) + layer["b_i"]
                          + c * layer["p_i"] + np.einsum("ni,nij->nj", h, layer["R_i"]))
            fg = _sigmoid(np.einsum("ni,nij->nj", inp, layer["W_f"]) + layer["b_f"]
                          + c * layer["p_f"] + np.einsum("ni,nij->nj", h, layer["R_f"]))
            cell = np.tanh(np.einsum("ni,nij->nj", inp, layer["W_c"]) + layer["b_c"]
                           + np.einsum("ni,nij->nj", h, layer["R_c"]))
            c = cell * ig + c * fg
            og = _sigmoid(np.einsum("ni,nij->nj", inp, layer["W_o"]) + layer["b_o"]
                          + c * layer["p_o"] + np.einsum("ni,nij->nj", h, layer["R_o"]))
            h = np.tanh(c) * og
            self._h[layer_idx], self._c[layer_idx] = h, c
            inp = h
        out = _sigmoid(np.einsum("nj,njk->nk", inp, self._ff_W) + self._ff_b)  # (nets, 1)
        return float(out.mean())
