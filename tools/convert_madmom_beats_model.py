"""Convert madmom's online beat-tracking LSTM ensemble to a single .npz.

Runs in the tool environment (needs madmom):

    .venv-gt/bin/python tools/convert_madmom_beats_model.py

Writes src/hue_visualizer/audio/models/beats_lstm_2016.npz with

    net{n}_l{l}_{W,R,b}_{i,f,c,o}   input / recurrent weights and biases per gate
    net{n}_l{l}_p_{i,f,o}           peephole weights (input, forget, output gates)
    net{n}_ff_W, net{n}_ff_b        sigmoid output layer
    filterbank                      (1024 bins x 81 bands) log filterbank, 12 bands
                                    per octave, 30-17000 Hz, filters normalized to 1
    meta                            JSON: sample_rate 44100, frame_size 2048, hop 441,
                                    diff_frames 1, log = log10(x + 1), num_nets

The weights are the BEATS_LSTM models from madmom (Böck, Krebs, Widmer 2016) and
are licensed CC BY-NC-SA 4.0 — see the LICENSE file next to the .npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from madmom.audio.filters import LogarithmicFilterbank
from madmom.audio.stft import fft_frequencies
from madmom.ml.nn import NeuralNetwork
from madmom.models import BEATS_LSTM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "hue_visualizer" / "audio" / "models" / "beats_lstm_2016.npz"
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP = 441

GATES = {"i": "input_gate", "f": "forget_gate", "c": "cell", "o": "output_gate"}


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    for n, path in enumerate(BEATS_LSTM):
        net = NeuralNetwork.load(path)
        lstm_layers = [layer for layer in net.layers if type(layer).__name__ == "LSTMLayer"]
        ff_layers = [layer for layer in net.layers if type(layer).__name__ == "FeedForwardLayer"]
        assert len(ff_layers) == 1, path
        for layer_idx, layer in enumerate(lstm_layers):
            for short, attr in GATES.items():
                gate = getattr(layer, attr)
                arrays[f"net{n}_l{layer_idx}_W_{short}"] = np.asarray(gate.weights, dtype=np.float32)
                arrays[f"net{n}_l{layer_idx}_R_{short}"] = np.asarray(
                    gate.recurrent_weights, dtype=np.float32
                )
                arrays[f"net{n}_l{layer_idx}_b_{short}"] = np.asarray(gate.bias, dtype=np.float32)
                if short != "c":
                    arrays[f"net{n}_l{layer_idx}_p_{short}"] = np.asarray(
                        gate.peephole_weights, dtype=np.float32
                    ).flatten()
        ff = ff_layers[0]
        arrays[f"net{n}_ff_W"] = np.asarray(ff.weights, dtype=np.float32)
        arrays[f"net{n}_ff_b"] = np.asarray(ff.bias, dtype=np.float32)

    num_bins = FRAME_SIZE // 2
    filterbank = LogarithmicFilterbank(
        fft_frequencies(num_bins, SAMPLE_RATE),
        num_bands=12, fmin=30, fmax=17000, norm_filters=True,
    )
    arrays["filterbank"] = np.asarray(filterbank, dtype=np.float32)
    meta = {
        "sample_rate": SAMPLE_RATE,
        "frame_size": FRAME_SIZE,
        "hop": HOP,
        "diff_frames": 1,
        "log": "log10(x + 1)",
        "num_nets": len(BEATS_LSTM),
        "num_layers": len(lstm_layers),
        "source": "madmom BEATS_LSTM (models/beats/2016/beats_lstm_[1-8].pkl)",
        "license": "CC BY-NC-SA 4.0",
    }
    arrays["meta"] = np.array(json.dumps(meta))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB, {len(arrays)} arrays)")


if __name__ == "__main__":
    main()
