import numpy as np

from pytorch_core.decoding import viterbi_chorus


def test_clean_block_recovered():
    probs = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
    out = viterbi_chorus(probs, switch_penalty=1.0, min_bars=2)
    assert out.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


def test_single_meter_spike_suppressed_by_min_bars():
    probs = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    out = viterbi_chorus(probs, switch_penalty=0.5, min_bars=4)
    assert out.sum() == 0  # 1-meter spike is shorter than min_bars


def test_high_switch_penalty_prevents_fragmentation():
    probs = np.array([0.9, 0.4, 0.9, 0.4, 0.9, 0.9, 0.9, 0.9])
    out = viterbi_chorus(probs, switch_penalty=5.0, min_bars=2)
    assert out.tolist() == [1, 1, 1, 1, 1, 1, 1, 1]


def test_output_length_and_dtype():
    out = viterbi_chorus(np.array([0.2, 0.8, 0.8]), min_bars=1)
    assert out.shape == (3,) and out.dtype == int
