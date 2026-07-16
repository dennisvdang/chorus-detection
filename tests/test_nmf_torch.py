"""Tests for the torch NMF decomposition against the librosa/sklearn reference."""

import librosa
import numpy as np
import pytest

from pytorch_core.nmf_torch import decompose


def _low_rank_matrix(n_features=40, n_frames=300, rank=4, seed=0):
    """Non-negative matrix with a known low-rank structure plus noise."""
    rng = np.random.RandomState(seed)
    W = rng.exponential(scale=1.0, size=(n_features, rank))
    H = rng.exponential(scale=1.0, size=(rank, n_frames))
    V = W @ H + 0.01 * rng.rand(n_features, n_frames)
    return V.astype(np.float32)


def test_output_shapes_and_nonnegativity():
    V = _low_rank_matrix()
    comps, acts = decompose(V, n_components=4, device="cpu")
    assert comps.shape == (40, 4)
    assert acts.shape == (4, 300)
    assert comps.min() >= 0
    assert acts.min() >= 0


def test_components_sorted_by_peak_row():
    V = _low_rank_matrix(seed=1)
    comps, _ = decompose(V, n_components=4, sort=True, device="cpu")
    peaks = np.argmax(comps, axis=0)
    assert list(peaks) == sorted(peaks)


def test_reconstruction_error_matches_sklearn():
    from sklearn.decomposition import NMF

    V = _low_rank_matrix(seed=2)
    comps, acts = decompose(V, n_components=4, device="cpu")
    err_torch = np.linalg.norm(V - comps @ acts) / np.linalg.norm(V)

    model = NMF(n_components=4, init="nndsvda", max_iter=200)
    W = model.fit_transform(V)
    err_sklearn = np.linalg.norm(V - W @ model.components_) / np.linalg.norm(V)

    # Same init and loss; allow a small margin for solver differences.
    assert err_torch <= err_sklearn * 1.10 + 1e-6


def test_activations_match_librosa_reference():
    V = _low_rank_matrix(seed=3)
    comps_ref, acts_ref = librosa.decompose.decompose(V, n_components=4, sort=True)
    comps, acts = decompose(V, n_components=4, sort=True, device="cpu")

    # Components may converge in a different order/scale; match each torch
    # activation to its best-correlated reference activation.
    used = set()
    for i in range(4):
        corrs = [abs(np.corrcoef(acts[i], acts_ref[j])[0, 1]) for j in range(4)]
        best = int(np.argmax(corrs))
        assert corrs[best] > 0.90, f"component {i}: best corr {corrs[best]:.3f}"
        used.add(best)
    assert len(used) == 4, "torch components collapsed onto the same reference"


def test_handles_zero_rows():
    V = _low_rank_matrix(seed=4)
    V[5, :] = 0.0
    comps, acts = decompose(V, n_components=3, device="cpu")
    assert np.isfinite(comps).all()
    assert np.isfinite(acts).all()
