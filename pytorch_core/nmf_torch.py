"""Torch implementation of NMF decomposition for GPU-accelerated preprocessing.

``librosa.decompose.decompose`` delegates to sklearn's NMF (NNDSVDA init,
coordinate-descent solver, Frobenius loss), which is CPU-only and dominates
per-song preprocessing cost (four fits per song: mel, chroma, tempogram,
MFCC). This module ports the same algorithm — NNDSVDA initialization followed
by HALS coordinate descent — to torch so the fits can run on CUDA, and
reproduces librosa's peak-position component sorting so the activations are a
drop-in replacement for the librosa call.
"""

from typing import Optional, Tuple

import numpy as np
import torch

_EPS = 1e-12


def _nndsvda_init(V: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """NNDSVDA initialization (Boutsidis & Gallopoulos), as in sklearn.

    SVD-based init where negative parts are folded into the dominant
    non-negative pair per component; zeros are filled with the matrix mean
    (the "a" variant), which suits dense spectral data.
    """
    n_features, n_samples = V.shape
    U, S, Vt = torch.linalg.svd(V, full_matrices=False)
    U, S, Vt = U[:, :n_components], S[:n_components], Vt[:n_components]

    W = torch.zeros((n_features, n_components), dtype=V.dtype, device=V.device)
    H = torch.zeros((n_components, n_samples), dtype=V.dtype, device=V.device)

    W[:, 0] = torch.sqrt(S[0]) * U[:, 0].abs()
    H[0] = torch.sqrt(S[0]) * Vt[0].abs()

    for j in range(1, n_components):
        x, y = U[:, j], Vt[j]
        x_p, y_p = x.clamp(min=0), y.clamp(min=0)
        x_n, y_n = (-x).clamp(min=0), (-y).clamp(min=0)
        x_p_nrm, y_p_nrm = x_p.norm(), y_p.norm()
        x_n_nrm, y_n_nrm = x_n.norm(), y_n.norm()
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
        if m_p > m_n:
            u = x_p / x_p_nrm.clamp(min=_EPS)
            v = y_p / y_p_nrm.clamp(min=_EPS)
            sigma = m_p
        else:
            u = x_n / x_n_nrm.clamp(min=_EPS)
            v = y_n / y_n_nrm.clamp(min=_EPS)
            sigma = m_n
        lbd = torch.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j] = lbd * v

    avg = V.mean()
    W[W < _EPS] = avg
    H[H < _EPS] = avg
    return W, H


def _hals(V: torch.Tensor, W: torch.Tensor, H: torch.Tensor,
          max_iter: int = 200, tol: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """HALS coordinate descent for Frobenius-loss NMF.

    Cyclic per-component updates equivalent to sklearn's ``solver="cd"``;
    stops when the relative reconstruction error changes by less than ``tol``
    between 10-iteration checkpoints.
    """
    v_norm = V.norm().clamp(min=_EPS)
    err_prev: Optional[torch.Tensor] = None

    for iteration in range(max_iter):
        HHt = H @ H.T
        VHt = V @ H.T
        for t in range(W.shape[1]):
            denom = HHt[t, t].clamp(min=_EPS)
            W[:, t] = (W[:, t] + (VHt[:, t] - W @ HHt[:, t]) / denom).clamp(min=0)

        WtW = W.T @ W
        WtV = W.T @ V
        for t in range(H.shape[0]):
            denom = WtW[t, t].clamp(min=_EPS)
            H[t] = (H[t] + (WtV[t] - WtW[t] @ H) / denom).clamp(min=0)

        if tol > 0 and (iteration + 1) % 10 == 0:
            err = (V - W @ H).norm() / v_norm
            if err_prev is not None and (err_prev - err).abs() < tol * err_prev:
                break
            err_prev = err

    return W, H


def decompose(S: np.ndarray, n_components: int, sort: bool = True,
              device: Optional[str] = None, max_iter: int = 200,
              tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Drop-in replacement for ``librosa.decompose.decompose(S, n_components, sort=...)``.

    Args:
        S: Non-negative feature matrix, shape (n_features, n_frames).
        n_components: Number of NMF components.
        sort: Sort components by peak row index (librosa's convention),
            reordering activations to match.
        device: Torch device for the fit ("cuda", "cpu", ...). Defaults to
            CUDA when available.
        max_iter: Maximum HALS iterations (sklearn's default is 200).
        tol: Relative-error convergence tolerance.

    Returns:
        Tuple of (components [n_features, n_components],
        activations [n_components, n_frames]) as float32 numpy arrays.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    V = torch.as_tensor(np.ascontiguousarray(S), dtype=torch.float32, device=device)
    V = V.clamp(min=0)

    W, H = _nndsvda_init(V, n_components)
    W, H = _hals(V, W, H, max_iter=max_iter, tol=tol)

    comps = W.cpu().numpy().astype(np.float32)
    acts = H.cpu().numpy().astype(np.float32)

    if sort:
        order = np.argsort(np.argmax(comps, axis=0), kind="stable")
        comps = comps[:, order]
        acts = acts[order]

    return comps, acts
