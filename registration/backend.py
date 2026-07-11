"""Optional GPU acceleration (PyTorch); transparent CPU fallback.

Set REG_DEVICE=cuda / cpu / auto (default auto) to control it. "auto" uses
CUDA when torch sees a GPU. The torch code paths are device-agnostic, so
they can also be exercised on CPU with REG_DEVICE=torch-cpu (mainly for
testing - scipy's cKDTree is faster than torch on CPU).
"""

import os

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _mode():
    want = os.environ.get("REG_DEVICE", "auto")
    if want == "cpu" or torch is None:
        return None
    if want == "torch-cpu":
        return "cpu"
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        return "cuda"
    return None


class NNIndex:
    """Nearest-neighbor index over a fixed point set.

    GPU: chunked brute-force distance minimization (exact).
    CPU: scipy cKDTree.
    """

    def __init__(self, pts):
        self.pts = np.ascontiguousarray(pts, dtype=np.float64)
        self.device = _mode()
        if self.device:
            self._t = torch.as_tensor(
                self.pts, dtype=torch.float32, device=self.device
            )
            self._t_sq = (self._t ** 2).sum(dim=1)
            self.tree = None
        else:
            from scipy.spatial import cKDTree

            self.tree = cKDTree(self.pts)

    def query(self, q, chunk=4096):
        """Distances and indices of the nearest point for each query."""
        if not self.device:
            return self.tree.query(q, workers=-1)
        qt = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        ds, idxs = [], []
        for i in range(0, len(qt), chunk):
            blk = qt[i : i + chunk]
            d2 = (
                (blk ** 2).sum(dim=1)[:, None]
                + self._t_sq[None, :]
                - 2.0 * blk @ self._t.T
            )
            dmin, imin = torch.min(d2, dim=1)
            ds.append(dmin.clamp_min_(0).sqrt_())
            idxs.append(imin)
        return (
            torch.cat(ds).double().cpu().numpy(),
            torch.cat(idxs).cpu().numpy(),
        )


def ransac_correspondences(src_down, src_feat, dst_down, dst_feat, dist,
                           iters=150000, seed=None):
    """GPU RANSAC over FPFH correspondences; returns (T 4x4, n_inliers).

    Mutual-nearest-neighbor feature matches are sampled in 3-point minimal
    sets; each hypothesis is solved in closed form (batched Umeyama with
    scaling) and scored by how many putative matches it brings within
    `dist`. Returns (None, 0) when there is nothing usable. Only called on
    the torch device; the CPU path uses Open3D's RANSAC instead.
    """
    device = _mode()
    sf = torch.as_tensor(np.ascontiguousarray(src_feat), dtype=torch.float32, device=device)
    df = torch.as_tensor(np.ascontiguousarray(dst_feat), dtype=torch.float32, device=device)
    sp = torch.as_tensor(np.ascontiguousarray(src_down), dtype=torch.float32, device=device)
    dp = torch.as_tensor(np.ascontiguousarray(dst_down), dtype=torch.float32, device=device)

    # Mutual nearest neighbors in feature space (chunked over source rows).
    fwd = []
    for i in range(0, len(sf), 2048):
        fwd.append(torch.cdist(sf[i : i + 2048], df).argmin(dim=1))
    fwd = torch.cat(fwd)
    bwd = []
    for i in range(0, len(df), 2048):
        bwd.append(torch.cdist(df[i : i + 2048], sf).argmin(dim=1))
    bwd = torch.cat(bwd)
    src_idx = torch.arange(len(sf), device=device)
    mutual = bwd[fwd] == src_idx
    if mutual.sum() < 10:
        mutual = torch.ones_like(mutual)  # fall back to forward matches
    P = sp[mutual]                      # (K, 3) matched source points
    Q = dp[fwd[mutual]]                 # (K, 3) matched target points
    K = len(P)
    if K < 4:
        return None, 0

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    H = min(iters, 200000)
    sel = torch.randint(0, K, (H, 3), device=device, generator=gen)
    p = P[sel]                          # (H, 3, 3)
    q = Q[sel]

    # Batched Umeyama with scale.
    mu_p = p.mean(dim=1, keepdim=True)
    mu_q = q.mean(dim=1, keepdim=True)
    pc, qc = p - mu_p, q - mu_q
    cov = qc.transpose(1, 2) @ pc / 3.0
    U, D, Vh = torch.linalg.svd(cov)
    det = torch.linalg.det(U @ Vh)
    S = torch.ones(H, 3, device=device)
    S[:, 2] = torch.sign(det)
    R = U @ (S[:, :, None] * Vh)
    var_p = (pc ** 2).sum(dim=(1, 2)) / 3.0
    s = (D * S).sum(dim=1) / var_p.clamp_min(1e-12)
    t = mu_q.squeeze(1) - s[:, None] * (R @ mu_p.transpose(1, 2)).squeeze(2)

    ok = (s > 1e-3) & torch.isfinite(s)
    # Inlier counting on (a subsample of) the putative matches, chunked
    # over hypotheses to bound memory.
    if K > 2000:
        sub = torch.randint(0, K, (2000,), device=device, generator=gen)
        Ps, Qs = P[sub], Q[sub]
    else:
        Ps, Qs = P, Q
    best_n, best_h = 0, -1
    for i in range(0, H, 4096):
        Ri, si, ti = R[i : i + 4096], s[i : i + 4096], t[i : i + 4096]
        proj = si[:, None, None] * (Ps @ Ri.transpose(1, 2)) + ti[:, None, :]
        n = ((proj - Qs).norm(dim=2) < dist).sum(dim=1)
        n = torch.where(ok[i : i + 4096], n, torch.zeros_like(n))
        ni, hi = torch.max(n, dim=0)
        if int(ni) > best_n:
            best_n, best_h = int(ni), i + int(hi)
    if best_h < 0 or best_n < 10:
        return None, 0

    # Refit on the best hypothesis' inliers (closed form, full match set).
    Rb, sb, tb = R[best_h], s[best_h], t[best_h]
    proj = sb * (P @ Rb.T) + tb
    inl = (proj - Q).norm(dim=1) < dist
    if inl.sum() >= 4:
        from .icp import umeyama

        s_, R_, t_ = umeyama(
            P[inl].cpu().numpy().astype(np.float64),
            Q[inl].cpu().numpy().astype(np.float64),
        )
    else:
        s_, R_, t_ = float(sb), Rb.cpu().numpy(), tb.cpu().numpy()
    T = np.eye(4)
    T[:3, :3] = s_ * np.asarray(R_)
    T[:3, 3] = np.asarray(t_)
    return T, best_n


def use_torch():
    return _mode() is not None
