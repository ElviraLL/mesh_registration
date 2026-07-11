"""Similarity-transform estimation and trimmed scaled ICP."""

import numpy as np
from scipy.spatial import cKDTree


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity transform mapping src -> dst (Umeyama 1991).

    Returns (s, R, t) with dst ~= s * R @ src + t.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / len(src)
        s = np.trace(np.diag(D) @ S) / var_s
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def compose(s, R, t):
    """4x4 homogeneous matrix for x -> s * R @ x + t."""
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def apply_srt(v, s, R, t):
    return s * (v @ R.T) + t


def trimmed_icp(
    src,
    target_tree,
    target_pts,
    s,
    R,
    t,
    iters=60,
    trim=0.7,
    with_scale=True,
    crop_margin=0.06,
    tol=1e-7,
):
    """Trimmed ICP estimating a similarity transform.

    src           : (N, 3) source sample points (garment).
    target_tree   : cKDTree over target_pts (reference surface samples).
    target_pts    : (M, 3) reference sample points.
    s, R, t       : initial similarity transform.
    trim          : fraction of best-matching pairs kept each iteration.
    crop_margin   : target points outside the transformed source's bounding
                    box (expanded by this margin) are ignored, so the garment
                    only registers against its own region of the avatar.

    Returns (s, R, t, err) where err is the mean kept-pair distance.
    """
    prev_err = np.inf
    err = np.inf
    for _ in range(iters):
        cur = apply_srt(src, s, R, t)

        lo = cur.min(axis=0) - crop_margin
        hi = cur.max(axis=0) + crop_margin
        box = np.all((target_pts >= lo) & (target_pts <= hi), axis=1)
        if box.sum() > 500:
            tree = cKDTree(target_pts[box])
            tpts = target_pts[box]
        else:
            tree, tpts = target_tree, target_pts

        d, idx = tree.query(cur, workers=-1)
        keep = np.argsort(d)[: max(int(len(d) * trim), 100)]
        err = d[keep].mean()
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        s, R, t = umeyama(src[keep], tpts[idx[keep]], with_scale=with_scale)
    return s, R, t, err


def yaw_matrix(degrees):
    a = np.radians(degrees)
    c, si = np.cos(a), np.sin(a)
    return np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])
