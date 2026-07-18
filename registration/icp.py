"""Similarity-transform estimation and trimmed scaled ICP."""

import numpy as np

from .backend import NNIndex


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
    scale_bounds=(0.6, 1.6),
    src_nrm=None,
    tgt_nrm=None,
    with_rot=True,
):
    """Trimmed ICP estimating a similarity transform.

    src           : (N, 3) source sample points (garment).
    target_tree   : NNIndex over target_pts (reference surface samples).
    target_pts    : (M, 3) reference sample points.
    s, R, t       : initial similarity transform.
    trim          : fraction of best-matching pairs kept each iteration.
    crop_margin   : target points outside the transformed source's bounding
                    box (expanded by this margin) are ignored, so the garment
                    only registers against its own region of the avatar.
    src_nrm/tgt_nrm : optional unit normals for source samples / target
                    points. When given, correspondences whose normals face
                    opposite directions are rejected, which stops the fit
                    from settling into flipped orientations or hugging a
                    surface from the wrong side.
    scale_bounds  : allowed scale range relative to the initial scale.
                    Unbounded scale estimation collapses the source onto the
                    target surface (a shrunken object always has a lower
                    absolute point-to-surface error), so the scale may only
                    move this far from the initialization.

    Returns (s, R, t, err) where err is the mean kept-pair distance.
    """
    s_lo, s_hi = scale_bounds[0] * s, scale_bounds[1] * s
    prev_err = np.inf
    err = np.inf
    for _ in range(iters):
        cur = apply_srt(src, s, R, t)

        lo = cur.min(axis=0) - crop_margin
        hi = cur.max(axis=0) + crop_margin
        box = np.all((target_pts >= lo) & (target_pts <= hi), axis=1)
        if box.sum() > 500:
            tree = NNIndex(target_pts[box])
            tpts = target_pts[box]
            tnrm = tgt_nrm[box] if tgt_nrm is not None else None
        else:
            tree, tpts, tnrm = target_tree, target_pts, tgt_nrm

        d, idx = tree.query(cur)
        if src_nrm is not None and tnrm is not None:
            agree = ((src_nrm @ R.T) * tnrm[idx]).sum(axis=1) > 0.0
            if agree.sum() > 300:
                d = np.where(agree, d, np.inf)
        keep = np.argsort(d)[: max(int(len(d) * trim), 100)]
        keep = keep[np.isfinite(d[keep])]
        if len(keep) < 100:
            keep = np.argsort(np.where(np.isfinite(d), d, np.inf))[:100]
        err = d[keep].mean()
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        if with_rot:
            s, R, t = umeyama(src[keep], tpts[idx[keep]], with_scale=with_scale)
        else:
            # Rotation locked to the initialization: closed-form s, t for
            # dst ~= s * (R0 @ src) + t.
            xs = src[keep] @ R.T
            xd = tpts[idx[keep]]
            mu_s, mu_d = xs.mean(axis=0), xd.mean(axis=0)
            if with_scale:
                var = ((xs - mu_s) ** 2).sum()
                s = float(((xs - mu_s) * (xd - mu_d)).sum() / max(var, 1e-12))
            t = mu_d - s * mu_s
        if with_scale and not (s_lo <= s <= s_hi):
            s = float(np.clip(s, s_lo, s_hi))
            if with_rot:
                # Re-solve rotation and translation for the clamped scale.
                _, R, t = umeyama(s * src[keep], tpts[idx[keep]], with_scale=False)
                R = R.copy()
                t = t.copy()
            else:
                xs = src[keep] @ R.T
                t = tpts[idx[keep]].mean(axis=0) - s * xs.mean(axis=0)
    return s, R, t, err


def fit_score(src_pts, s, R, t, ref_tree):
    """Scale-fair fit quality for comparing competing registrations.

    Mean UNtrimmed point-to-reference distance, normalized by scale.
    - Trimmed error hides the part of the garment that sticks out of a
      wrong region, so trimming cannot be used for selection.
    - Absolute error favors collapsed (shrunken) fits; relative trimmed
      error favors inflated fits onto large smooth regions. The untrimmed
      relative error exposes both: a garment registered to its true
      counterpart region lies on the reference over its WHOLE surface.
    """
    d, _ = ref_tree.query(apply_srt(src_pts, s, R, t))
    return d.mean() / s


def oriented_score(src_pts, src_nrm, s, R, t, ref_tree, ref_nrm):
    """Normal-aware absolute fit error, for choosing between orientations.

    Each point's distance is weighted by (2 - n_src.n_ref): aligned normals
    weigh 1, opposing normals weigh up to 3, so a flipped fit (identical in
    pure point distance for near-symmetric shapes) scores ~3x worse.
    """
    d, idx = ref_tree.query(apply_srt(src_pts, s, R, t))
    dot = ((src_nrm @ R.T) * ref_nrm[idx]).sum(axis=1)
    return float((d * (2.0 - np.clip(dot, -1.0, 1.0))).mean())


def rot180(axis):
    """180-degree rotation matrix about x, y or z."""
    R = -np.eye(3)
    R[axis, axis] = 1.0
    return R


def yaw_matrix(degrees):
    a = np.radians(degrees)
    c, si = np.cos(a), np.sin(a)
    return np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])
