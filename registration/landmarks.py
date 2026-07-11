"""Geometric landmark extraction for y-up, T-pose humanoid meshes.

All heuristics assume the conventions of the test data: y is up, the
character faces +/-z, arms stretched out along x (T-pose), and every mesh is
independently normalized to roughly a unit cube centered at the origin.
"""

import numpy as np


def _slice(v, y, thickness=0.012):
    """Vertices whose y coordinate lies within a horizontal slab."""
    mask = np.abs(v[:, 1] - y) < thickness
    return v[mask]


def _x_clusters(points, gap=0.04):
    """Cluster points along x by splitting at gaps; returns list of arrays."""
    if len(points) == 0:
        return []
    order = np.argsort(points[:, 0])
    pts = points[order]
    splits = np.where(np.diff(pts[:, 0]) > gap)[0]
    return np.split(pts, splits + 1)


def leg_split_y(v, y_lo, y_hi, steps=80):
    """Highest y in [y_lo, y_hi] where the cross-section splits into two legs.

    Scans slices from bottom to top; the crotch is the top of the contiguous
    run of two-cluster slices that starts in the leg region.
    """
    crotch = None
    ys = np.linspace(y_lo, y_hi, steps)
    run = 0
    for y in ys:
        sl = _slice(v, y)
        if len(sl) < 8:
            continue
        n = len([c for c in _x_clusters(sl) if len(c) >= 4])
        if n >= 2:
            run += 1
            crotch = y
        else:
            # Require a real run of two-legged slices before stopping so a
            # single noisy slice near the bottom doesn't end the scan.
            if run >= 5:
                break
    return crotch


def avatar_landmarks(v):
    """Landmarks of the reference avatar (or a full-body mesh).

    Returns a dict of named scalars/points. `v` is the (N, 3) vertex array.
    """
    lm = {}
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    height = y_max - y_min
    lm["foot_bottom_y"] = y_min
    lm["head_top"] = v[v[:, 1].argmax()]
    lm["height"] = height

    # Hand tips: extreme x. In a T-pose these are the farthest points along x.
    lm["hand_r"] = v[v[:, 0].argmax()]  # +x side
    lm["hand_l"] = v[v[:, 0].argmin()]  # -x side
    lm["hand_span"] = lm["hand_r"][0] - lm["hand_l"][0]
    lm["arm_y"] = 0.5 * (lm["hand_r"][1] + lm["hand_l"][1])

    # Crotch: where the silhouette splits into two legs.
    crotch_y = leg_split_y(v, y_min + 0.02 * height, y_max - 0.3 * height)
    if crotch_y is None:
        crotch_y = y_min + 0.45 * height
    lm["crotch_y"] = crotch_y

    # Ankle: a bit above the sole. Foot geometry (long z extent) sits below.
    lm["ankle_y"] = y_min + 0.045 * height

    # Feet: cluster everything below the ankle into left/right foot.
    feet = v[v[:, 1] < y_min + 0.08 * height]
    clusters = [c for c in _x_clusters(feet, gap=0.05) if len(c) >= 10]
    clusters.sort(key=lambda c: c[:, 0].mean())
    if len(clusters) >= 2:
        lm["foot_l_center"] = clusters[0].mean(axis=0)
        lm["foot_r_center"] = clusters[-1].mean(axis=0)
        lm["foot_z_extent"] = max(
            c[:, 2].max() - c[:, 2].min() for c in (clusters[0], clusters[-1])
        )
        lm["foot_z_front"] = max(clusters[0][:, 2].max(), clusters[-1][:, 2].max())
    # Torso center in z at chest height (between crotch and arms).
    chest_y = 0.5 * (crotch_y + lm["arm_y"])
    torso = v[(np.abs(v[:, 1] - chest_y) < 0.08 * height) & (np.abs(v[:, 0]) < 0.2)]
    lm["torso_z_center"] = torso[:, 2].mean() if len(torso) else 0.0

    # Neck: narrowest slice (x width) between the arms and the head.
    best_w, neck_y = np.inf, lm["arm_y"] + 0.1 * height
    for y in np.linspace(lm["arm_y"] + 0.02 * height, y_max - 0.05 * height, 40):
        sl = _slice(v, y)
        if len(sl) < 8:
            continue
        w = sl[:, 0].max() - sl[:, 0].min()
        if w < best_w:
            best_w, neck_y = w, y
    lm["neck_y"] = neck_y

    head = v[v[:, 1] > neck_y]
    if len(head):
        lm["head_center"] = head.mean(axis=0)
        lm["head_x_extent"] = head[:, 0].max() - head[:, 0].min()
        lm["head_z_extent"] = head[:, 2].max() - head[:, 2].min()
    return lm


def bottoms_landmarks(v):
    """Landmarks for a pair of pants in its own frame: crotch and hem."""
    lm = {}
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    extent = y_max - y_min
    lm["waist_top_y"] = y_max
    lm["hem_y"] = y_min
    crotch_y = leg_split_y(v, y_min + 0.02 * extent, y_max - 0.05 * extent)
    if crotch_y is None:
        crotch_y = y_min + 0.5 * extent
    lm["crotch_y"] = crotch_y
    lm["center_xz"] = np.array([v[:, 0].mean(), v[:, 2].mean()])
    return lm


def shoes_landmarks(v):
    """Landmarks for a pair of shoes: per-foot centers, sole, toe direction."""
    lm = {}
    clusters = [c for c in _x_clusters(v, gap=0.05) if len(c) >= 50]
    clusters.sort(key=lambda c: c[:, 0].mean())
    lm["sole_y"] = v[:, 1].min()
    if len(clusters) >= 2:
        left, right = clusters[0], clusters[-1]
    else:  # fell back: split at x median
        left = v[v[:, 0] < np.median(v[:, 0])]
        right = v[v[:, 0] >= np.median(v[:, 0])]
    lm["foot_l_center"] = left.mean(axis=0)
    lm["foot_r_center"] = right.mean(axis=0)
    lm["z_extent"] = max(
        left[:, 2].max() - left[:, 2].min(), right[:, 2].max() - right[:, 2].min()
    )
    return lm


def tops_landmarks(v):
    """Landmarks for a T-pose top: sleeve tips and their height."""
    lm = {}
    lm["sleeve_r"] = v[v[:, 0].argmax()]
    lm["sleeve_l"] = v[v[:, 0].argmin()]
    lm["span"] = lm["sleeve_r"][0] - lm["sleeve_l"][0]
    # Average y of the outermost 2% of vertices on each side = sleeve axis.
    x_lo = np.quantile(v[:, 0], 0.02)
    x_hi = np.quantile(v[:, 0], 0.98)
    tips = v[(v[:, 0] <= x_lo) | (v[:, 0] >= x_hi)]
    lm["sleeve_y"] = tips[:, 1].mean()
    lm["center_z"] = v[:, 2].mean()
    return lm
