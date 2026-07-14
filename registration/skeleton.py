"""Skeleton estimation for the registered avatar and its garments.

Usage:
    python -m registration.skeleton --data data --out output

Requires a previous registration run (reads output/registered/). Produces
output/skeletons.json and a preview.

Body skeleton: joints are estimated geometrically from the registered bald
body (fallback: the reference avatar) using the same y-up T-pose analysis
as the landmark extractor - leg/arm centroid lines from horizontal slices,
shoulders from the torso width at arm height, spine from torso slice
centroids.

Garment skeletons: every garment is already registered into the reference
frame, so each body bone can be tested for support (garment surface within
a margin of the bone segment). A garment's skeleton is the supported subset
of body bones, with two corrections estimated from the garment's own
geometry:
- lateral: each joint is pulled toward the centroid of the garment
  vertices that support its bones (captures a garment hanging off-axis);
- terminal: end joints (wrists, ankles, head, foot tips) are moved to the
  garment's own extent along the bone axis (captures sleeve/pant length
  mismatch vs. the body skeleton).

The per-joint offset between the garment skeleton and the body skeleton is
exactly the mismatch to correct when binding the garment to the body rig.
Per-vertex nearest-bone assignments and soft weights are saved alongside
for skinning.
"""

import argparse
import json
import os

import numpy as np
import trimesh

from .landmarks import avatar_landmarks, _slice, _x_clusters

BONES = [
    ("pelvis", "spine"), ("spine", "chest"), ("chest", "neck"),
    ("neck", "head"),
    ("chest", "shoulder_l"), ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("chest", "shoulder_r"), ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
    ("pelvis", "hip_l"), ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
    ("ankle_l", "foot_l"),
    ("pelvis", "hip_r"), ("hip_r", "knee_r"), ("knee_r", "ankle_r"),
    ("ankle_r", "foot_r"),
]

TERMINAL = {"wrist_l", "wrist_r", "ankle_l", "ankle_r", "head",
            "foot_l", "foot_r"}


def _side_centroid(v, y, side, thickness=0.02):
    """Centroid of the left (side=-1) or right (side=+1) cluster at height y."""
    sl = _slice(v, y, thickness)
    clusters = [c for c in _x_clusters(sl) if len(c) >= 4]
    if not clusters:
        return None
    on_side = [c for c in clusters if np.sign(c[:, 0].mean()) == side]
    if not on_side:
        return None
    # The cluster with the most points on that side is the limb/torso.
    return max(on_side, key=len).mean(axis=0)


def estimate_body_skeleton(v):
    """Estimate humanoid joints from a y-up T-pose mesh; returns {name: xyz}."""
    lm = avatar_landmarks(v)
    y0, H = lm["foot_bottom_y"], lm["height"]
    crotch = lm["crotch_y"]
    joints = {}

    # Legs: centroid of each side's cluster at anatomical heights.
    ankle_y = y0 + 0.05 * H
    hip_y = crotch + 0.04 * H
    knee_y = 0.5 * (ankle_y + hip_y)
    for side, tag in ((-1, "l"), (1, "r")):
        for name, y in (("hip", hip_y), ("knee", knee_y), ("ankle", ankle_y)):
            c = _side_centroid(v, y, side)
            if c is None:  # single cluster at hip height: split by x sign
                sl = _slice(v, y, 0.02)
                sl = sl[np.sign(sl[:, 0]) == side]
                c = sl.mean(axis=0) if len(sl) else np.array([0.1 * side, y, 0])
            joints[f"{name}_{tag}"] = c
        # Foot tip: front-most point of that side's foot.
        feet = v[(v[:, 1] < y0 + 0.08 * H) & (np.sign(v[:, 0]) == side)]
        joints[f"foot_{tag}"] = (
            feet[feet[:, 2].argmax()] if len(feet) else joints[f"ankle_{tag}"]
        )

    joints["pelvis"] = 0.5 * (joints["hip_l"] + joints["hip_r"])
    arm_y = lm["arm_y"]

    # Spine from torso slice centroids.
    for name, y in (
        ("spine", crotch + 0.18 * H),
        ("chest", arm_y - 0.06 * H),
        ("neck", lm["neck_y"]),
    ):
        sl = _slice(v, y, 0.02)
        sl = sl[np.abs(sl[:, 0]) < 0.25]  # torso only, not the arms
        joints[name] = (
            sl.mean(axis=0) if len(sl) else np.array([0.0, y, 0.0])
        )
    joints["head"] = lm.get("head_center", np.array([0, lm["head_top"][1], 0]))

    # Arms: shoulder at the torso edge at arm height, wrist near the hand
    # tip, elbow midway.
    chest_sl = _slice(v, arm_y - 0.08 * H, 0.02)
    torso_w = (
        np.percentile(np.abs(chest_sl[:, 0]), 95) if len(chest_sl) else 0.15
    )
    for tag, hand in (("l", lm["hand_l"]), ("r", lm["hand_r"])):
        sgn = -1.0 if tag == "l" else 1.0
        shoulder = np.array([sgn * torso_w, arm_y, joints["chest"][2]])
        wrist = hand + (shoulder - hand) * 0.08
        joints[f"shoulder_{tag}"] = shoulder
        joints[f"wrist_{tag}"] = wrist
        joints[f"elbow_{tag}"] = 0.5 * (shoulder + wrist)

    return {k: np.asarray(p, dtype=float) for k, p in joints.items()}


def _point_segment_dist(p, a, b):
    """Distances from points p to segment ab, plus the projection parameter."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(p - a, axis=1), np.zeros(len(p))
    t = np.clip((p - a) @ ab / L2, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(p - closest, axis=1), t


def garment_skeleton(gv, joints, margin=0.07, min_support=200):
    """Fit the supported subset of body bones to a registered garment.

    Returns (garment_joints, kept_bones, bone_id_per_vertex, weights).
    """
    dists = np.full((len(BONES), len(gv)), np.inf)
    ts = np.zeros((len(BONES), len(gv)))
    for i, (a, b) in enumerate(BONES):
        dists[i], ts[i] = _point_segment_dist(gv, joints[a], joints[b])

    support = (dists < margin).sum(axis=1)
    kept = [i for i, n in enumerate(support) if n >= min_support]
    if not kept:
        kept = [int(np.argmin(dists.min(axis=1)))]

    g_joints = {}
    for i in kept:
        a, b = BONES[i]
        near = dists[i] < margin
        pts = gv[near]
        axis = joints[b] - joints[a]
        L = np.linalg.norm(axis)
        axis = axis / max(L, 1e-9)

        for name, anchor in ((a, joints[a]), (b, joints[b])):
            # Lateral correction: pull the joint toward the local centroid
            # of its supporting surface, in the plane perpendicular to the
            # bone. Average across bones sharing the joint.
            proj_t = (pts - joints[a]) @ axis
            end_t = 0.0 if name == a else L
            local = pts[np.abs(proj_t - end_t) < 0.35 * L]
            if len(local) < 20:
                candidate = anchor.copy()
            else:
                centroid = local.mean(axis=0)
                offset = centroid - anchor
                offset -= (offset @ axis) * axis  # keep the bone length
                candidate = anchor + offset
            if name in g_joints:
                g_joints[name] = 0.5 * (g_joints[name] + candidate)
            else:
                g_joints[name] = candidate

        # Terminal correction: move end joints to the garment's own extent
        # along the bone (sleeve/pant/shell length).
        if b in TERMINAL and len(pts):
            reach = (pts - joints[a]) @ axis
            far = np.quantile(reach, 0.98)
            g_joints[b] = g_joints[b] + axis * (far - L)

    kept_dists = dists[kept]
    nearest = np.argmin(kept_dists, axis=0)
    bone_id = np.array([kept[i] for i in nearest])
    # Soft weights over the two nearest kept bones (inverse distance).
    order = np.argsort(kept_dists, axis=0)
    d1 = kept_dists[order[0], np.arange(len(gv))]
    d2 = (
        kept_dists[order[1], np.arange(len(gv))]
        if len(kept) > 1 else np.full(len(gv), np.inf)
    )
    w1 = np.where(np.isfinite(d2), d2 / np.maximum(d1 + d2, 1e-9), 1.0)
    weights = np.stack(
        [np.array([kept[i] for i in order[0]]),
         np.array([kept[i] for i in order[1 if len(kept) > 1 else 0]]),
         w1, 1.0 - w1], axis=1,
    )
    return g_joints, kept, bone_id, weights


def run(data_dir, out_dir, preview=True):
    reg_dir = os.path.join(out_dir, "registered")
    body_path = None
    for cand in ("body_bald_registered.glb", "body_registered.glb"):
        p = os.path.join(reg_dir, cand)
        if os.path.exists(p):
            body_path = p
            break
    if body_path is None:
        raise SystemExit(
            f"no registered body found in {reg_dir}; run the registration "
            f"first (python -m registration.register)"
        )
    body = trimesh.load(body_path, process=False, force="mesh")
    joints = estimate_body_skeleton(body.vertices)
    print(f"[ok]   body skeleton: {len(joints)} joints from "
          f"{os.path.basename(body_path)}")

    result = {
        "joints": {k: v.tolist() for k, v in joints.items()},
        "bones": BONES,
        "garments": {},
    }
    os.makedirs(os.path.join(out_dir, "skinning"), exist_ok=True)
    garments = {}
    for f in sorted(os.listdir(reg_dir)):
        if not f.endswith("_registered.glb") or f.startswith("body"):
            continue
        name = f[: -len("_registered.glb")]
        mesh = trimesh.load(os.path.join(reg_dir, f), process=False,
                            force="mesh")
        g_joints, kept, bone_id, weights = garment_skeleton(
            mesh.vertices, joints
        )
        mismatch = {
            k: float(np.linalg.norm(g_joints[k] - joints[k]))
            for k in g_joints
        }
        result["garments"][name] = {
            "joints": {k: v.tolist() for k, v in g_joints.items()},
            "bones": [BONES[i] for i in kept],
            "joint_offset_from_body": mismatch,
        }
        np.savez_compressed(
            os.path.join(out_dir, "skinning", f"{name}_weights.npz"),
            bone_id=bone_id, weights=weights, bones=np.array(BONES),
        )
        garments[name] = (mesh.vertices, g_joints, kept)
        worst = max(mismatch, key=mismatch.get)
        print(f"[ok]   {name}: {len(kept)} bones, max joint offset "
              f"{mismatch[worst]:.3f} at {worst}")

    with open(os.path.join(out_dir, "skeletons.json"), "w") as fh:
        json.dump(result, fh, indent=1)

    if preview:
        _render(out_dir, body.vertices, joints, garments)
    return result


def _render(out_dir, bv, joints, garments):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + len(garments)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    axes = np.atleast_1d(axes)

    def draw(ax, pts, js, bones, title):
        ax.scatter(pts[:, 0], pts[:, 1], s=0.05, c="lightgray")
        for a, b in bones:
            if a in js and b in js:
                ax.plot([js[a][0], js[b][0]], [js[a][1], js[b][1]],
                        "o-", color="tab:red", ms=3, lw=2)
        ax.set_aspect("equal")
        ax.set_title(title)

    draw(axes[0], bv, joints, BONES, "body skeleton")
    for ax, (name, (gv, g_joints, kept)) in zip(axes[1:], garments.items()):
        draw(ax, gv, g_joints, [BONES[i] for i in kept], name)
    plt.tight_layout()
    out = os.path.join(out_dir, "previews", "skeletons.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=80)
    plt.close(fig)
    print(f"[ok]   preview: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
    run(args.data, args.out, preview=not args.no_preview)


if __name__ == "__main__":
    main()
