"""Register separately generated garment meshes onto the reference avatar.

Usage:
    python -m registration.register --data data --out output

Each garment gets a landmark-based similarity initialization, then a trimmed
scaled ICP refinement against the reference "one piece" avatar (which has the
same garments baked in). Several initializations (yaw flips, scale
perturbations) are tried and the one with the lowest ICP error wins.
"""

import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .landmarks import (
    avatar_landmarks,
    bottoms_landmarks,
    shoes_landmarks,
    tops_landmarks,
)
from .icp import trimmed_icp, compose, yaw_matrix

REFERENCE = "3d_generated_mesh_one_piece_avatar_body_processed.glb"

GARMENTS = {
    "body": "3d_generated_mesh_body_processed.glb",
    "body_bald": "3d_generated_mesh_body_bald_processed.glb",
    "tops": "3d_generated_mesh_tops_processed.glb",
    "bottoms": "3d_generated_mesh_bottoms_processed.glb",
    "shoes": "3d_generated_mesh_shoes_processed.glb",
    "head_accessories": "3d_generated_mesh_head_accessories_processed.glb",
}


def load_mesh(path):
    """Load a single-geometry GLB; returns (scene, geometry).

    The geometry is used both for registration and for baking the result, so
    working on the scene's own geometry object keeps vertex order consistent.
    """
    scene = trimesh.load(path, process=False)
    geoms = list(scene.geometry.values())
    assert len(geoms) == 1, f"expected a single geometry in {path}"
    return scene, geoms[0]


def sample(mesh, n):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts)


def _init_body(gv, ref_lm):
    """Full body -> full avatar: match height and bounding-box anchor points."""
    lm = avatar_landmarks(gv)
    s = ref_lm["height"] / lm["height"]
    src = np.array([lm["head_top"], lm["hand_l"], lm["hand_r"]])
    dst = np.array([ref_lm["head_top"], ref_lm["hand_l"], ref_lm["hand_r"]])
    # Scale from height is more reliable than from 3 points; solve t only.
    t = (dst - s * src).mean(axis=0)
    # Pin the feet to the floor.
    t[1] += ref_lm["foot_bottom_y"] - (s * gv[:, 1].min() + t[1])
    return [(s, np.eye(3), t)]


def _init_tops(gv, ref_lm):
    lm = tops_landmarks(gv)
    s = ref_lm["hand_span"] / lm["span"] * 0.98
    center_x = 0.5 * (lm["sleeve_l"][0] + lm["sleeve_r"][0])
    t = np.array(
        [
            -s * center_x,
            ref_lm["arm_y"] - s * lm["sleeve_y"],
            ref_lm["torso_z_center"] - s * lm["center_z"],
        ]
    )
    return [(s * k, np.eye(3), t) for k in (1.0, 0.9, 1.1)]


def _init_bottoms(gv, ref_lm):
    lm = bottoms_landmarks(gv)
    # Vertical scale: crotch-to-hem should match crotch-to-ankle on the avatar.
    g_len = lm["crotch_y"] - lm["hem_y"]
    a_len = ref_lm["crotch_y"] - ref_lm["ankle_y"]
    inits = []
    for k in (1.0, 0.9, 1.15):
        s = a_len / g_len * k
        t = np.array(
            [
                -s * lm["center_xz"][0],
                ref_lm["crotch_y"] - s * lm["crotch_y"],
                ref_lm["torso_z_center"] - s * lm["center_xz"][1],
            ]
        )
        inits.append((s, np.eye(3), t))
    return inits


def _init_shoes(gv, ref_lm):
    lm = shoes_landmarks(gv)
    inits = []
    if "foot_l_center" not in ref_lm:
        return inits
    s0 = ref_lm["foot_z_extent"] / lm["z_extent"]
    ref_mid = 0.5 * (ref_lm["foot_l_center"] + ref_lm["foot_r_center"])
    g_mid = 0.5 * (lm["foot_l_center"] + lm["foot_r_center"])
    for yaw in (0.0, 180.0):
        R = yaw_matrix(yaw)
        for k in (1.0, 1.2, 0.85):
            s = s0 * k
            t = ref_mid - s * (R @ g_mid)
            # Sole on the floor.
            t[1] += ref_lm["foot_bottom_y"] - (s * lm["sole_y"] + t[1])
            inits.append((s, R, t))
    return inits


def _init_head(gv, ref_lm):
    center = gv.mean(axis=0)
    x_extent = gv[:, 0].max() - gv[:, 0].min()
    inits = []
    for yaw in (0.0, 180.0):
        R = yaw_matrix(yaw)
        for k in (1.0, 0.85, 1.2):
            s = ref_lm["head_x_extent"] / x_extent * k
            t = ref_lm["head_center"] - s * (R @ center)
            inits.append((s, R, t))
    return inits


INITIALIZERS = {
    "body": _init_body,
    "body_bald": _init_body,
    "tops": _init_tops,
    "bottoms": _init_bottoms,
    "shoes": _init_shoes,
    "head_accessories": _init_head,
}


def _refine(src, ref_pts, ref_tree, inits):
    best = None
    for s0, R0, t0 in inits:
        s, R, t, err = trimmed_icp(src, ref_tree, ref_pts, s0, R0, t0)
        if best is None or err < best[3]:
            best = (s, R, t, err)
    return best


def _side_submesh(mesh, vmask):
    fmask = vmask[mesh.faces].all(axis=1)
    return mesh.submesh([np.where(fmask)[0]], append=True)


def register_shoes(garment_mesh, ref_lm, ref_pts, ref_tree, n_src=8000):
    """Register each shoe independently to its own foot.

    A single similarity transform cannot match both the shoe size and the
    pair spacing (the generated pair is spaced differently than the avatar's
    stance), so the pair is split at the x gap between the two shoes and each
    side gets its own transform. Left/right foot assignment is chosen jointly
    by total ICP error, which also resolves any 180-degree yaw ambiguity.
    """
    gv = garment_mesh.vertices
    lm = shoes_landmarks(gv)
    x_split = 0.5 * (lm["foot_l_center"][0] + lm["foot_r_center"][0])
    sides = {"left": gv[:, 0] < x_split, "right": gv[:, 0] >= x_split}

    feet = {
        "left": ref_lm["foot_l_center"],
        "right": ref_lm["foot_r_center"],
    }
    s0 = ref_lm["foot_z_extent"] / lm["z_extent"]

    # results[garment_side][target_foot] = (s, R, t, err)
    results = {}
    for side, vmask in sides.items():
        sub = _side_submesh(garment_mesh, vmask)
        src = sample(sub, n_src)
        center = sub.vertices.mean(axis=0)
        sole_y = sub.vertices[:, 1].min()
        results[side] = {}
        for foot, foot_center in feet.items():
            inits = []
            for yaw in (0.0, 180.0):
                R = yaw_matrix(yaw)
                for k in (1.0, 1.2, 0.85):
                    s = s0 * k
                    t = foot_center - s * (R @ center)
                    t[1] += ref_lm["foot_bottom_y"] - (s * sole_y + t[1])
                    inits.append((s, R, t))
            results[side][foot] = _refine(src, ref_pts, ref_tree, inits)

    # Joint assignment: straight vs crossed, whichever fits better overall.
    straight = results["left"]["left"][3] + results["right"]["right"][3]
    crossed = results["left"]["right"][3] + results["right"]["left"][3]
    if straight <= crossed:
        pairs = [("left", "left"), ("right", "right")]
    else:
        pairs = [("left", "right"), ("right", "left")]

    parts = []
    for side, foot in pairs:
        s, R, t, err = results[side][foot]
        parts.append(
            {"part": f"{side}_shoe_to_{foot}_foot", "mask": sides[side],
             "s": s, "R": R, "t": t, "err": err}
        )
    return parts


def register_garment(name, garment_mesh, ref_lm, ref_pts, ref_tree, n_src=15000):
    """Return a list of parts, each with a vertex mask and its transform."""
    if name == "shoes":
        return register_shoes(garment_mesh, ref_lm, ref_pts, ref_tree)

    gv = garment_mesh.vertices
    src = sample(garment_mesh, n_src)
    inits = INITIALIZERS[name](gv, ref_lm)
    if not inits:
        inits = [(1.0, np.eye(3), np.zeros(3))]
    s, R, t, err = _refine(src, ref_pts, ref_tree, inits)
    return [
        {"part": name, "mask": np.ones(len(gv), dtype=bool),
         "s": s, "R": R, "t": t, "err": err}
    ]


def register_all(data_dir, out_dir, preview=True):
    os.makedirs(os.path.join(out_dir, "registered"), exist_ok=True)

    ref_scene, ref_mesh = load_mesh(os.path.join(data_dir, REFERENCE))
    ref_lm = avatar_landmarks(ref_mesh.vertices)
    ref_pts = sample(ref_mesh, 120000)
    ref_tree = cKDTree(ref_pts)

    results = {}
    combined = trimesh.Scene()
    for name, fname in GARMENTS.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[skip] {name}: {fname} not found")
            continue
        g_scene, g_mesh = load_mesh(path)
        parts = register_garment(name, g_mesh, ref_lm, ref_pts, ref_tree)

        # Bake the per-part transforms into the vertices (a garment may have
        # several parts with different transforms, e.g. left/right shoe).
        baked = g_mesh.vertices.copy()
        entry = {"file": fname, "parts": []}
        for p in parts:
            T = compose(p["s"], p["R"], p["t"])
            baked[p["mask"]] = (
                p["s"] * (g_mesh.vertices[p["mask"]] @ p["R"].T) + p["t"]
            )
            entry["parts"].append(
                {
                    "part": p["part"],
                    "scale": float(p["s"]),
                    "matrix": T.tolist(),
                    "mean_icp_error": float(p["err"]),
                }
            )
            print(f"[ok]   {name}/{p['part']}: scale={p['s']:.4f} icp_err={p['err']:.5f}")
        results[name] = entry

        for geom in g_scene.geometry.values():
            if len(geom.vertices) == len(baked):
                geom.vertices = baked
        g_scene.export(os.path.join(out_dir, "registered", f"{name}_registered.glb"))
        if name not in ("body", "body_bald"):
            for gname, geom in g_scene.geometry.items():
                combined.add_geometry(geom, node_name=f"{name}_{gname}")

    # Assembled look: bald body + all registered garments.
    body_path = os.path.join(out_dir, "registered", "body_bald_registered.glb")
    if os.path.exists(body_path):
        body_scene = trimesh.load(body_path, process=False)
        for gname, geom in body_scene.geometry.items():
            combined.add_geometry(geom, node_name=f"body_bald_{gname}")
    combined.export(os.path.join(out_dir, "assembled_avatar.glb"))

    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(results, f, indent=2)

    if preview:
        _render_previews(data_dir, out_dir, results, ref_mesh)
    return results


def _render_previews(data_dir, out_dir, results, ref_mesh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)
    rv = ref_mesh.vertices
    names = list(results)
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 9))
    for i, name in enumerate(names):
        reg = trimesh.load(
            os.path.join(out_dir, "registered", f"{name}_registered.glb"),
            process=False,
            force="mesh",
        )
        gv = reg.vertices
        err = max(p["mean_icp_error"] for p in results[name]["parts"])
        for row, (a, b, label) in enumerate([(0, 1, "front"), (2, 1, "side")]):
            ax = axes[row, i]
            ax.scatter(rv[:, a], rv[:, b], s=0.05, c="lightgray")
            ax.scatter(gv[:, a], gv[:, b], s=0.05, c="tab:red", alpha=0.5)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"{name}\nerr={err:.4f}")
    plt.tight_layout()
    out = os.path.join(out_dir, "previews", "registration_overlay.png")
    plt.savefig(out, dpi=80)
    plt.close(fig)
    print(f"[ok]   preview: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
    register_all(args.data, args.out, preview=not args.no_preview)


if __name__ == "__main__":
    main()
