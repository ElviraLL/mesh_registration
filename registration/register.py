"""Register separately generated garment meshes onto the reference avatar.

Usage:
    python -m registration.register --data data --out output

Every garment ends up with per-part similarity transforms refined by trimmed
scaled ICP against the reference "one piece" avatar (which has the same
garments baked in). Initialization comes from either type-agnostic global
registration with joint coverage selection (--method fpfh, default; see
global_reg.py) or hand-crafted T-pose landmarks (--method landmarks).
"""

import argparse
import json
import os

import numpy as np
import trimesh
from .backend import NNIndex

from .landmarks import (
    avatar_landmarks,
    bottoms_landmarks,
    shoes_landmarks,
    tops_landmarks,
)
from .icp import trimmed_icp, compose, yaw_matrix, fit_score, oriented_score, rot180

# Garment-type inference from filenames, needed only by --method landmarks
# (the fpfh method is type-agnostic and registers any mesh). Checked in
# order; first hit wins.
TYPE_KEYWORDS = [
    ("body", "body"),
    ("tops", "top"), ("tops", "shirt"), ("tops", "jacket"),
    ("bottoms", "bottom"), ("bottoms", "pant"), ("bottoms", "trouser"),
    ("shoes", "shoe"), ("shoes", "boot"), ("shoes", "feet"),
    ("head_accessories", "head"), ("head_accessories", "helmet"),
    ("head_accessories", "hat"), ("head_accessories", "hair"),
]


def discover(data_dir):
    """Find the reference and the garment meshes in a data directory.

    The reference is the GLB whose filename contains "one_piece" (the
    avatar with all garments baked in); every other .glb is a garment.
    Returns (reference_filename, {name: filename}).
    """
    import glob as _glob

    files = sorted(
        os.path.basename(f)
        for f in _glob.glob(os.path.join(data_dir, "*.glb"))
        # rigged auto-setup avatars are skeleton sources, not garments
        if "autosetup" not in os.path.basename(f).lower()
    )
    refs = [f for f in files if "one_piece" in f.lower()]
    if len(refs) != 1:
        raise SystemExit(
            f"expected exactly one reference GLB with 'one_piece' in its "
            f"name in {data_dir}, found {refs or 'none'}"
        )
    garments = {}
    for f in files:
        if f == refs[0]:
            continue
        name = os.path.splitext(f)[0]
        for prefix in ("3d_generated_mesh_",):
            if name.startswith(prefix):
                name = name[len(prefix):]
        for suffix in ("_processed",):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        garments[name] = f
    return refs[0], garments


def garment_type(name):
    """Infer the landmark-method garment type from a mesh name, or None."""
    low = name.lower()
    for gtype, kw in TYPE_KEYWORDS:
        if kw in low:
            return gtype
    return None


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


def sample_with_normals(mesh, n):
    pts, fidx = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts), mesh.face_normals[fidx]


def orient_and_polish(mesh, part, ref_tree, ref_pts, ref_nrm, n_src=15000):
    """Fix flipped/tilted orientations of a chosen part, then final-polish.

    Pure point-distance ICP cannot distinguish a near-symmetric part (a
    helmet dome, a boot) from its 180-degree flip, and shape differences
    between the standalone garment and its baked counterpart can tilt the
    converged pose. This stage re-polishes the chosen transform with
    normal-consistent ICP (opposing-normal correspondences rejected) and
    compares it against its three 180-degree flips about the part's own
    axes, scored by normal-weighted error.
    """
    sub = _mask_submesh(mesh, part["mask"])
    src, src_nrm = sample_with_normals(sub, n_src)

    # The scale was already established (and cross-checked by the joint
    # coverage selection); orientation refinement must not touch it, or the
    # absolute-error comparison reopens the shrink exploit. Pre-scale the
    # source and run rotation+translation-only ICP: for x -> s*R*x + t, the
    # (R, t) of the scaled source are exactly the part's own (R, t).
    S = part["s"]
    srcS = S * src
    center = srcS.mean(axis=0)

    _, R, t, err = trimmed_icp(
        srcS, ref_tree, ref_pts, 1.0, part["R"], part["t"],
        src_nrm=src_nrm, tgt_nrm=ref_nrm, with_scale=False,
    )
    score = oriented_score(src, src_nrm, S, R, t, ref_tree, ref_nrm)

    # Yaw-flip test with an identity-preferring tie-break: generated assets
    # face forward, and for a near-symmetric part (helmet dome) the
    # geometric score difference between forward and backward sits inside
    # sampling noise. Adopt whichever variant is closer to the identity
    # unless the other is decisively (>10%) better.
    Q = rot180(1)
    R2 = R @ Q
    t2 = t + R @ (center - Q @ center)
    _, R3, t3, err3 = trimmed_icp(
        srcS, ref_tree, ref_pts, 1.0, R2, t2,
        src_nrm=src_nrm, tgt_nrm=ref_nrm, with_scale=False,
    )
    sc3 = oriented_score(src, src_nrm, S, R3, t3, ref_tree, ref_nrm)

    from .global_reg import _near_canonical, _rotation_angle_deg

    cur_first = _rotation_angle_deg(R) <= _rotation_angle_deg(R3)
    if cur_first:
        take_flip = sc3 < 0.9 * score
    else:
        take_flip = not (score < 0.9 * sc3)
    if take_flip:
        R, t, err, score = R3, t3, err3, sc3

    if _near_canonical(R):
        part["s"], part["R"], part["t"], part["err"] = S, R, t, err
    # else: keep the pre-orientation transform; the polish drifted off the
    # canonical orientation prior and cannot be trusted.
    return part


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
    "tops": _init_tops,
    "bottoms": _init_bottoms,
    "shoes": _init_shoes,
    "head_accessories": _init_head,
}


def _refine(src, ref_pts, ref_tree, inits):
    best, best_score = None, np.inf
    for s0, R0, t0 in inits:
        s, R, t, err = trimmed_icp(src, ref_tree, ref_pts, s0, R0, t0)
        score = fit_score(src, s, R, t, ref_tree)
        if score < best_score:
            best, best_score = (s, R, t, err), score
    return best


def _mask_submesh(mesh, vmask):
    fmask = vmask[mesh.faces].all(axis=1)
    return mesh.submesh([np.where(fmask)[0]], append=True)



def register_all_fpfh(loaded, ref_cache, ref_pts, ref_tree, ref_nrm, ref_area,
                      n_src=30000, rig_masks=None):
    """Type-agnostic registration of every garment: candidates + joint pick.

    Phase A collects FPFH+RANSAC candidates per spatially-separate part.
    Phase B picks each body candidate standalone (bodies are unambiguous:
    they explain by far the most area), then selects the garments jointly
    against the body as a down-weighted background layer - the body sits
    UNDER the clothes, so garments override it, but its claim on the skin
    (face, hands) stops an inflated garment from freeloading there.
    Parts that end up explaining almost nothing get a coverage-guided
    RANSAC retry against only the still-unclaimed surface.
    """
    from .global_reg import (
        collect_candidates,
        joint_select,
        split_parts,
        _coverage_weights,
        RETRY_SCALE_GRID,
    )

    bodies = ("body", "body_bald")
    entries = []
    for name, (scene, mesh) in loaded.items():
        masks = split_parts(mesh)
        for i, vmask in enumerate(masks):
            sub = _mask_submesh(mesh, vmask)
            src, src_nrm = sample_with_normals(sub, n_src)
            # ICP refinement of the many candidates runs on a subsample;
            # coverage weights and the final polish use the full set.
            cands = collect_candidates(
                src[::3], ref_cache, ref_pts, ref_tree, ref_nrm, src_nrm[::3]
            )
            label = name if len(masks) == 1 else f"{name}_part{i}"
            entries.append(
                {"name": name, "part": label, "mask": vmask, "src": src,
                 "src_nrm": src_nrm, "area": sub.area, "cands": cands}
            )
            # Skeleton-guided anchors complement FPFH: they cover poses the
            # feature matching misses (tiny targets, principal-axis flips).
            if rig_masks is not None and name not in bodies:
                from .rig import anchor_candidates

                extra = anchor_candidates(
                    src, src_nrm, rig_masks, ref_pts, ref_tree, ref_nrm
                )
                entries[-1]["cands"] = cands + extra
                print(f"[..]   {label}: {len(cands)} candidates "
                      f"+ {len(extra)} anchors")
            else:
                print(f"[..]   {label}: {len(cands)} candidates")

    n_ref = len(ref_pts)

    def solo_net(e, c, w):
        # Same discounted-coverage currency as joint_select.
        return w.sum() / n_ref * (1.0 - c["float"]) ** 2

    background = np.zeros(n_ref, dtype=np.float32)
    for e in entries:
        if e["name"] in bodies:
            e["weights"] = [
                _coverage_weights(e["src"], c, ref_pts) for c in e["cands"]
            ]
            k = int(np.argmax([
                solo_net(e, c, w) for c, w in zip(e["cands"], e["weights"])
            ]))
            e["chosen"] = e["cands"][k]
            if e["name"] == "body_bald" or not background.any():
                background = 0.5 * e["weights"][k]

    garment_entries = [e for e in entries if e["name"] not in bodies]
    chosen = joint_select(garment_entries, ref_pts, ref_area, background)

    # Coverage-guided retry: a part whose chosen fit explains almost no
    # exclusive area is mis-registered (e.g. both shoes landed on the same
    # foot). Re-run RANSAC for it against only the unclaimed surface, where
    # its true region no longer competes with the whole avatar.
    weights = [e["weights"][ci] for e, ci in zip(garment_entries, chosen)]
    retried = []
    for i, e in enumerate(garment_entries):
        others = background.copy()
        for j, w in enumerate(weights):
            if j != i:
                others = np.maximum(others, w)
        marginal = np.maximum(weights[i] - others, 0.0).sum() / n_ref
        if marginal < 0.02:
            e["cands"] = e["cands"] + collect_candidates(
                e["src"][::3], {}, ref_pts, ref_tree, ref_nrm,
                e.get("src_nrm")[::3] if e.get("src_nrm") is not None else None,
                scales=RETRY_SCALE_GRID,
                ransac_ref_pts=ref_pts[others < 0.5],
            )
            retried.append(e["part"])
    if retried:
        print(f"[..]   coverage retry: {', '.join(retried)}")
        chosen = joint_select(garment_entries, ref_pts, ref_area, background)

    for e, ci in zip(garment_entries, chosen):
        e["chosen"] = e["cands"][ci]

    # Selection diagnostics: every candidate of every part with the terms
    # of the joint objective, so a wrong pick can be diagnosed offline.
    report = []
    for e in garment_entries + [x for x in entries if x["name"] in bodies]:
        others = background.copy()
        for o in garment_entries:
            if o is not e and "weights" in o and "chosen" in o:
                others = np.maximum(
                    # index by identity: dict == on candidates compares
                    # numpy arrays and raises when scales tie exactly
                    others, o["weights"][next(
                        i for i, c in enumerate(o["cands"]) if c is o["chosen"]
                    )]
                )
        rows = []
        ws = e.get("weights")
        for k, c in enumerate(e["cands"]):
            row = {"s": float(c["s"]), "rel": float(c["rel"]),
                   "float": float(c["float"]), "chosen": c is e["chosen"]}
            if ws is not None and k < len(ws):
                row["total_cov"] = float(ws[k].sum() / n_ref)
                row["marginal"] = float(
                    np.maximum(ws[k] - others, 0.0).sum() / n_ref
                )
                row["net"] = row["marginal"] * (1.0 - c["float"]) ** 2
            rows.append(row)
        report.append({"part": e["part"], "cands": rows})
    register_all_fpfh.last_report = report

    parts_by_name = {}
    for e in entries:
        c = e["chosen"]
        parts_by_name.setdefault(e["name"], []).append(
            {"part": e["part"], "mask": e["mask"],
             "s": c["s"], "R": c["R"], "t": c["t"], "err": c["err"]}
        )
    return parts_by_name


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
        sub = _mask_submesh(garment_mesh, vmask)
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


def register_garment(gtype, name, garment_mesh, ref_lm, ref_pts, ref_tree,
                     n_src=15000):
    """Return a list of parts, each with a vertex mask and its transform."""
    if gtype == "shoes":
        return register_shoes(garment_mesh, ref_lm, ref_pts, ref_tree)

    gv = garment_mesh.vertices
    src = sample(garment_mesh, n_src)
    inits = INITIALIZERS[gtype](gv, ref_lm)
    if not inits:
        inits = [(1.0, np.eye(3), np.zeros(3))]
    s, R, t, err = _refine(src, ref_pts, ref_tree, inits)
    return [
        {"part": name, "mask": np.ones(len(gv), dtype=bool),
         "s": s, "R": R, "t": t, "err": err}
    ]


def register_all(data_dir, out_dir, preview=True, method="fpfh", seed=0):
    np.random.seed(seed)
    os.makedirs(os.path.join(out_dir, "registered"), exist_ok=True)

    reference, garments = discover(data_dir)
    print(f"[..]   reference: {reference}")
    ref_scene, ref_mesh = load_mesh(os.path.join(data_dir, reference))
    ref_lm = avatar_landmarks(ref_mesh.vertices)
    ref_pts, ref_nrm = sample_with_normals(ref_mesh, 120000)
    ref_tree = NNIndex(ref_pts)
    ref_feat = {} if method == "fpfh" else None

    loaded = {}
    for name, fname in garments.items():
        loaded[name] = load_mesh(os.path.join(data_dir, fname))

    # With a rigged avatar available, its joints partition the reference
    # surface into body regions that guide extra registration candidates.
    rig_masks = None
    if method == "fpfh":
        from .rig import find_rig, region_masks, rig_body_skeleton

        rig_path = find_rig(data_dir)
        if rig_path is not None:
            joints, rig_err = rig_body_skeleton(
                rig_path, ref_mesh.vertices, ref_pts, ref_tree
            )
            rig_masks = region_masks(joints, ref_pts)
            print(f"[..]   rig: {os.path.basename(rig_path)} "
                  f"(align err {rig_err:.5f})")

    if method == "fpfh":
        parts_by_name = register_all_fpfh(loaded, ref_feat, ref_pts, ref_tree, ref_nrm, ref_mesh.area, rig_masks=rig_masks)
        with open(os.path.join(out_dir, "selection_report.json"), "w") as f:
            json.dump(getattr(register_all_fpfh, "last_report", []), f, indent=1)
    else:
        parts_by_name = {}
        for name, (g_scene, g_mesh) in loaded.items():
            gtype = garment_type(name)
            if gtype is None:
                print(f"[skip] {name}: no landmark rules for this type; "
                      f"use --method fpfh for arbitrary garments")
                continue
            parts_by_name[name] = register_garment(
                gtype, name, g_mesh, ref_lm, ref_pts, ref_tree
            )
        loaded = {n: m for n, m in loaded.items() if n in parts_by_name}

    results = {}
    combined = trimesh.Scene()
    for name, (g_scene, g_mesh) in loaded.items():
        fname = garments[name]
        parts = [
            orient_and_polish(g_mesh, p, ref_tree, ref_pts, ref_nrm)
            for p in parts_by_name[name]
        ]

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
    ap.add_argument(
        "--method",
        choices=["fpfh", "landmarks"],
        default="fpfh",
        help="fpfh: type-agnostic FPFH+RANSAC global registration with "
        "joint coverage selection (generalizes to new avatars); "
        "landmarks: fast hand-crafted T-pose landmark initialization",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    register_all(
        args.data, args.out,
        preview=not args.no_preview, method=args.method, seed=args.seed,
    )


if __name__ == "__main__":
    main()
