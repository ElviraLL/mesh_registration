"""Type-agnostic global registration: FPFH + RANSAC + Umeyama, ICP-refined.

Replaces the per-garment-type landmark initializers: correspondences come
from FPFH feature matching (Open3D), RANSAC rejects bad matches and solves a
closed-form similarity (Umeyama with scaling), and trimmed scaled ICP
polishes the result against the reference surface.

FPFH is not scale invariant and the garment-to-avatar scale ratio is unknown
(every input is normalized to its own unit cube), so RANSAC runs over a
log-spaced grid of scale hypotheses: the source is pre-scaled, features are
recomputed, and the hypothesis whose refined fit has the lowest trimmed ICP
error wins. RANSAC's with-scaling estimation absorbs the gap between grid
points.
"""

import numpy as np
import open3d as o3d
import trimesh

# Reproducible CPU RANSAC (the torch path seeds its own generator).
if hasattr(o3d.utility, "random"):
    o3d.utility.random.seed(0)

from .icp import trimmed_icp, apply_srt, yaw_matrix
from .backend import NNIndex, use_torch, ransac_correspondences

SCALE_GRID = np.geomspace(0.08, 1.35, 10)

# Resolution pyramid (in reference units; the avatar is ~1.0 tall). A fixed
# voxel cannot serve both body-sized and shoe-sized garments: a small garment
# downsampled at the body's voxel keeps too few points and its FPFH radius
# swallows the whole object. Each scale hypothesis therefore picks the level
# matched to the scaled garment's size, and the reference is preprocessed
# once per level (cached).
VOXEL_LEVELS = (0.02, 0.013, 0.008, 0.005)


def _pick_voxel(extent):
    """Voxel level ~1/25 of the object's largest dimension."""
    v = extent / 25.0
    return min(VOXEL_LEVELS, key=lambda L: abs(np.log(L / v)))


def _preprocess(points, voxel):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
    )
    return pcd, fpfh


def _global_match(src_pcd, src_fpfh, dst_pcd, dst_fpfh, dist):
    """One global-registration attempt; returns a 4x4 similarity or None."""
    if use_torch():
        T, n_inliers = ransac_correspondences(
            np.asarray(src_pcd.points),
            np.asarray(src_fpfh.data).T,
            np.asarray(dst_pcd.points),
            np.asarray(dst_fpfh.data).T,
            dist,
        )
        return T if n_inliers >= 10 else None
    res = _ransac(src_pcd, src_fpfh, dst_pcd, dst_fpfh, dist)
    if len(res.correspondence_set) < 10:
        return None
    return np.asarray(res.transformation)


def _ransac(src_pcd, src_fpfh, dst_pcd, dst_fpfh, dist):
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd,
        dst_pcd,
        src_fpfh,
        dst_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(150000, 0.9999),
    )


def _decompose_similarity(T):
    """Split a 4x4 similarity matrix into (s, R, t)."""
    M = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(M)))
    R = M / s
    return s, R, T[:3, 3].copy()


def collect_candidates(
    src_pts, ref_cache, ref_pts, ref_tree, scales=SCALE_GRID, ransac_ref_pts=None
):
    """Collect registration candidates for one part across scale hypotheses.

    Returns a list of dicts with the refined transform, the trimmed ICP
    error, and the relative (untrimmed) fit error. No single-candidate
    verdict is made here: with unknown scale, per-part geometric scores are
    unreliable (absolute error favors shrunken fits hiding in surface folds,
    relative error favors inflated fits on large smooth regions), so the
    final choice is made jointly across all parts by `joint_select`.
    """
    rp = ref_pts if ransac_ref_pts is None else ransac_ref_pts
    extent = (src_pts.max(axis=0) - src_pts.min(axis=0)).max()
    cands = []
    for s0 in scales:
        voxel = _pick_voxel(extent * s0)
        if voxel not in ref_cache:
            ref_cache[voxel] = _preprocess(rp, voxel)
        dst_pcd, dst_fpfh = ref_cache[voxel]
        src_pcd, src_fpfh = _preprocess(src_pts * s0, voxel)
        # RANSAC is stochastic and a missing true candidate cannot be fixed
        # by any later selection logic, so take several attempts on the fine
        # resolution levels where it actually fails (small garments); the
        # coarse levels for large garments are reliable with one.
        attempts = 2 if voxel < 0.02 else 1
        if use_torch():
            attempts += 1  # attempts are cheap on the GPU
        for attempt in range(attempts):
            T = _global_match(src_pcd, src_fpfh, dst_pcd, dst_fpfh, 3 * voxel)
            if T is None:
                continue
            s1, R, t = _decompose_similarity(T)
            if not np.isfinite(s1) or not (0.4 <= s1 <= 2.5):
                # RANSAC collapsed or drifted far from this hypothesis; a
                # neighboring hypothesis covers that scale.
                continue
            s, R, t, err = trimmed_icp(src_pts, ref_tree, ref_pts, s0 * s1, R, t)
            cand = _finish(src_pts, s, R, t, err, ref_tree)
            if cand["rel"] > 0.03:
                # Sloppy convergence: a true fit stuck in a shallow local
                # minimum loses the joint selection to parasites, so try to
                # polish it from slightly perturbed poses.
                for ang in (-10.0, 10.0):
                    s2, R2, t2, err2 = trimmed_icp(
                        src_pts, ref_tree, ref_pts, s, yaw_matrix(ang) @ R, t
                    )
                    c2 = _finish(src_pts, s2, R2, t2, err2, ref_tree)
                    if c2["rel"] < cand["rel"]:
                        cand = c2
            cands.append(cand)
    cands = [c for c in cands if c["rel"] < 0.12]
    if not cands:  # RANSAC never found a fit; fall back to identity + ICP
        s, R, t, err = trimmed_icp(
            src_pts, ref_tree, ref_pts, 1.0, np.eye(3), np.zeros(3)
        )
        cands.append(_finish(src_pts, s, R, t, err, ref_tree))
    return cands


def _finish(src_pts, s, R, t, err, ref_tree, tau=0.012):
    """Package a refined candidate with its quality statistics.

    rel   : mean untrimmed distance to the reference, relative to scale.
    float : fraction of the garment's surface samples farther than tau from
            the reference surface, i.e. "floating in air". A garment truly
            baked into the reference cannot float (apart from hidden inner
            layers), so this is the scale-fair badness measure.
    """
    d, _ = ref_tree.query(apply_srt(src_pts, s, R, t))
    return {
        "s": s, "R": R, "t": t, "err": err,
        "rel": d.mean() / s, "float": float((d > tau).mean()),
    }


def _coverage_weights(src_pts, cand, ref_pts, tau=0.012):
    """Soft per-reference-point coverage in [0, 1].

    1 at distance 0, falling linearly to 0 at tau. Binary coverage lets an
    inflated fit "tarp over" a region it only crosses roughly; the soft
    weight scores that region low while a true, tight fit scores near 1.
    """
    cur = apply_srt(src_pts, cand["s"], cand["R"], cand["t"])
    d, _ = NNIndex(cur).query(ref_pts)
    d = np.minimum(d, tau)
    return (1.0 - d / tau).astype(np.float32)


def joint_select(parts, ref_pts, ref_area, background=None, beta=0.5,
                 rounds=4, tau=0.012):
    """Choose one candidate per part by maximizing net explained area.

    parts: dicts {"src": samples, "area": unit-frame surface area,
    "cands": [...]}. background: optional soft-coverage array claimed by an
    under-layer (the body), at reduced weight.

    Everything is measured in one currency, fractions of reference surface
    area:

    net = (marginal soft coverage vs. other parts and the background)
        - beta * float_fraction * (part_area * s^2 / ref_area)

    The reference is the union of the garments, so the correct joint
    solution tiles its surface. A shrunken fit explains almost no area; an
    inflated fit "tarps" a large region but pays for the surface it brings
    that lands nowhere (float), a cost that grows with s^2 and therefore
    cannot be gamed by scale in either direction. Hidden inner layers make
    real garments float a little, hence beta < 1.

    Coordinate descent; returns chosen candidate indices.
    """
    n_ref = len(ref_pts)
    if background is None:
        background = np.zeros(n_ref, dtype=np.float32)
    for part in parts:
        part["weights"] = [
            _coverage_weights(part["src"], c, ref_pts, tau) for c in part["cands"]
        ]

    def net(part, k, others):
        c = part["cands"][k]
        marginal = np.maximum(part["weights"][k] - others, 0.0).sum() / n_ref
        cost = beta * c["float"] * (part["area"] * c["s"] ** 2 / ref_area)
        return marginal - cost

    chosen = [
        int(np.argmax([net(p, k, background) for k in range(len(p["cands"]))]))
        for p in parts
    ]
    for _ in range(rounds):
        changed = False
        for i, part in enumerate(parts):
            others = background.copy()
            for j, other in enumerate(parts):
                if j != i:
                    others = np.maximum(others, other["weights"][chosen[j]])
            best = int(np.argmax(
                [net(part, k, others) for k in range(len(part["cands"]))]
            ))
            if best != chosen[i]:
                chosen[i] = best
                changed = True
        if not changed:
            break
    return chosen


def split_parts(mesh, gap=0.05):
    """Split a mesh into spatially separate parts, type-agnostically.

    Connected components are merged whenever their axis-aligned bounding
    boxes (expanded by gap/2) intersect, so a pair of shoes becomes two
    parts while a single garment with floating bits stays one part.
    Returns a list of vertex masks.
    """
    comps = trimesh.graph.connected_components(
        mesh.edges, min_len=1, nodes=np.arange(len(mesh.vertices))
    )
    boxes = []
    for c in comps:
        v = mesh.vertices[c]
        boxes.append((v.min(axis=0) - gap / 2, v.max(axis=0) + gap / 2))

    # Union-find over overlapping boxes.
    parent = list(range(len(comps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            (lo_i, hi_i), (lo_j, hi_j) = boxes[i], boxes[j]
            if np.all(hi_i >= lo_j) and np.all(hi_j >= lo_i):
                parent[find(i)] = find(j)

    groups = {}
    for i, c in enumerate(comps):
        groups.setdefault(find(i), []).append(c)

    masks = []
    for members in groups.values():
        mask = np.zeros(len(mesh.vertices), dtype=bool)
        mask[np.concatenate(members)] = True
        masks.append(mask)
    masks.sort(key=lambda m: -m.sum())
    return masks
