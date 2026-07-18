"""Load a rigged (auto-setup) avatar GLB and express its skeleton in the
reference frame of the processed one-piece mesh.

The rigged file is the same character as the one-piece reference but in its
own (un-normalized) coordinate frame, with an R15-style joint hierarchy
(HumanoidRootNode / LowerTorso / UpperTorso / Head / *UpperArm / *LowerArm /
*Hand / *UpperLeg / *LowerLeg / *Foot) and skinned body-part meshes named
"*_Geo". Alignment to the processed reference is a similarity transform:
initialized from the bounding boxes, refined with trimmed ICP.

Joint mapping to the pipeline's names (see skeleton.BONES): R15 has no
explicit neck/spine/foot-tip joints, so `neck` is the R15 Head pivot,
`spine` the LowerTorso-UpperTorso midpoint, `head` the centroid of the mesh
above the neck, and foot tips come from the mesh like the heuristic
estimator. Left/right (_l/_r) follow the pipeline's x-sign convention, not
the rig's Left*/Right* names.
"""

import glob
import os

import numpy as np

R15_JOINTS = [
    "HumanoidRootNode", "LowerTorso", "UpperTorso", "Head",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
]


def _trs_matrix(node):
    if node.matrix:
        return np.array(node.matrix, dtype=float).reshape(4, 4).T
    M = np.eye(4)
    R = np.eye(3)
    if node.rotation:
        x, y, z, w = node.rotation
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    S = np.diag(node.scale) if node.scale else np.eye(3)
    M[:3, :3] = R @ S
    if node.translation:
        M[:3, 3] = node.translation
    return M


def _read_accessor(g, idx, blob):
    acc = g.accessors[idx]
    bv = g.bufferViews[acc.bufferView]
    off = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc.type]
    dt = {5126: "f4", 5123: "u2", 5125: "u4", 5121: "u1"}[acc.componentType]
    a = np.frombuffer(blob, dtype=dt, count=acc.count * ncomp, offset=off)
    return a.reshape(acc.count, ncomp)


def load_rig(path):
    """Parse a rigged GLB; returns (joints {r15_name: xyz}, geo_vertices).

    Positions are rest-pose world space. geo_vertices stacks every "*_Geo"
    body-part mesh, for aligning the rig frame to the reference frame.
    """
    from pygltflib import GLTF2

    g = GLTF2().load(path)
    blob = g.binary_blob()

    parent = {}
    for i, n in enumerate(g.nodes):
        for c in n.children or []:
            parent[c] = i
    world = {}

    def get_world(i):
        if i not in world:
            M = _trs_matrix(g.nodes[i])
            world[i] = get_world(parent[i]) @ M if i in parent else M
        return world[i]

    name2idx = {n.name: i for i, n in enumerate(g.nodes)}
    missing = [nm for nm in R15_JOINTS if nm not in name2idx]
    if missing:
        raise ValueError(f"{path}: rig is missing joints {missing}")
    joints = {nm: get_world(name2idx[nm])[:3, 3].copy() for nm in R15_JOINTS}

    geo = []
    for i, n in enumerate(g.nodes):
        if n.mesh is None:
            continue
        mesh_name = g.meshes[n.mesh].name or ""
        if not mesh_name.endswith("_Geo"):
            continue
        W = get_world(i)
        for prim in g.meshes[n.mesh].primitives:
            v = _read_accessor(g, prim.attributes.POSITION, blob)
            geo.append(v.astype(float) @ W[:3, :3].T + W[:3, 3])
    if not geo:
        raise ValueError(f"{path}: no '*_Geo' meshes found for alignment")
    return joints, np.vstack(geo)


def align_to_reference(joints, geo, ref_pts, ref_tree):
    """Similarity-align the rig frame onto the reference; returns
    (aligned joints, mean ICP error)."""
    from .icp import trimmed_icp

    s0 = np.median(
        (ref_pts.max(axis=0) - ref_pts.min(axis=0))
        / (geo.max(axis=0) - geo.min(axis=0))
    )
    t0 = (
        0.5 * (ref_pts.max(axis=0) + ref_pts.min(axis=0))
        - s0 * 0.5 * (geo.max(axis=0) + geo.min(axis=0))
    )
    src = geo[:: max(1, len(geo) // 20000)]
    s, R, t, err = trimmed_icp(src, ref_tree, ref_pts, s0, np.eye(3), t0)
    aligned = {k: s * (R @ p) + t for k, p in joints.items()}
    return aligned, err


def _pick_side(aligned, base, tag):
    """R15 joint name for pipeline side `tag`, by x sign (pipeline _l = -x)."""
    left, right = aligned[f"Left{base}"], aligned[f"Right{base}"]
    if tag == "l":
        return left if left[0] <= right[0] else right
    return right if left[0] <= right[0] else left


def rig_body_skeleton(rig_path, ref_vertices, ref_pts, ref_tree):
    """Pipeline-named body joints from a rigged GLB, in the reference frame.

    Returns (joints {name: xyz}, alignment ICP error). Mesh-derived
    fallbacks (head center, foot tips) use ref_vertices.
    """
    r15, geo = load_rig(rig_path)
    aligned, err = align_to_reference(r15, geo, ref_pts, ref_tree)

    joints = {}
    joints["pelvis"] = aligned["LowerTorso"]
    joints["chest"] = aligned["UpperTorso"]
    joints["spine"] = 0.5 * (aligned["LowerTorso"] + aligned["UpperTorso"])
    joints["neck"] = aligned["Head"]
    for tag in ("l", "r"):
        joints[f"shoulder_{tag}"] = _pick_side(aligned, "UpperArm", tag)
        joints[f"elbow_{tag}"] = _pick_side(aligned, "LowerArm", tag)
        joints[f"wrist_{tag}"] = _pick_side(aligned, "Hand", tag)
        joints[f"hip_{tag}"] = _pick_side(aligned, "UpperLeg", tag)
        joints[f"knee_{tag}"] = _pick_side(aligned, "LowerLeg", tag)
        joints[f"ankle_{tag}"] = _pick_side(aligned, "Foot", tag)

    # Head center: centroid of the mesh above the neck pivot.
    head_region = ref_vertices[ref_vertices[:, 1] > joints["neck"][1]]
    joints["head"] = (
        head_region.mean(axis=0)
        if len(head_region)
        else joints["neck"] + np.array([0.0, 0.05, 0.0])
    )

    # Foot tips: front-most mesh point below each ankle, split by x sign.
    for tag in ("l", "r"):
        ankle = joints[f"ankle_{tag}"]
        sgn = -1.0 if tag == "l" else 1.0
        feet = ref_vertices[
            (ref_vertices[:, 1] < ankle[1] + 0.02)
            & (np.sign(ref_vertices[:, 0] - joints["pelvis"][0]) == sgn)
        ]
        joints[f"foot_{tag}"] = (
            feet[feet[:, 2].argmax()] if len(feet) else ankle
        )

    return {k: np.asarray(p, dtype=float) for k, p in joints.items()}, err


def find_rig(data_dir):
    """Rigged GLB for a data set: data_dir/rig/*.glb or *autosetup*.glb."""
    cands = sorted(glob.glob(os.path.join(data_dir, "rig", "*.glb")))
    cands += sorted(glob.glob(os.path.join(data_dir, "*autosetup*.glb")))
    return cands[0] if cands else None


# Body regions as groups of skeleton bones. Composites ('legs', 'upper')
# cover garments that span several limbs (pants, tops).
REGION_BONES = {
    "torso": [("pelvis", "spine"), ("spine", "chest")],
    "head": [("chest", "neck"), ("neck", "head")],
    "arm_l": [("chest", "shoulder_l"), ("shoulder_l", "elbow_l"),
              ("elbow_l", "wrist_l")],
    "arm_r": [("chest", "shoulder_r"), ("shoulder_r", "elbow_r"),
              ("elbow_r", "wrist_r")],
    "leg_l": [("pelvis", "hip_l"), ("hip_l", "knee_l"), ("knee_l", "ankle_l")],
    "leg_r": [("pelvis", "hip_r"), ("hip_r", "knee_r"), ("knee_r", "ankle_r")],
    "foot_l": [("ankle_l", "foot_l")],
    "foot_r": [("ankle_r", "foot_r")],
}
REGION_COMPOSITES = {
    "legs": ["leg_l", "leg_r"],
    "feet": ["foot_l", "foot_r"],
    "upper": ["torso", "arm_l", "arm_r"],
}


def region_masks(joints, ref_pts):
    """Partition the reference samples by nearest skeleton bone.

    Returns {region: bool mask over ref_pts}. Every point belongs to
    exactly one base region (nearest-bone assignment tiles the surface, so
    a garment baked into the reference is fully claimed by the regions its
    bones span - including props like a held sword, which joins the arm
    region of the hand that holds it).
    """
    from .skeleton import BONES, _point_segment_dist

    dists = np.stack([
        _point_segment_dist(ref_pts, joints[a], joints[b])[0] for a, b in BONES
    ])
    nearest = np.argmin(dists, axis=0)

    bone_region = {}
    for region, bones in REGION_BONES.items():
        for ab in bones:
            bone_region[BONES.index(ab)] = region
    masks = {
        region: np.isin(nearest, [i for i, r in bone_region.items() if r == region])
        for region in REGION_BONES
    }
    for name, members in REGION_COMPOSITES.items():
        masks[name] = np.any([masks[m] for m in members], axis=0)
    return masks


def _principal_axis(pts):
    """(unit principal axis, sorted stddevs desc) of a point cloud."""
    w, V = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
    return V[:, -1], np.sqrt(np.maximum(w[::-1], 0.0))


def _axis_align_rot(v, u):
    """Minimal rotation taking unit vector v onto unit vector u."""
    c = float(np.clip(v @ u, -1.0, 1.0))
    axis = np.cross(v, u)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees about any axis perpendicular to v.
        perp = np.cross(v, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(v, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    axis /= s
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def anchor_candidates(src, src_nrm, masks, ref_pts, ref_tree, ref_nrm):
    """Skeleton-guided registration candidates for one garment part.

    For each body region: scale from bounding-box extents, translation from
    centroids, rotations from the canonical prior (identity / yaw 180) plus
    principal-axis alignment for elongated parts (a held prop like a sword
    is vertical in its own frame but horizontal in the avatar's hand - a 90
    degree pose FPFH's canonical-orientation gate can never produce). Each
    init is refined with normal-consistent trimmed ICP; the joint coverage
    selection judges them against the FPFH candidates on equal footing.
    """
    from .global_reg import _finish
    from .icp import trimmed_icp, yaw_matrix

    sub = src[:: max(1, len(src) // 6000)]
    sub_nrm = src_nrm[:: max(1, len(src) // 6000)] if src_nrm is not None else None
    p_axis, p_std = _principal_axis(src)
    p_ext = (src.max(axis=0) - src.min(axis=0)).max()
    p_center = src.mean(axis=0)
    elongated = p_std[0] > 2.5 * max(p_std[1], 1e-9)

    cands = []
    for region, mask in masks.items():
        pts = ref_pts[mask]
        if len(pts) < 300:
            continue
        r_ext = (pts.max(axis=0) - pts.min(axis=0)).max()
        s0 = r_ext / max(p_ext, 1e-9)
        if not (0.05 <= s0 <= 2.0):
            continue
        rots = [np.eye(3), yaw_matrix(180.0)]
        if elongated:
            r_axis, _ = _principal_axis(pts)
            rots += [_axis_align_rot(p_axis, r_axis),
                     _axis_align_rot(p_axis, -r_axis)]
        center = pts.mean(axis=0)
        for R0 in rots:
            t0 = center - s0 * (R0 @ p_center)
            s, R, t, err = trimmed_icp(
                sub, ref_tree, ref_pts, s0, R0, t0,
                src_nrm=sub_nrm, tgt_nrm=ref_nrm,
            )
            cand = _finish(src, s, R, t, err, ref_tree, ref_pts, ref_nrm)
            if cand["rel"] < 0.12:
                cand["anchor"] = region
                cands.append(cand)
    return cands
