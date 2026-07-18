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
