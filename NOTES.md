# Notes — rigged skeletons + garment registration

## What this branch adds

Two new inputs arrived as separate data sets, each with a rigged
"auto-setup" avatar (an R15-style skeleton + skinned body-part meshes):

- `data/set1/` — the original meshes (Link character).
- `data/set2/` — the new meshes (swordsman: body, tops, bottoms, shoes,
  and a held sword under the name "gears").
- `data/set1/rig/`, `data/set2/rig/` — the rigged one-piece GLBs.
- `data/pending_rigs/link_one_piece_autosetup.glb` — Link's rig, parked
  until Link's own garment set is provided.

### 1. Rigged skeletons drive the body skeleton (`registration/rig.py`)

`skeleton.py` used geometric proportion heuristics (`hip = crotch + 0.04H`
etc.) to guess joints. Those break on stylized proportions — the set2
chibi's neck landed at y≈0.38 (top of the big head) instead of ~0.2.

Now, when a rig is present under `<data>/rig/`, we parse its rest-pose
joints, similarity-align the rig frame onto the processed one-piece
reference (bounding box + trimmed ICP), and map the R15 joints to the
pipeline's 19-joint naming. Alignment errors: set1 0.008, set2 0.002 —
the rigs are the same characters as the references.

### 2. Skeleton-guided garment registration

The FPFH global registration is type-agnostic but, on stylized data, its
coverage-based selection could not place garments that are mostly occluded
in the baked reference (pants under a coat) or that need a non-canonical
pose (the sword is horizontal in the hand). Additions:

- **Region masks** (`rig.region_masks`): partition the reference surface
  by nearest skeleton bone into body regions (torso, legs, shins, feet,
  arms, head, held-prop regions past each wrist).
- **Anchor candidates** (`rig.anchor_candidates`): per-region
  initializations (scale from region extent, translation from centroid,
  rotation from the canonical prior + principal-axis alignment for held
  props), refined with rotation-locked ICP outside prop regions.
- **Type→region constraints** (`register.TYPE_REGIONS`): a filename-typed
  garment is restricted to its plausible regions.
- **Region-fit override** (`register.register_all_fpfh`): for type-known
  garments the final pick is by how well the fit matches its region's
  extent and center (plus an upright/tilt penalty and left/right-side
  exclusion so a pair can't both land on one foot), because coverage
  scoring is unreliable for occluded garments.
- **Scoring** changed from a subtractive area cost to a multiplicative
  float discount `marginal·(1-float)²`, which does not push a correct
  high-coverage fit below zero on stylized data.
- Two ICP bugs fixed: the scale-clamp branch ignored `with_rot`; the
  RANSAC-failure fallback did not lock orientation.

## Result (set2, swordsman)

All six parts land in their correct regions (was 3/6 before): body, tops
on torso, bottoms across both legs, one boot per foot, sword horizontal in
the right hand bound to `wrist_r`. set1 is unchanged (no rig path when no
rig is present, and its results match the prior baseline).

## Known limitation — this is the real open item

Registration still does **surface fitting** of an independently generated
garment onto the *baked* one-piece surface. That is ill-posed here:

- garments are heavily occluded in the reference (pants under coat+boots),
  so there is little correct surface to fit;
- the baked garment and the standalone one differ in shape/thickness;
- Open3D RANSAC is non-deterministic, so the good candidate is not always
  found (tops sometimes converges to scale ~0.4, sometimes ~0.25).

Consequence: **tops, shoes, and bottoms are placed in the right region but
their scale/tilt is not yet reliable.** Chasing this with more scoring
tweaks is whack-a-mole.

### The fix: per-garment segmentation labels

The "nearest-bone region" masks are a geometric *approximation* of each
garment's target surface. A real segmentation of the one-piece reference
makes the problem well-posed: each garment fits only its own labelled
surface — no joint selection, no type heuristics, no anchor whack-a-mole,
and occlusion stops mattering because the visible sliver is exclusively
that garment's.

**Preferred source: per-face labels exported from the generation side**
(material groups / part IDs), which are ground truth and zero-error. A
segmentation model is only the fallback when labels aren't available.

Interface needed (any of these):

```
one_piece reference: one label per vertex OR per face
label set: body, tops, bottoms, shoes, gears (or the project's categories)
format: .npy / .json / vertex-coloured GLB, in the reference mesh order
```

Drop-in: `region_masks` reads the labels instead of nearest-bone
assignment (bone regions stay as the no-label fallback). Then re-run set2
and the garment scale/tilt should improve directly.

## Running

```bash
pip install -r requirements.txt          # + `pip install torch` for GPU
python -m registration.register --data data/set2 --out output/set2
python -m registration.skeleton --data data/set2 --out output/set2
```

CPU only, ~7–10 min per set (FPFH+RANSAC dominates). With a CUDA GPU,
`REG_DEVICE=auto` (default) uses it and RANSAC does extra cheap attempts,
which also steadies the tops non-determinism.
