# mesh_registration

Landmark-based registration of separately generated garment meshes onto a
reference avatar.

## Problem

Each mesh in `data/` is generated independently and normalized to its own
unit cube, so the body, tops, bottoms, shoes and head accessories do not
share a scale or position. The "one piece" avatar
(`3d_generated_mesh_one_piece_avatar_body_processed.glb`) is the reference:
it has all garments baked in as a single mesh, so every separate garment has
a nearly coincident counterpart region on its surface.

## Approach

Two interchangeable methods produce per-part similarity transforms
(uniform scale + rotation + translation), refined by trimmed scaled ICP
against surface samples of the reference (the reference contains the same
garments baked in, so each garment snaps onto its counterpart region).

### `--method fpfh` (default; type-agnostic, generalizes to new avatars)

No landmarks and no knowledge of garment types:

1. **Parts** — each mesh is split into spatially separate parts
   (connected components merged by bounding-box overlap), so a shoe pair
   becomes two independently-registered parts.
2. **Candidates** — FPFH features + RANSAC with a closed-form similarity
   solve (Umeyama with scaling) propose registrations. FPFH is not scale
   invariant and the garment-to-avatar scale is unknown, so RANSAC runs
   over a log-spaced grid of scale hypotheses, with the voxel/feature
   resolution adapted to the scaled garment size and the reference
   preprocessed once per resolution level. Sloppily converged candidates
   are re-polished from perturbed poses.
3. **Joint selection** — one candidate per part is chosen by coordinate
   descent on a single area-currency objective:

   `net = marginal soft coverage of the reference − β · floating_fraction · (part_area · s² / ref_area)`

   The reference surface is the union of the garments, so the correct
   joint solution tiles it. This objective is scale-fair where per-part
   scores are not: absolute fit error favors shrunken fits hiding in
   surface folds, relative error favors inflated fits "tarping" large
   regions - but a shrunken fit explains almost no area, and an inflated
   fit pays s²-growing cost for the surface it brings that lands nowhere.
   The body participates as a down-weighted background layer (it sits
   under the clothes but its claim on the skin stops garments from
   freeloading on the face/hands), and parts left with near-zero marginal
   coverage get a second RANSAC pass against only the still-unclaimed
   surface (this also resolves both shoes landing on the same foot).

### `--method landmarks` (fast, avatar-convention specific)

Hand-crafted geometric landmarks on the y-up T-pose avatar (head top, hand
tips, crotch detected by the silhouette splitting into two legs, per-foot
centers) matched to per-garment-type landmarks (sleeve tips, crotch/hem,
per-shoe centers and sole), giving the similarity initialization directly.
Shoes are split at the x gap and assigned to feet jointly by ICP error.

## Usage

```bash
pip install -r requirements.txt
python -m registration.register --data data --out output            # fpfh
python -m registration.register --data data --out output --method landmarks
```

## GPU acceleration

The two hot spots - nearest-neighbor queries inside trimmed ICP / coverage
scoring, and RANSAC hypothesis testing - run on the GPU automatically when
PyTorch with CUDA is installed (`pip install torch`):

- NN queries become exact chunked brute-force distance minimization.
- FPFH+RANSAC is replaced by a batched GPU RANSAC: mutual feature matches,
  ~150k minimal sets solved by batched closed-form Umeyama, inliers counted
  in parallel (`registration/backend.py`).

Control with `REG_DEVICE=auto|cuda|cpu` (default `auto`). Without torch or
a GPU everything falls back to scipy/Open3D on CPU; `--method fpfh` then
takes minutes rather than seconds (`--method landmarks` runs in under a
minute on CPU and is a good fast path when the avatar follows the y-up
T-pose convention).

## Outputs (`output/`)

- `registered/<name>_registered.glb` — each garment transformed into the
  reference avatar's frame (materials/UVs preserved; transforms are baked
  into the vertices).
- `transforms.json` — per-part 4x4 similarity matrices, scales and mean
  trimmed-ICP errors (units are the reference's normalized frame, where the
  avatar is ~1.0 tall).
- `assembled_avatar.glb` — bald body + all registered garments combined,
  for a quick visual check against the reference.
- `previews/registration_overlay.png` — front/side overlays of every
  registered garment (red) on the reference avatar (gray).

## Code layout

- `registration/landmarks.py` — heuristic landmark extraction for y-up
  T-pose humanoids and garments.
- `registration/icp.py` — Umeyama similarity solve and trimmed scaled ICP.
- `registration/register.py` — per-garment initializers, the registration
  driver and CLI.
