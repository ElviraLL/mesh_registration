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

For each garment:

1. **Landmarks** — geometric landmarks are extracted from the y-up T-pose
   reference avatar (head top, hand tips, crotch, ankles, per-foot centers,
   neck, torso center) and matching landmarks from the garment in its own
   frame (e.g. sleeve tips for tops, crotch/hem for bottoms, per-shoe
   centers and sole for shoes).
2. **Similarity initialization** — the landmark correspondences give a
   uniform scale + translation (plus 180° yaw candidates for orientation
   ambiguity and a few scale perturbations).
3. **Trimmed scaled ICP** — the initialization is refined against surface
   samples of the reference avatar. Because the reference already contains
   the same garment baked in, the garment snaps onto its counterpart region.
   The target is cropped to the garment's moving bounding box and only the
   best 70% of correspondences are used each iteration, so the rest of the
   avatar does not pull the garment away.

Shoes are special-cased: a single similarity transform cannot match both
the shoe size and the pair spacing (the generated pair is spaced differently
than the avatar's stance), so the pair is split at the x gap between the two
shoes and each shoe is registered to its own foot. The left/right assignment
is chosen jointly by total ICP error.

## Usage

```bash
pip install -r requirements.txt
python -m registration.register --data data --out output
```

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
