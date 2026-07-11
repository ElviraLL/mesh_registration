"""Landmark-based registration of separately generated garment meshes onto a
reference avatar mesh (the "one piece" avatar with all garments baked in).

Pipeline per garment:
1. Extract anatomical landmarks from the reference avatar (head top, hands,
   crotch, ankles, feet, ...) using geometric heuristics on the y-up T-pose.
2. Extract matching landmarks on the garment in its own normalized frame.
3. Solve a similarity transform (uniform scale + rotation + translation) from
   the landmark correspondences (Umeyama).
4. Refine with trimmed scaled ICP against the reference surface - the
   reference already contains the same garment baked in, so the garment
   snaps onto its counterpart region.
"""

from .landmarks import avatar_landmarks
from .icp import umeyama, trimmed_icp
from .register import register_all
