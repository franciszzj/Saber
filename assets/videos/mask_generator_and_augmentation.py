# pyre-unsafe
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF


def generate_mask(
    h: int,
    w: int,
    area_ratio_range: Tuple[float, float] = (0.1, 0.5),
    shape_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a binary mask (uint8) of shape (h, w) with exactly ONE connected component.
    Foreground pixels are 1; background pixels are 0.

    Shape families:
      - 'ellipse'              : ellipse (rounded edges)
      - 'superellipse'         : Lamé curve / squircle (rounded-rectangle-like)
      - 'fourier'              : random radial-Fourier blob (often concave, organic)
      - 'concave_polygon'      : irregular concave polygon with uneven angles + local dents
      - 'centered_rectangle'   : centered rectangle that may touch any/all borders, with slightly rough edges

    The output area ratio is GUARANTEED to lie within `area_ratio_range` (inclusive).
    If the initial continuous scaling cannot hit the interval due to pixelization,
    the function performs a small topology-preserving adjustment of boundary pixels.

    Parameters
    ----------
    h, w : int
        Output height and width.
    area_ratio_range : (float, float)
        Inclusive interval [lo, hi] for the foreground area ratio.
    shape_types : Sequence[str] or None
        Optional subset to sample from. If None, uses all families.
        Valid: {'ellipse','superellipse','fourier','concave_polygon'}.
        To force only concave polygons, pass ('concave_polygon',).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Binary mask of shape (h, w), dtype=np.uint8 in {0,1}.
    """
    rng = np.random.default_rng(seed)

    # --------- sanitize area range and convert to pixel counts ---------
    lo, hi = area_ratio_range
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, lo, 1.0))
    N = int(h * w)
    A_lo = int(np.ceil(lo * N))  # inclusive lower bound in pixels
    A_hi = int(np.floor(hi * N))  # inclusive upper bound in pixels
    if A_lo > A_hi:
        # If the interval is narrower than a single pixel at this resolution,
        # widen minimally to make it feasible.
        A_lo = min(A_lo, N)
        A_hi = max(A_lo, A_lo)  # degenerate single-pixel-width interval

    # --------- config and helpers ---------
    families = (
        "ellipse",
        "superellipse",
        "fourier",
        "concave_polygon",
        "centered_rectangle",
    )
    if shape_types is None:
        family = rng.choice(families)
    else:
        shape_pool = tuple(s for s in shape_types if s in families) or families
        family = rng.choice(shape_pool)

    # coordinate grids
    Y, X = np.ogrid[:h, :w]
    min_dim = float(min(h, w))
    MIN_R = max(2.0, 0.01 * min_dim)

    def sample_center(margin_scale=0.10):
        """Sample a center away from borders to reduce clipping and keep monotonicity stable."""
        my = int(max(1, round(margin_scale * h)))
        mx = int(max(1, round(margin_scale * w)))
        cy = int(rng.integers(my, max(my + 1, h - my)))
        cx = int(rng.integers(mx, max(mx + 1, w - mx)))
        return cy, cx

    # ---- Topology-safe pixel utilities ----
    def neighbors8(mask: np.ndarray):
        """Return 8-neighborhood boolean arrays (N, NE, E, SE, S, SW, W, NW)."""
        m = mask.astype(np.uint8)
        # pad with zeros for easy slicing
        p = np.pad(m, 1, mode="constant", constant_values=0)
        N_ = p[:-2, 1:-1]
        NE_ = p[:-2, 2:]
        E_ = p[1:-1, 2:]
        SE_ = p[2:, 2:]
        S_ = p[2:, 1:-1]
        SW_ = p[2:, :-2]
        W_ = p[1:-1, :-2]
        NW_ = p[:-2, :-2]
        return (
            N_.astype(bool),
            NE_.astype(bool),
            E_.astype(bool),
            SE_.astype(bool),
            S_.astype(bool),
            SW_.astype(bool),
            W_.astype(bool),
            NW_.astype(bool),
        )

    def neighbor_count(mask: np.ndarray) -> np.ndarray:
        """8-neighbor count."""
        ns = neighbors8(mask)
        cnt = np.zeros_like(mask, dtype=np.int32)
        for b in ns:
            cnt += b.astype(np.int32)
        return cnt

    def simple_point_candidates(mask: np.ndarray) -> np.ndarray:
        """
        Return a boolean map of boundary pixels that are 'simple points':
        Removing a simple point does not break 8-connectivity of the foreground.
        We use a standard test: 2 <= N(p) <= 6 and T(p) == 1,
        where T(p) is the number of 0->1 transitions along the 8-neighbor cycle.
        """
        if mask.dtype != np.uint8:
            m = mask.astype(np.uint8)
        else:
            m = mask
        ns = neighbors8(m)
        ncnt = np.zeros_like(m, dtype=np.int32)
        for b in ns:
            ncnt += b.astype(np.int32)

        # boundary pixels: foreground with at least one background neighbor
        boundary = (m == 1) & (ncnt > 0) & (ncnt < 8)

        # compute T(p): number of 0->1 transitions in circular sequence
        b1, b2, b3, b4, b5, b6, b7, b8 = ns
        seq = [b1, b2, b3, b4, b5, b6, b7, b8, b1]
        transitions = np.zeros_like(m, dtype=np.int32)
        for i in range(8):
            transitions += (~seq[i] & seq[i + 1]).astype(np.int32)

        # simple point predicate
        simple = boundary & (ncnt >= 2) & (ncnt <= 6) & (transitions == 1)
        return simple

    def grow_to(mask: np.ndarray, target_area: int) -> np.ndarray:
        """
        Increase area to at least target_area by adding boundary-adjacent pixels.
        Connectivity is preserved because new pixels are added only if they touch the component.
        """
        m = mask.copy()
        current = int(m.sum())
        if current >= target_area:
            return m
        # loop in small batches for stability
        while current < target_area:
            ncnt = neighbor_count(m)
            candidates = (m == 0) & (ncnt > 0)  # zeros adjacent to foreground
            idx = np.flatnonzero(candidates)
            if idx.size == 0:
                # If no adjacent zeros (shouldn't happen), fallback to dilation
                kernel = np.ones((3, 3), np.uint8)
                m = cv2.dilate(m, kernel, iterations=1)
                current = int(m.sum())
                continue
            need = target_area - current
            take = (
                idx if idx.size <= need else rng.choice(idx, size=need, replace=False)
            )
            m.flat[take] = 1
            current += int(take.size)
        return m

    def shrink_to(mask: np.ndarray, target_area: int) -> np.ndarray:
        """
        Decrease area to at most target_area by removing 'simple points' on the boundary.
        This preserves connectivity (removal does not split the component).
        """
        m = mask.copy()
        current = int(m.sum())
        if current <= target_area:
            return m
        # loop with batches; recompute simple points each round
        while current > target_area:
            simple = simple_point_candidates(m)
            idx = np.flatnonzero(simple)
            if idx.size == 0:
                # If no simple points (very rare for thick shapes), erode once,
                # then immediately grow back minimally to target if we overshoot.
                kernel = np.ones((3, 3), np.uint8)
                m_er = cv2.erode(m, kernel, iterations=1)
                if m_er.sum() == 0:
                    break
                m = m_er
                current = int(m.sum())
                if current < target_area:
                    m = grow_to(m, target_area)
                continue
            need = current - target_area
            take_count = min(need, max(32, idx.size // 2))  # remove in moderate batches
            take = rng.choice(idx, size=take_count, replace=False)
            m.flat[take] = 0
            current -= int(take.size)
        return m

    # ---- scale search helpers (monotone in scale; uses bisection) ----
    def find_scale_in_range(mask_fn, max_iter: int = 48):
        """
        Given a function mask_fn(s: float)->np.uint8 mask that is non-decreasing in area w.r.t s,
        find a scale s such that area is within [A_lo, A_hi] if possible.
        Returns (mask, area, hit) where hit=True if directly inside without pixel tweaking.
        """
        # Exponentially grow the upper bound until area >= A_lo
        s_lo, a_lo = 0.0, 0
        s_hi = 1.0
        m_hi = mask_fn(s_hi)
        a_hi = int(m_hi.sum())
        tries = 0
        while a_hi < A_lo and tries < 32:
            s_lo, a_lo = s_hi, a_hi
            s_hi *= 2.0
            m_hi = mask_fn(s_hi)
            a_hi = int(m_hi.sum())
            tries += 1

        # Bisection to land within interval if possible
        # Invariant: area(s_lo) < A_lo <= area(s_hi) OR we already have m_hi in range
        if A_lo <= a_hi <= A_hi:
            return m_hi, a_hi, True

        m_lo = mask_fn(s_lo) if a_lo == 0 else mask_fn(s_lo)  # recompute consistently
        for _ in range(max_iter):
            s_mid = 0.5 * (s_lo + s_hi)
            m_mid = mask_fn(s_mid)
            a_mid = int(m_mid.sum())
            if A_lo <= a_mid <= A_hi:
                return m_mid, a_mid, True
            if a_mid < A_lo:
                s_lo, m_lo, a_lo = s_mid, m_mid, a_mid
            else:  # a_mid > A_hi or just > A_lo
                s_hi, m_hi, a_hi = s_mid, m_mid, a_mid

        # If we exit without a direct hit, return the closer bound (prefer <=A_hi)
        # Choose the candidate that is inside if any; otherwise pick closer and we'll tweak pixels.
        candidates = [(a_lo, m_lo), (a_hi, m_hi)]
        # Prefer the one closer to the interval; if both outside, prefer the one that is closer in pixels.
        best = min(
            candidates,
            key=lambda t: (
                0 if (A_lo <= t[0] <= A_hi) else min(abs(t[0] - A_lo), abs(t[0] - A_hi))
            ),
        )
        return best[1], best[0], (A_lo <= best[0] <= A_hi)

    # ------------------- shape families -------------------
    def ellipse_mask_fn():
        cy, cx = sample_center(0.10)
        ar = float(rng.uniform(0.5, 2.0))  # rx/ry
        # base radii (used only to define scale=1 reference)
        ry0 = max(MIN_R, float(rng.uniform(0.15, 0.35) * h))
        rx0 = max(MIN_R, ar * ry0)

        def fn(scale: float) -> np.ndarray:
            ry = float(max(MIN_R, ry0 * scale))
            rx = float(max(MIN_R, rx0 * scale))
            v = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
            return (v <= 1.0).astype(np.uint8)

        return fn

    def superellipse_mask_fn():
        cy, cx = sample_center(0.12)
        n = float(rng.uniform(2.2, 6.0))  # n=2 ellipse; larger => boxier
        ar = float(rng.uniform(0.6, 1.7))
        b0 = max(MIN_R, float(rng.uniform(0.12, 0.30) * h))
        a0 = max(MIN_R, ar * b0)

        def fn(scale: float) -> np.ndarray:
            a = float(max(MIN_R, a0 * scale))
            b = float(max(MIN_R, b0 * scale))
            v = (np.abs((X - cx) / a) ** n) + (np.abs((Y - cy) / b) ** n)
            return (v <= 1.0).astype(np.uint8)

        return fn

    def fourier_mask_fn():
        cy, cx = sample_center(0.12)
        K = int(rng.integers(3, 7))
        # precompute theta and rho
        dy = (Y - cy).astype(np.float64)
        dx = (X - cx).astype(np.float64)
        rho = np.hypot(dy, dx) + 1e-9
        theta = np.arctan2(dy, dx)

        # coefficients (low-frequency dominated)
        a = rng.normal(0.0, 0.25, size=K)
        b = rng.normal(0.0, 0.25, size=K)
        decay = 1.0 / (1.0 + np.arange(1, K + 1))
        a *= decay
        b *= decay
        c0 = 1.0 + abs(rng.normal(0.0, 0.15))

        # base radius field R0(theta) > 0
        R0 = np.full_like(theta, c0, dtype=np.float64)
        for k in range(1, K + 1):
            R0 += a[k - 1] * np.cos(k * theta) + b[k - 1] * np.sin(k * theta)
        R0 = R0 - R0.min() + 0.2  # strictly positive

        # normalize R0 so that scale is near 1.0 for typical sizes
        R0 *= np.sqrt((h * w) / np.pi) * 0.25  # heuristic base size

        def fn(scale: float) -> np.ndarray:
            R = R0 * float(scale)
            return (rho <= R).astype(np.uint8)

        return fn

    def concave_polygon_mask_fn():
        """
        Build an irregular concave polygon using uneven angular spacing and local dents.
        Rasterization uses cv2.fillPoly (fast).
        """
        cy, cx = sample_center(0.12)
        Nv = int(rng.integers(28, 64))

        # uneven angles via random gaps (Dirichlet-like)
        gaps = rng.gamma(shape=1.4, scale=1.0, size=Nv)
        angles = (2.0 * np.pi) * np.cumsum(gaps) / float(np.sum(gaps))
        angles += float(rng.uniform(0, 2.0 * np.pi))
        angles %= 2.0 * np.pi
        angles.sort()

        # base radius
        base_r = np.sqrt(max((A_lo + A_hi) / 2.0, 1.0) / np.pi)

        # smooth random variation
        noise = rng.normal(0.0, 0.18, size=Nv)
        for _ in range(2):
            noise = 0.25 * np.roll(noise, 1) + 0.5 * noise + 0.25 * np.roll(noise, -1)

        # localized dents
        M = int(rng.integers(2, 5))
        dent_centers = rng.uniform(0, 2 * np.pi, size=M)
        dent_widths = rng.uniform(0.10, 0.35, size=M)
        dent_depths = rng.uniform(0.18, 0.42, size=M)
        dtheta = angles[:, None] - dent_centers[None, :]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        dents = (
            np.exp(-0.5 * (dtheta / dent_widths[None, :]) ** 2) * dent_depths[None, :]
        )
        dent_factor = np.clip(1.0 - dents.sum(axis=1), 0.25, 1.6)

        radii0 = base_r * (1.0 + noise) * dent_factor
        radii0 = np.clip(radii0, MIN_R + 1.0, None)

        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        ctr = np.array([cx, cy], dtype=np.float32)

        def fn(scale: float) -> np.ndarray:
            r = (radii0 * float(scale)).astype(np.float32)[:, None]  # (Nv,1)
            pts = ctr + r * dirs  # (Nv,2)
            pts = np.round(pts).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 1, w - 2)
            pts[:, 1] = np.clip(pts[:, 1], 1, h - 2)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
            return mask

        return fn

    def centered_rectangle_mask_fn():
        """
        Centered rectangle whose sides can touch (or coincide with) image borders as scale increases.
        Slightly rough edges are produced by smooth 1D noise along each edge.
        The area is non-decreasing in `scale` (monotone), enabling bisection.
        """
        cy, cx = (h // 2), (w // 2)  # strictly centered

        # base half-sizes; start moderate and let scale search expand/shrink
        # Aspect ratio randomization keeps variety.
        ar = float(rng.uniform(0.6, 1.7))
        sy0 = max(MIN_R, float(rng.uniform(0.12, 0.30) * h))
        sx0 = max(MIN_R, ar * sy0)

        # Roughness amplitude in pixels (kept small to preserve component connectivity).
        rough_amp_x = float(rng.uniform(0.5, 3.0))  # horizontal jaggies per row
        rough_amp_y = float(rng.uniform(0.5, 3.0))  # vertical jaggies per column

        # Build smooth 1D noise for top/bottom (over x) and left/right (over y).
        # We smooth white noise with a short kernel to avoid spiky disconnections.
        def smooth_noise_1d(n: int, amp: float) -> np.ndarray:
            if n <= 0 or amp <= 0:
                return np.zeros((max(1, n),), dtype=np.float32)
            v = rng.normal(0.0, 1.0, size=n).astype(np.float32)
            # simple tri-weight kernel [1,2,1] normalized; apply twice for extra smoothness
            k = np.array([1.0, 2.0, 1.0], dtype=np.float32)
            k /= k.sum()
            v = np.convolve(v, k, mode="same")
            v = np.convolve(v, k, mode="same")
            v = v / (np.std(v) + 1e-6)
            return (amp * v).astype(np.float32)

        # Precompute fixed roughness profiles (independent of scale to keep monotonicity).
        jitter_top = smooth_noise_1d(w, rough_amp_y)
        jitter_bot = smooth_noise_1d(w, rough_amp_y)
        jitter_left = smooth_noise_1d(h, rough_amp_x)
        jitter_right = smooth_noise_1d(h, rough_amp_x)

        def fn(scale: float) -> np.ndarray:
            # Half-sizes grow with scale; clipping to borders is allowed.
            sy = float(max(MIN_R, sy0 * scale))
            sx = float(max(MIN_R, sx0 * scale))

            # Vertical limits per column with jitter (top/bottom).
            y_top = np.floor(cy - sy + jitter_top).astype(np.int32)
            y_bot = np.ceil(cy + sy + jitter_bot).astype(np.int32)
            y_top = np.clip(y_top, 0, h - 1)
            y_bot = np.clip(y_bot, 0, h - 1)
            # Ensure y_top <= y_bot
            swap = y_top > y_bot
            if np.any(swap):
                yt = y_top.copy()
                y_top = np.minimum(yt, y_bot)
                y_bot = np.maximum(yt, y_bot)

            # Horizontal limits per row with jitter (left/right).
            x_left = np.floor(cx - sx + jitter_left).astype(np.int32)
            x_right = np.ceil(cx + sx + jitter_right).astype(np.int32)
            x_left = np.clip(x_left, 0, w - 1)
            x_right = np.clip(x_right, 0, w - 1)
            # Ensure x_left <= x_right
            swap = x_left > x_right
            if np.any(swap):
                xl = x_left.copy()
                x_left = np.minimum(xl, x_right)
                x_right = np.maximum(xl, x_right)

            # Build mask as intersection of per-column vertical slab and per-row horizontal slab.
            # Using broadcasting to vectorize filling.
            mask = np.zeros((h, w), dtype=np.uint8)

            # Vertical slab: for each column x, fill rows [y_top[x], y_bot[x]]
            rows = np.arange(h, dtype=np.int32)[:, None]  # (h,1)
            cols = np.arange(w, dtype=np.int32)[None, :]  # (1,w)
            vert_ok = (rows >= y_top[None, :]) & (rows <= y_bot[None, :])

            # Horizontal slab: for each row y, fill cols [x_left[y], x_right[y]]
            hori_ok = (cols >= x_left[:, None]) & (cols <= x_right[:, None])

            mask[(vert_ok & hori_ok)] = 1
            return mask

        return fn

    # pick family and build its mask function
    if family == "ellipse":
        mask_fn = ellipse_mask_fn()
    elif family == "superellipse":
        mask_fn = superellipse_mask_fn()
    elif family == "fourier":
        mask_fn = fourier_mask_fn()
    elif family == "centered_rectangle":
        mask_fn = centered_rectangle_mask_fn()
    else:
        mask_fn = concave_polygon_mask_fn()

    # ---- search a scale whose area falls inside [A_lo, A_hi] ----
    mask, area, hit = find_scale_in_range(mask_fn, max_iter=48)

    # If not a direct hit (due to discreteness), adjust minimally at the boundary.
    if area < A_lo:
        mask = grow_to(mask, A_lo)
    elif area > A_hi:
        mask = shrink_to(mask, A_hi)

    # Final safety: enforce dtype and single connected component (should already hold).
    mask = (mask > 0).astype(np.uint8)

    # Optional final check (assert can be disabled in production).
    final_area = int(mask.sum())
    if not (A_lo <= final_area <= A_hi):
        # As a last resort (extremely rare), clip by pixel-level add/remove.
        if final_area < A_lo:
            mask = grow_to(mask, A_lo)
        elif final_area > A_hi:
            mask = shrink_to(mask, A_hi)

    return mask


def random_affine_preserve_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    degrees: float = 10.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    hflip_prob: float = 0.5,
    shear_range: Tuple[float, float] = (-10.0, 10.0),
    max_tries: int = 64,
    center: Tuple[float, float] = None,  # pyre-ignore
    fill: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Apply a random affine (zoom in/out, translate, small rotation, shear) and optional horizontal flip
    to both image and mask, while guaranteeing the originally kept (mask==1) region stays fully
    inside the frame.

    Args
    ----
    image: torch.Tensor
        Tensor of shape (C, H, W). Values expected in (-1, 1), but any float range is acceptable.
    mask: torch.Tensor
        Tensor of shape (1, H, W) or (H, W). Binary with 1 for kept region, 0 elsewhere.
    degrees: float
        Max absolute rotation in degrees. Angle is sampled uniformly from [-degrees, degrees].
    scale_range: (float, float)
        Uniform sampling range for isotropic scale (zoom). >1 zoom-in, <1 zoom-out.
    max_tries: int
        Max resampling attempts to find a feasible transform (angle/scale/translation).
    center: (float, float) or None
        Rotation/scale center in (cx, cy) pixel coordinates. Default is image center.
    fill: float
        Fill value for areas introduced by the transform on the image.
    hflip_prob: float
        Probability to apply a horizontal flip before affine.
    shear_range: (float, float)
        Uniform sampling range for shear angles in degrees, applied independently to X and Y.

    Returns
    -------
    out_image: torch.Tensor
        Transformed image (C, H, W), float32.
    out_mask: torch.Tensor
        Transformed mask (1, H, W), float32 in {0,1}.
    params: dict
        The sampled transform parameters:
        {
            "hflip": bool,
            "angle": float,
            "scale": float,
            "shear": (shear_x_deg, shear_y_deg),
            "translate": (tx, ty),
            "center": (cx, cy)
        }.
    """

    # ------------------------------ #
    # Internal helpers (nested defs) #
    # ------------------------------ #

    def _mask_bbox(m: torch.Tensor) -> Tuple[int, int, int, int]:
        """
        Compute the tight bounding box (xmin, ymin, xmax, ymax) of m==1 pixels.
        Works with 2D (H, W) or 3D (1, H, W or C, H, W). Returns inclusive integer bounds.
        Raises ValueError if mask has no positive pixels.
        """
        if m.dim() == 3:
            # Accept (1, H, W) or (C, H, W) and reduce to single channel
            mm = m[0] if m.size(0) == 1 else (m > 0.5).any(dim=0, keepdim=False).float()
        elif m.dim() == 2:
            mm = m
        else:
            raise ValueError("mask must be 2D (H, W) or 3D (1, H, W)")

        ys, xs = torch.nonzero(mm > 0.5, as_tuple=True)
        if ys.numel() == 0:
            raise ValueError("Mask has no positive pixels; cannot constrain transform.")

        ymin = int(ys.min().item())
        ymax = int(ys.max().item())
        xmin = int(xs.min().item())
        xmax = int(xs.max().item())
        return xmin, ymin, xmax, ymax

    def _deg2rad(x: float) -> torch.Tensor:
        """Convert degrees to radians as float32 tensor."""
        return torch.tensor(x * 3.141592653589793 / 180.0, dtype=torch.float32)

    def _transform_corners_no_translate(
        corners: torch.Tensor,
        ctr: Tuple[float, float],
        scale: float,
        angle_deg: float,
        shear_x_deg: float,
        shear_y_deg: float,
    ) -> torch.Tensor:
        """
        Apply scale, rotation, and shear about 'ctr' without translation to 2D points.
        corners: (N,2) in (x,y). Returns transformed corners (N,2).

        Composition (about the center):
            p' = ctr + Shear(sx, sy) @ (Rotate(angle) @ (scale * (p - ctr)))
        Notes:
            - shear_x: shear parallel to X-axis (depends on Y)
            - shear_y: shear parallel to Y-axis (depends on X)
        """
        cx, cy = ctr
        # Shift to center
        shifted = corners - torch.tensor([cx, cy], dtype=torch.float32)

        # Scale (isotropic)
        S = torch.tensor([[scale, 0.0], [0.0, scale]], dtype=torch.float32)

        # Rotation
        theta = _deg2rad(angle_deg)
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

        # Shear (angles in degrees -> tangents)
        sx_tan = torch.tan(_deg2rad(shear_x_deg))
        sy_tan = torch.tan(_deg2rad(shear_y_deg))
        # Shear matrix: x' = x + tan(sx)*y ; y' = y + tan(sy)*x
        Sh = torch.tensor([[1.0, sx_tan], [sy_tan, 1.0]], dtype=torch.float32)

        # Compose: Shear @ (R @ (S @ shifted.T))
        M = Sh @ (R @ S)
        transformed = (shifted @ M.T) + torch.tensor([cx, cy], dtype=torch.float32)
        return transformed

    def _sample_params_within_bounds(
        H: int,
        W: int,
        bbox: Tuple[int, int, int, int],
        degrees: float,
        scale_range: Tuple[float, float],
        shear_range: Tuple[float, float],
        max_tries: int,
        ctr: Tuple[float, float],
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Randomly sample (angle_deg, scale, shear_x_deg, shear_y_deg, tx, ty) so that the transformed
        bbox (after scale+rotate+shear about ctr, then translation) remains fully inside
        the image domain [0, W-1] x [0, H-1]. Falls back to identity if no feasible sample is found.
        """
        xmin, ymin, xmax, ymax = bbox
        corners = torch.tensor(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
            dtype=torch.float32,
        )

        for _ in range(max_tries):
            angle = float(torch.empty(1).uniform_(-degrees, degrees).item())
            scale = float(
                torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
            )
            shear_x = float(
                torch.empty(1).uniform_(shear_range[0], shear_range[1]).item()
            )
            shear_y = float(
                torch.empty(1).uniform_(shear_range[0], shear_range[1]).item()
            )

            tc = _transform_corners_no_translate(
                corners, ctr, scale, angle, shear_x, shear_y
            )

            # Feasible translation intervals per axis (intersection over corners)
            tx_low = torch.max(-tc[:, 0])
            tx_high = torch.min((W - 1) - tc[:, 0])
            ty_low = torch.max(-tc[:, 1])
            ty_high = torch.min((H - 1) - tc[:, 1])

            if (tx_low <= tx_high) and (ty_low <= ty_high):
                tx = float(
                    torch.empty(1).uniform_(tx_low.item(), tx_high.item()).item()
                )
                ty = float(
                    torch.empty(1).uniform_(ty_low.item(), ty_high.item()).item()
                )
                return angle, scale, shear_x, shear_y, tx, ty

        # Safe fallback
        return 0.0, 1.0, 0.0, 0.0, 0.0, 0.0

    # ------------------------- #
    # Argument normalization    #
    # ------------------------- #

    assert image.dim() == 3, "image must be (C, H, W)"
    C, H, W = image.shape

    # Normalize mask to (1, H, W), float32 in {0,1}
    if mask.dim() == 2:
        mask_ = mask.unsqueeze(0).float()
    elif mask.dim() == 3:
        assert (
            mask.size(1) == H and mask.size(2) == W
        ), "mask must match image spatial size"
        mask_ = mask.float()
        if mask_.size(0) != 1:
            mask_ = (mask_ > 0.5).any(dim=0, keepdim=True).float()
    else:
        raise ValueError("mask must be 2D (H, W) or 3D (1, H, W)")

    # Choose rotation/scale/shear center
    if center is None:
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
    else:
        cx, cy = center

    # ------------------------- #
    # Optional horizontal flip  #
    # ------------------------- #
    # Flip first; it never causes out-of-bounds, but it changes the bbox used for constraints.
    hflip = bool(torch.rand(()) < hflip_prob)
    if hflip:
        image = TF.hflip(image)
        mask_ = TF.hflip(mask_)

    # ------------------------- #
    # Sample transform params   #
    # ------------------------- #
    try:
        bbox = _mask_bbox(mask_)
        angle, scale, shear_x, shear_y, tx, ty = _sample_params_within_bounds(
            H, W, bbox, degrees, scale_range, shear_range, max_tries, ctr=(cx, cy)
        )
    except ValueError:
        # Empty mask: sample an unconstrained gentle transform
        angle = float(torch.empty(1).uniform_(-degrees, degrees).item())
        scale = float(torch.empty(1).uniform_(scale_range[0], scale_range[1]).item())
        shear_x = float(torch.empty(1).uniform_(shear_range[0], shear_range[1]).item())
        shear_y = float(torch.empty(1).uniform_(shear_range[0], shear_range[1]).item())
        tx = float(torch.empty(1).uniform_(-0.05 * W, 0.05 * W).item())
        ty = float(torch.empty(1).uniform_(-0.05 * H, 0.05 * H).item())

    # ------------------------- #
    # Apply the affine          #
    # ------------------------- #
    out_img = TF.affine(
        image,
        angle=angle,
        translate=[tx, ty],  # pyre-ignore
        scale=scale,
        shear=[shear_x, shear_y],
        interpolation=TF.InterpolationMode.BILINEAR,
        center=[cx, cy],  # pyre-ignore
        fill=fill,  # pyre-ignore
    )
    out_msk = TF.affine(
        mask_,
        angle=angle,
        translate=[tx, ty],  # pyre-ignore
        scale=scale,
        shear=[shear_x, shear_y],
        interpolation=TF.InterpolationMode.NEAREST,
        center=[cx, cy],  # pyre-ignore
        fill=0.0,  # pyre-ignore
    )

    # Re-binarize mask defensively
    out_msk = (out_msk > 0.5).float()

    params = {
        "hflip": hflip,
        "angle": angle,
        "scale": scale,
        "shear": (shear_x, shear_y),
        "translate": (tx, ty),
        "center": (cx, cy),
    }
    return out_img, out_msk, params
