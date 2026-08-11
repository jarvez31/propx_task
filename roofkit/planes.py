"""Roof structure from the height model, via sequential RANSAC plane fitting.

Every roof pixel is a 3D point (x, y, height). RANSAC finds the plane the most pixels agree
with; we bank it, drop its pixels, and repeat. The facets ARE the roof structure: how many
planes and which way they face gives the TYPE, and pixels that fit no plane but sit above one
are the SUPERSTRUCTURES (chimneys / dormers / HVAC).
"""
import numpy as np
from scipy.ndimage import label


def fit_plane(xy, z):
    """Least-squares plane z = a*x + b*y + c through points; returns (a, b, c)."""
    A = np.c_[xy, np.ones(len(xy))]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coef


def ransac_plane(xy, z, tol=0.3, iters=200, seed=42):
    """Boolean inlier mask for the single plane the most points agree with (within `tol` metres)."""
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(z), bool)
    for _ in range(iters):
        sample = rng.choice(len(z), 3, replace=False)        # 3 points define a plane
        a, b, c = fit_plane(xy[sample], z[sample])
        inliers = np.abs(z - (a * xy[:, 0] + b * xy[:, 1] + c)) < tol
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    return best_inliers


def fit_roof_planes(roof, ndsm, dsm_transform, tol=0.3, min_facet_px=30, max_planes=6):
    """Peel planar facets off the roof one at a time. Returns (planes, coverage)."""
    ys, xs = np.where(roof)
    px = abs(dsm_transform.a)                                 # 0.5 m
    xy = np.c_[xs * px, ys * px]                              # metre coords (relative origin is fine)
    z = ndsm[ys, xs].astype("float64")
    remaining = np.ones(len(z), bool)
    planes = []
    for _ in range(max_planes):
        if remaining.sum() < min_facet_px:
            break
        local = ransac_plane(xy[remaining], z[remaining], tol=tol)
        if local.sum() < min_facet_px:
            break
        gidx = np.where(remaining)[0][local]                 # local inliers -> global point indices
        a, b, c = fit_plane(xy[gidx], z[gidx])
        slope  = float(np.degrees(np.arctan(np.hypot(a, b))))
        aspect = float(np.degrees(np.arctan2(-a, b)) % 360)  # same convention as the DSM gradient aspect
        planes.append({"coef": (a, b, c), "slope_deg": round(slope, 1),
                       "aspect_deg": round(aspect, 0), "area_m2": round(len(gidx) * px * px, 1)})
        remaining[gidx] = False
    coverage = 1.0 - remaining.sum() / len(z)                # fraction of roof explained by planes
    return planes, round(float(coverage), 2)


def roof_type_from_planes(planes, coverage, flat_thresh=10.0):
    """Classify the roof from its facets. Confidence = fraction of roof explained by planes."""
    facets = [p for p in planes if p["area_m2"] >= 5]        # ignore slivers
    if not facets:
        return "unresolved", 0.2
    if max(p["slope_deg"] for p in facets) < flat_thresh:
        return "flat", coverage
    pitched = [p for p in facets if p["slope_deg"] >= flat_thresh]
    rtype = {1: "mono_pitch", 2: "gable", 3: "hipped", 4: "hipped_pyramidal"}.get(len(pitched), "complex")
    return rtype, coverage


def plane_base_image(roof, ndsm, dsm_transform, planes):
    """Per roof pixel, height of its NEAREST fitted plane -> 2D array (nan off-roof)."""
    ys, xs = np.where(roof)
    px = abs(dsm_transform.a)
    x, y, z = xs * px, ys * px, ndsm[ys, xs]
    plane_z = np.array([a * x + b * y + c for (a, b, c) in (p["coef"] for p in planes)])
    base = plane_z[np.argmin(np.abs(plane_z - z), axis=0), np.arange(len(z))]
    out = np.full(roof.shape, np.nan, "float32")
    out[ys, xs] = base
    return out


def find_superstructures(roof, ndsm, dsm_transform, planes):
    """Pixels sitting >1 m above their nearest fitted plane -> chimneys / dormers / HVAC."""
    if not planes:
        return []
    residual = np.zeros(roof.shape, "float32")
    residual[roof] = ndsm[roof] - plane_base_image(roof, ndsm, dsm_transform, planes)[roof]
    bumps = (residual > 1.0) & roof
    labelled, n = label(bumps)
    px = abs(dsm_transform.a)
    objects = []
    for i in range(1, n + 1):
        comp = labelled == i
        area = comp.sum() * px * px
        height = float(residual[comp].max())
        if area < 1.0 or height > 8.0:                       # <1 m^2 = noise, >8 m = adjacent taller wing
            continue
        kind = "chimney" if (area < 3 and height > 1.2) else \
               ("large_superstructure" if area >= 6 else "dormer_or_small")
        objects.append({"class": kind, "area_m2": round(area, 1), "height_m": round(height, 1)})
    return objects
