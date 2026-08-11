"""Pure-function tests for the RANSAC plane logic — no network, no torch."""
import numpy as np

from roofkit.planes import fit_plane, ransac_plane, roof_type_from_planes


def test_fit_plane_recovers_coefficients():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, (200, 2))
    a, b, c = 0.5, -0.3, 2.0
    z = a * xy[:, 0] + b * xy[:, 1] + c
    assert np.allclose(fit_plane(xy, z), [a, b, c], atol=1e-6)


def test_ransac_ignores_outliers():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10, (300, 2))
    z = 0.2 * xy[:, 0] + 0.1 * xy[:, 1] + 1.0
    z[:60] += 5.0                                   # 20% gross outliers (a bump above the plane)
    inliers = ransac_plane(xy, z, tol=0.3, iters=300)
    assert inliers[60:].mean() > 0.95               # nearly all true-plane points recovered
    assert inliers[:60].mean() < 0.2                # bump mostly excluded


def test_roof_type_mapping():
    flat  = [{"slope_deg": 2,  "area_m2": 50, "aspect_deg": 0}]
    gable = [{"slope_deg": 30, "area_m2": 40, "aspect_deg": 0},
             {"slope_deg": 30, "area_m2": 40, "aspect_deg": 180}]
    hip   = [{"slope_deg": 30, "area_m2": 20, "aspect_deg": a} for a in (0, 90, 180)]
    assert roof_type_from_planes(flat,  0.9)[0] == "flat"
    assert roof_type_from_planes(gable, 0.9)[0] == "gable"
    assert roof_type_from_planes(hip,   0.9)[0] == "hipped"
    assert roof_type_from_planes(gable, 0.88)[1] == 0.88     # confidence == plane coverage
