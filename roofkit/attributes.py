"""Geometry-derived attributes and small helpers for assembling the output record."""
import geopandas as gpd
import numpy as np

EPSG_M = 31256


def geometric_attrs(roof, ndsm, slope_deg, aspect_deg):
    """Height, slope and orientation measured over the roof mask, in EPSG:31256."""
    h_vals, s_vals, a_vals = ndsm[roof], slope_deg[roof], aspect_deg[roof]
    valid = np.isfinite(h_vals)
    npix = int(roof.sum())
    valid_frac = float(valid.mean()) if npix else 0.0

    height = float(np.nanmedian(h_vals)) if npix else 0.0
    slope  = float(np.nanmedian(s_vals)) if npix else 0.0

    ang = np.deg2rad(a_vals[valid]); ng = len(ang)           # orientation via circular mean (angles wrap at 360)
    sin_sum, cos_sum = np.sin(ang).sum(), np.cos(ang).sum()
    orientation = float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360)
    resultant = float(np.hypot(sin_sum, cos_sum) / ng) if ng else 0.0   # concentration -> orientation confidence
    size_ok = float(np.clip(npix / 50.0, 0, 1))

    return {"height_m": round(height, 1), "slope_deg": round(slope, 1),
            "orientation_deg": round(orientation, 0), "resultant": resultant,
            "valid_frac": valid_frac, "size_ok": size_ok, "npix": npix}


def outline_lonlat(outline):
    """Largest polygon's exterior ring as [[lon, lat], ...] in EPSG:4326."""
    poly = max(outline.geoms, key=lambda g: g.area) if outline.geom_type == "MultiPolygon" else outline
    ring = gpd.GeoSeries([poly], crs=EPSG_M).to_crs(4326).iloc[0].exterior
    return [[round(x, 6), round(y, 6)] for x, y in ring.coords]
