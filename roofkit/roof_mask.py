"""Height-derived roof mask + outline.

FMZK footprints can enclose inner courtyards; keeping only nDSM > roof_min_h samples the roof
and not the yard. The kept region, polygonised, is a roof outline derived from height alone —
a real detection, no segmentation model.
"""
import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape as to_shape
from shapely.ops import unary_union


def roof_from_height(geom, dsm_transform, ndsm, roof_min_h=2.0, min_facet_px=30):
    """Return (roof mask on DSM grid, roof outline poly in EPSG:31256, roof area m^2)."""
    full = geometry_mask([geom], out_shape=ndsm.shape, transform=dsm_transform, invert=True)
    roof = full & np.isfinite(ndsm) & (ndsm > roof_min_h)
    if roof.sum() < min_facet_px:                            # tiny/low building: fall back to footprint
        roof = full & np.isfinite(ndsm)
    parts = [to_shape(g) for g, v in shapes(roof.astype("uint8"), mask=roof, transform=dsm_transform) if v == 1]
    outline = unary_union(parts).simplify(0.3) if parts else geom
    roof_area = float(roof.sum()) * abs(dsm_transform.a) * abs(dsm_transform.e)
    return roof, outline, roof_area


def roof_mask_on_ortho(roof, dsm_transform, img, ortho_transform):
    """Reproject the 31256 roof mask onto the ortho (3857) grid -> boolean mask over img."""
    on_ortho = np.zeros(img.shape[:2], "float32")
    reproject(roof.astype("float32"), on_ortho,
              src_transform=dsm_transform, src_crs=CRS.from_epsg(31256),
              dst_transform=ortho_transform, dst_crs=CRS.from_epsg(3857),
              resampling=Resampling.nearest)
    return on_ortho > 0.5


def crop_to_roof(img, on_ortho):
    """Crop ortho to the roof bbox and black out non-roof pixels -> PIL image (or None if tiny)."""
    ys, xs = np.where(on_ortho)
    if len(ys) < 30:
        return None
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[r0:r1, c0:c1].copy()
    crop[~on_ortho[r0:r1, c0:c1]] = 0
    return Image.fromarray(crop)
