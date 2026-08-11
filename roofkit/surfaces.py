"""Height surfaces: nDSM = DSM - DGM, plus slope and aspect from the height gradient."""
import numpy as np
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

EPSG_M = 31256   # Vienna metre CRS — all height/area/slope maths happen here


def compute_surfaces(dsm_arr, dsm_transform, dgm_arr, dgm_transform):
    """Return (nDSM, slope_deg, aspect_deg), all on the DSM grid (0.5 m, EPSG:31256).

    DGM (bare ground, 1 m) is regridded onto the DSM grid with bilinear resampling because
    height is continuous. nDSM = surface - ground = height above ground.
    """
    ground = np.full(dsm_arr.shape, np.nan, "float32")
    reproject(dgm_arr, ground,
              src_transform=dgm_transform, src_crs=CRS.from_epsg(EPSG_M),
              dst_transform=dsm_transform, dst_crs=CRS.from_epsg(EPSG_M),
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)

    ndsm = dsm_arr - ground
    px = abs(dsm_transform.a)                                # pixel size in metres (0.5)
    gy, gx = np.gradient(dsm_arr, px)
    slope_deg  = np.degrees(np.arctan(np.hypot(gx, gy)))     # 0 = flat, ~30 = pitched
    aspect_deg = (np.degrees(np.arctan2(-gx, gy))) % 360     # compass direction the slope faces
    return ndsm, slope_deg, aspect_deg
