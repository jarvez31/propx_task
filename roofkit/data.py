"""Open-data access for Vienna: FMZK footprints (WFS), ALS DSM/DGM (tiled GeoTIFF), ortho (WMTS).

Design choices worth defending:
  * Footprints and DEM are pulled by querying authoritative indexes (FMZK layer, the 5000 sheet
    index) — we never guess tile names.
  * All measuring happens in EPSG:31256 (metres). EPSG:3857 is display-only.
  * DEM tiles are downloaded resumably (the server drops mid-stream) and cached on disk.
"""
import glob
import os
import subprocess
import zipfile
from dataclasses import dataclass

import contextily as cx
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import from_bounds

EPSG_M      = 31256
WFS_BASE    = "https://data.wien.gv.at/daten/geo"
DEM_URL     = "https://www.wien.gv.at/ma41datenviewer/downloads/geodaten/{m}_tif/{s}_{m}_tif.zip"
CACHE_DIR   = "data/dem_cache"
MIN_AREA_M2 = 60.0     # drop tiny fragments (sheds, canopies)
MARGIN_M    = 40.0     # metres of padding around buildings when cropping the DEM

_TO_METRE = Transformer.from_crs(4326, EPSG_M, always_xy=True)


@dataclass
class Scene:
    """Everything an extractor needs for one study area, built once by the pipeline."""
    chosen_buildings: "gpd.GeoDataFrame"
    dsm_transform: object
    ndsm: np.ndarray
    slope_deg: np.ndarray
    aspect_deg: np.ndarray
    img: np.ndarray
    ortho_transform: object
    extent: tuple


def fetch_footprints(west, south, east, north):
    """WFS GetFeature -> GeoDataFrame of FMZK buildings (F_KLASSE==11) in EPSG:31256."""
    (minx, miny), (maxx, maxy) = _TO_METRE.transform(west, south), _TO_METRE.transform(east, north)
    url = (f"{WFS_BASE}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeName=ogdwien:FMZKGEBOGD&srsName=EPSG:{EPSG_M}&outputFormat=json"
           f"&bbox={minx},{miny},{maxx},{maxy},EPSG:{EPSG_M}")
    gdf = gpd.read_file(url)
    assert gdf.crs.to_epsg() == EPSG_M, f"unexpected CRS {gdf.crs}"
    buildings = gdf[gdf.F_KLASSE == 11].copy()
    buildings["area"] = buildings.geometry.area              # real m^2, because 31256
    return buildings[buildings["area"] > MIN_AREA_M2].reset_index(drop=True)


def pick_buildings(gdf, n, seed):
    return gdf.sample(n=min(n, len(gdf)), random_state=seed).reset_index(drop=True)


def dem_sheets_for(bounds_31256):
    """Query the authoritative 5000 sheet index for sheets covering bounds. Returns e.g. ['35_4','45_2']."""
    minx, miny, maxx, maxy = bounds_31256
    url = (f"{WFS_BASE}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeName=ogdwien:MZKBLATT5000OGD&srsName=EPSG:{EPSG_M}&outputFormat=json"
           f"&bbox={minx},{miny},{maxx},{maxy},EPSG:{EPSG_M}")
    index = gpd.read_file(url)
    return sorted(s.replace("/", "_") for s in index["BNR5000"])


def _sheet_tif(sheet, model):
    """Download (resumable) + unzip one DEM sheet if not cached; return path to its GeoTIFF."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = glob.glob(f"{CACHE_DIR}/{sheet}_{model}/*.tif")
    if cached:
        return cached[0]
    url = DEM_URL.format(m=model, s=sheet)
    zip_path = f"{CACHE_DIR}/{sheet}_{model}.zip"
    subprocess.run(["curl", "-s", "--retry", "10", "--retry-all-errors", "--retry-delay", "2",
                    "-C", "-", "--max-time", "600", "-o", zip_path, url], check=True)
    with open(zip_path, "rb") as f:
        assert f.read().rfind(b"PK\x05\x06") != -1, f"{zip_path} truncated"
    zipfile.ZipFile(zip_path).extractall(f"{CACHE_DIR}/{sheet}_{model}")
    return glob.glob(f"{CACHE_DIR}/{sheet}_{model}/*.tif")[0]


def get_dem(model, sheets, bounds_31256):
    """Mosaic the needed sheets and crop to bounds (+margin). Returns (array, transform)."""
    minx, miny, maxx, maxy = bounds_31256
    box = (minx - MARGIN_M, miny - MARGIN_M, maxx + MARGIN_M, maxy + MARGIN_M)
    srcs = [rasterio.open(_sheet_tif(s, model)) for s in sheets]
    nodata = srcs[0].nodata
    mosaic, transform = merge(srcs, bounds=box)
    for s in srcs:
        s.close()
    arr = mosaic[0].astype("float32")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform


def fetch_ortho(chosen_buildings, ortho_year=2024, zoom=20, pad_m=25):
    """RGB ortho (EPSG:3857) covering all buildings. Returns (img[H,W,3], ortho_transform, extent)."""
    buildings_3857 = chosen_buildings.to_crs(epsg=3857)
    minx, miny, maxx, maxy = buildings_3857.total_bounds
    minx -= pad_m; miny -= pad_m; maxx += pad_m; maxy += pad_m
    to_lonlat = Transformer.from_crs(3857, 4326, always_xy=True)   # contextily wants lon/lat (ll=True)
    west, south = to_lonlat.transform(minx, miny)
    east, north = to_lonlat.transform(maxx, maxy)
    url = f"https://maps.wien.gv.at/wmts/lb{ortho_year}/farbe/google3857/{{z}}/{{y}}/{{x}}.jpeg"
    img, extent = cx.bounds2img(west, south, east, north, zoom=zoom, source=url, ll=True, n_connections=4)
    img = img[:, :, :3]
    xmin, xmax, ymin, ymax = extent
    ortho_transform = from_bounds(xmin, ymin, xmax, ymax, img.shape[1], img.shape[0])
    return img, ortho_transform, extent
