"""Compatibility shim for roof_attributes.ipynb.

The notebook does `import roof_pipeline as rp` and calls the data-loading helpers below.
The canonical implementation now lives in the `roofkit` package; this file just re-exports it so
the exploratory notebook keeps running unchanged. New code should import from `roofkit` directly.
"""
from roofkit.data import fetch_footprints, pick_buildings, dem_sheets_for, get_dem, fetch_ortho
from roofkit.surfaces import compute_surfaces

__all__ = ["fetch_footprints", "pick_buildings", "dem_sheets_for", "get_dem",
           "fetch_ortho", "compute_surfaces"]
