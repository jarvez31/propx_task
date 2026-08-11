"""Tests for the geometry helpers that don't need a network or a GPU."""
import numpy as np
from shapely.geometry import Polygon

from roofkit.attributes import geometric_attrs, outline_lonlat


def test_outline_lonlat_lands_in_vienna_and_closes():
    # a 10 m square in EPSG:31256, central Vienna
    square = Polygon([(5000, 340000), (5010, 340000), (5010, 340010), (5000, 340010)])
    ring = outline_lonlat(square)
    assert ring[0] == ring[-1]                       # closed ring
    lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
    assert 16 < min(lons) < 17 and 48 < min(lats) < 49


def test_geometric_attrs_flat_roof():
    # a 20x20 roof mask, all pixels 12 m high, zero slope
    roof = np.zeros((30, 30), bool); roof[5:25, 5:25] = True
    ndsm = np.where(roof, 12.0, np.nan).astype("float32")
    slope = np.zeros((30, 30), "float32")
    aspect = np.zeros((30, 30), "float32")
    g = geometric_attrs(roof, ndsm, slope, aspect)
    assert g["height_m"] == 12.0
    assert g["slope_deg"] == 0.0
    assert g["npix"] == 400
