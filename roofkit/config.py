"""Run configuration — one dataclass you can point at any small box in Vienna."""
from dataclasses import dataclass

# A few well-understood study areas (west, south, east, north in lon/lat).
LOCATIONS = {
    "karlsplatz":       (16.3676, 48.1978, 16.3689, 48.1987),   # Gruenderzeit, DEM well matched to ortho
    "sonnwendviertel":  (16.3745, 48.1835, 16.3775, 48.1860),   # new build -> DEM older than ortho (temporal gap)
    "museumsquartier":  (16.3550, 48.2025, 16.3580, 48.2045),   # mixed historic + modern
}


@dataclass
class Config:
    # study area (lon/lat). Defaults to Karlsplatz / Resselpark.
    west:  float = 16.3676
    south: float = 48.1978
    east:  float = 16.3689
    north: float = 48.1987

    n_buildings: int = 10       # how many buildings to sample from the box
    seed:        int = 42       # reproducible selection

    ortho_year:   int   = 2024  # Vienna WMTS orthophoto layer (lb2024 = newest)
    zoom:         int   = 20    # ortho zoom (20 ~= 0.1 m/px)
    roof_min_h:   float = 2.0   # m above ground: below this is yard/street, not roof
    flat_thresh:  float = 10.0  # deg: a facet flatter than this counts as flat
    plane_tol:    float = 0.3   # m: RANSAC inlier band around a fitted roof plane
    min_facet_px: int   = 30    # 0.5 m px -> a real roof facet is >= ~7.5 m^2
    gravel_max_slope: float = 20.0  # deg: above this facet pitch, flat gravel/bitumen is physically implausible

    out_json: str = "outputs/roof_attributes.json"
    fig_path: str = "figures/attributes_overlay.png"

    @classmethod
    def at(cls, location: str, **kwargs) -> "Config":
        """Config for a named LOCATIONS preset, e.g. Config.at('sonnwendviertel', seed=7)."""
        west, south, east, north = LOCATIONS[location]
        return cls(west=west, south=south, east=east, north=north, **kwargs)
