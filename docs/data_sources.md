# Data sources — selection & trade-offs

All layers are City of Vienna Open Government Data (data.wien.gv.at / maps.wien.gv.at),
**CC BY 4.0** — attribute "Datenquelle: Stadt Wien".

| Source | Used for | Resolution | Access | Why chosen / rejected |
| --- | --- | --- | --- | --- |
| **Orthophoto `lb2024`** | material, PV, green, condition, overlay | 0.10 m | WMTS `google3857` | Highest usable detail; **true ortho** (roof sits on footprint). Year pinned for reproducibility. |
| **ALS DOM (DSM)** | surface height | 0.5 m | tiled GeoTIFF | Raw metres for arithmetic (not a hillshade image). |
| **ALS DGM (DTM)** | ground height | 1.0 m | tiled GeoTIFF | nDSM = DSM − DGM = height above ground. |
| **FMZK footprints** | building prior / masking | < 0.2 m | WFS (`FMZKGEBOGD`) | Clean, non-overlapping outlines; anchors each roof. |
| **5000 sheet index** | which DEM tiles to fetch | — | WFS (`MZKBLATT5000OGD`) | Authoritative tile names — never guessed. |
| Sentinel-2 | — | 10 m | Copernicus | **Rejected** for detection: ~1.4 px per roof. Kept idea: NIR/NDVI for city-scale green screening. |
| basemap.at ortho | — | 0.15–0.29 m | WMTS | No advantage over the city ortho here. |
| ECOSTRESS LST | thermal | ~70 m | NASA | **Not per-building**: one pixel spans many roofs. |
| Mapillary / street view | facade validation | var. | API | Occludes flat roofs; not needed for top-down attributes. |

## How endpoints were found (repeatable method)

1. **Open-data catalog** (data.gv.at): search the dataset by local name; its record lists the real
   service URLs, CRS, licence, formats. Copy endpoints, don't guess.
2. **OGC GetCapabilities**: every WMS/WFS/WMTS lists what it serves — that's how the ortho `lb` layer
   and `FMZKGEBOGD` were confirmed.

## CRS discipline

- **EPSG:31256** (MGI Austria GK East, metres) — all area/height/slope/orientation maths.
- **EPSG:3857** (web-mercator) — display / tile overlay only (area-distorted; never measured in).
- **EPSG:4326** (degrees) — portable export of the polygon coordinates.

Continuous rasters (elevation) are resampled **bilinear**; categorical masks use **nearest**.

## Known data caveats

- **Temporal mismatch:** DSM currency is per 2.5 km sheet (≈2022–2025); on redeveloped land it can
  predate the ortho, so new buildings are missing from the height model. Catchable cases (near-zero
  nDSM) are flagged; the proper fix is a co-temporal photogrammetric surface.
- **No NIR** in the standard RGB ortho WMTS → NDVI-based green-roof detection needs the CIR ortho or
  the 2023 RGBI LiDAR point cloud.
- **0.5 m DSM** blurs small superstructures (vents, thin chimneys) even when temporally matched.
