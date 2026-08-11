# roofkit — Rooftop Detection & Attribute Extraction (Vienna, open data)

Detect each building's roof from **open** imagery/geodata and extract as many roof attributes as
can be *reliably justified*, each with a **confidence score**. Built for a small set of Vienna
buildings, designed to scale to many thousands.

The core idea is **multi-source fusion**, because no single source sees everything:

| Source | Gives | Blind to |
| --- | --- | --- |
| **FMZK footprints** (WFS vector) | trusted building outline | roof contents |
| **ALS DSM − DGM = nDSM** (0.5 m) | height → slope, type, orientation, superstructures | appearance |
| **Orthophoto `lb2024`** (0.1 m) | material, solar PV, green, condition | height |

Height gives **structure** (via RANSAC roof-plane fitting); the orthophoto gives **appearance**
(via a CLIP ViT zero-shot classifier); the footprint anchors *where*.

---

## Quickstart

```bash
pip install -r requirements.txt          # or: make setup   (adds dev/notebook tools)
python -m roofkit --location karlsplatz   # or: make run
```

This fetches footprints + DEM + ortho for the area, extracts attributes for 10 buildings, and writes:

- `outputs/roof_attributes.json` — one record per building (schema below)
- `figures/attributes_overlay.png` — detected roofs overlaid on the ortho

Other presets: `--location sonnwendviertel|museumsquartier`; change the mix with `--seed`, count with `--n`.

### The 10 buildings (Karlsplatz, seed 42)

| FMZK id | lat, lon | type | material | roof m² | height m |
| --- | --- | --- | --- | --- | --- |
| 4004308561 | 48.19838, 16.36812 | gable | slate | 812 | 24 |
| 4006223345 | 48.19785, 16.36807 | mono_pitch | slate | 238 | 5 |
| 4002350624 | 48.19789, 16.36853 | hipped_pyramidal | metal | 514 | 25 |
| 4003631581 | 48.19874, 16.36815 | complex | terracotta_tile | 452 | 14 |
| 4002350549 | 48.19816, 16.36900 | gable | slate | 177 | 26 |
| 4002350509 | 48.19823, 16.36866 | complex | slate | 564 | 29 |
| 4002350654 | 48.19788, 16.36784 | hipped | metal | 258 | 28 |
| 4002350643 | 48.19784, 16.36879 | mono_pitch | metal | 510 | 25 |
| 4003631637 | 48.19856, 16.36806 | hipped_pyramidal | slate | 284 | 10 |
| 4003631562 | 48.19879, 16.36853 | hipped | terracotta_tile | 221 | 14 |

**Explore interactively:** open `roof_attributes.ipynb` (`make notebook`). Change the CONFIG box to
any part of Vienna, then `inspect_roof(<FMZK_ID>)` for a zoomable, per-building view (ortho with the
roof outline, nDSM + superstructures, slope, and the RANSAC facets — with a masked/unmasked toggle).

---

## Output schema

```jsonc
{
  "4004308561": {
    "building_id": 4004308561,
    "source_used": "Stadt Wien OGD: ortho lb2024 + ALS DSM/DGM (nDSM) + FMZK; CLIP ViT-B/32; RANSAC planes",
    "roof": {
      "polygon": [[16.3679, 48.1985], ...],   // height-derived outline, EPSG:4326
      "area_m2": 743.2, "footprint_area_m2": 743.0,
      "type": "gable", "material": "slate",
      "orientation_deg": 210.0, "slope_deg": 29.3, "height_m": 18.4,
      "n_planes": 4,
      "planes": [ { "slope_deg": 29.9, "aspect_deg": 78.0, "area_m2": 143.5 }, ... ],
      "solar_pv": false, "green_roof": false, "condition": "good",
      "superstructures": [ { "class": "chimney", "area_m2": 1.5, "height_m": 1.9 }, ... ]
    },
    "confidence": { "area": 0.98, "type": 0.92, "material": 0.65, "orientation": 0.71,
                    "slope": 0.9, "solar_pv": 0.3, "green_roof": 0.02, "condition": 0.88 },
    "notes": "..."
  }
}
```

## What each source lets us extract (honest judgement)

| Attribute | Method | Reliability |
| --- | --- | --- |
| Roof outline / area | nDSM > 2 m, polygonised, clipped to FMZK | **High** — real height-based detection |
| Roof type | RANSAC plane count + orientation | **High** — confidence = plane coverage (0.8–0.96) |
| Slope / orientation / height | median + circular mean over roof | **High** |
| Superstructures | pixels > 1 m above their fitted plane | **Medium** — finds dormers/HVAC/large chimneys; 0.5 m DSM misses small vents |
| Material | CLIP prompt-ensemble, slope-constrained | **Medium** — vocabulary incl. slate (Vienna mansards); structure vetoes flat gravel on steep facets; deep shadow can still read tile as slate |
| Solar PV | CLIP zero-shot | **Medium** — flagged by confidence; borderline cases stay ~0.5 |
| Condition | CLIP zero-shot | **Advisory** |
| Green roof | CLIP zero-shot (RGB only) | **Advisory** — needs NIR/NDVI for a real answer |
| Thermal / insulation | — | **Not recoverable** per-building (ECOSTRESS ≈ 70 m ≫ a roof) |

See [`docs/writeup.md`](docs/writeup.md) for the full design & reasoning, and
[`docs/data_sources.md`](docs/data_sources.md) for source selection and trade-offs.

## Confidence scores

Each attribute carries its own score in `[0, 1]`, and each is *earned*, not asserted:
structure confidence is the **fraction of the roof explained by fitted planes**; appearance
confidence is the **CLIP softmax**; geometry confidence reflects **valid-pixel coverage** and the
**circular concentration** of aspect. Downstream users can threshold per attribute (e.g. trust
`type` at ≥0.8, treat `green_roof` as advisory).

## Repository layout

```
roofkit/            the package (production pipeline)
  config.py         run configuration + named study areas
  data.py           open-data access (WFS footprints, DEM tiles, WMTS ortho) + Scene
  surfaces.py       nDSM / slope / aspect
  roof_mask.py      height-derived roof mask + outline
  planes.py         RANSAC roof-plane segmentation -> type + superstructures
  appearance.py     CLIP ViT zero-shot -> material / PV / green / condition
  attributes.py     geometry attributes + record helpers
  pipeline.py       orchestration -> outputs/roof_attributes.json
  viz.py            overview overlay figure
  __main__.py       CLI
roof_attributes.ipynb   interactive exploration notebook (mirrors the package)
roof_pipeline.py        thin compatibility shim for the notebook
tests/                  pure-function unit tests (run in CI)
docs/                   design write-up + data-source justification
outputs/ figures/       sample results (committed); data/ is fetched & cached (gitignored)
```

## Development

```bash
make setup     # install runtime + dev deps
make test      # pytest
make lint      # ruff
```

CI (GitHub Actions) runs the unit tests on every push. Data is © Stadt Wien
(data.wien.gv.at), CC BY 4.0; code is MIT (see `LICENSE`).

**Tools used:** Python (geopandas, rasterio, shapely, contextily, open_clip / PyTorch CPU),
CLIP ViT-B/32 (OpenAI weights) for zero-shot appearance. AI assistants (Claude) were used for
design review, debugging and documentation editing; the pipeline design, data-source choices and
final code are my own.

## What I'd do with more time

- Replace the flat/pitched slope threshold entirely with per-facet types from the plane set.
- Add NIR (Vienna CIR ortho or the 2023 RGBI LiDAR) for a real NDVI green-roof signal.
- Calibrate the material classifier against a small labelled set (prompt ensembles + a slate class +
  a slope prior fixed the systematic terracotta under-call, but shadowed tile can still read as slate).
- Handle the **DSM/ortho temporal gap** on redeveloped land (e.g. Sonnwendviertel) with a
  co-temporal photogrammetric surface or a build-year source — currently flagged, not corrected.
