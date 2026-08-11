# Design & Reasoning

## Why these sources, and the trade-offs

A rooftop is a 3-D object photographed from directly above, so I fused three open Stadt-Wien
sources that each cover a different blind spot:

- **Orthophoto `lb2024`** (WMTS, 0.1 m, CC BY 4.0) for *appearance* — material, panels, vegetation.
  I chose the city ortho over Sentinel-2 (10 m ≈ 1.4 px per roof — rejected for detection) and over
  basemap.at (no advantage here). Vienna's is a **true ortho**, so roof pixels sit on the footprint
  and overlays are trustworthy. I pin the year (`lb2024`) rather than the moving `lb` alias so runs
  are reproducible.
- **ALS DSM − DGM = nDSM** (0.5 m) for *height*. A top-down photo cannot tell a flat roof from a
  pitched one; height can. I deliberately took the raw-value GeoTIFFs (not a WMS hillshade) because I
  need metres to do arithmetic. I query the authoritative 5000 sheet index for tile names rather than
  guessing them.
- **FMZK footprints** (WFS vector) as a *trusted prior* for where each building is.

Everything is measured in **EPSG:31256** (metres — area distortion negligible over Vienna) and only
displayed in EPSG:3857; coordinates are exported in EPSG:4326 for portability.

## What the sources made possible — and what they didn't

The height model is the workhorse. I fit **roof planes with RANSAC** (each roof pixel is a 3-D point;
peel off the plane most pixels agree with, repeat). The facets give **roof type** (plane count +
orientation → flat / mono-pitch / gable / hipped / complex) and, as pixels sitting >1 m above their
plane, the **superstructures** (chimneys, dormers, HVAC). Slope, orientation and height fall out of
the same mask. For **appearance** I run a **CLIP ViT-B/32 zero-shot** classifier on each roof crop —
no training data, and the softmax gives a calibrated-ish confidence for **material, solar PV, green
roof, condition**.

Reliably recoverable: outline/area, type, slope, orientation, height, superstructures (down to
~1 m), and — with honest confidence — material and PV. **Not** reliably recoverable from these
sources: fine material distinctions (terracotta is under-called by CLIP), **green roof** (no NIR in
the RGB ortho → advisory only; NDVI needs the CIR ortho or 2023 RGBI LiDAR), and **thermal /
insulation** (ECOSTRESS ≈ 70 m per pixel covers many buildings — not resolvable per roof). I flag
these rather than fake them.

A key limitation I found and document: on **redeveloped land** the DSM can predate the ortho by years,
so a new building is missing from the height model. The honest fix is a co-temporal photogrammetric
surface (same flight as the ortho); for now I detect the catchable case (near-zero height) and flag it.

## Alignment and scaling to thousands

**Geolocation** is by construction: every layer carries a CRS, so I reproject all of them into
EPSG:31256, rasterise each footprint onto the DSM grid, and reproject that mask onto the ortho grid —
the roof is sampled identically in both worlds. The true ortho means no layover correction is needed.

**Scaling 10 → 100k:** the pipeline is already per-building and stateless, so it parallelises directly.
For a city run I would (1) serve the ortho/DEM as **Cloud-Optimised GeoTIFFs** and fetch only each
building's bbox via HTTP range requests, (2) index footprints with a spatial grid (H3/S2) and shard
across workers, (3) cache CLIP roof-crop embeddings, and (4) run the geometry (NumPy/RANSAC) on CPU
while batching CLIP on a GPU. The output is one self-contained JSON record per building — trivially
appendable to a datastore.

## Confidence

Every attribute ships a score in `[0, 1]`, and each is *earned* from the evidence, not hand-set:
- **structure** (`type`) = fraction of the roof explained by fitted planes;
- **appearance** (`material`, `pv`, `green`, `condition`) = the CLIP softmax;
- **geometry** (`area`, `slope`, `orientation`) = valid-pixel coverage and the circular concentration
  (resultant length) of aspect — a symmetric gable genuinely has low orientation confidence, which is
  information, not error.

Downstream, scores are used as **per-attribute thresholds**: trust `type` at ≥0.8, treat `green_roof`
as advisory, and surface low-confidence PV for human review rather than auto-accepting it. This keeps
the component honest about what it knows.
