# Design & reasoning

Task: for 10 real buildings in Vienna, detect each roof from open data and extract as many
attributes as I can honestly support, each with its own confidence score. The output is
`outputs/roof_attributes.json`, one record per building. This page covers the four questions the
task asks; the block-by-block detail, including figures from the real run, is in
[`walkthrough.md`](walkthrough.md), and the full source comparison is in
[`data_sources.md`](data_sources.md).

## Why these sources, and the trade-offs

A roof is a 3-D object photographed from above, so no single source is enough: a photo cannot tell
a flat roof from a pitched one, and a height model has no colour. I fused three Stadt Wien open
layers (all CC BY 4.0), each covering a blind spot of the others
([block 0](walkthrough.md#0-the-fusion-idea)):

- **FMZK building footprints** (WFS vector) as the trusted prior for where each building is. Chosen
  over OSM because it is cadastral-grade, complete, and consistently classed
  ([block 2](walkthrough.md#2-footprints-from-the-wfs)).
- **ALS DSM − DGM = nDSM** (0.5 m raw GeoTIFFs) for height. This is the workhorse: RANSAC plane
  fitting on the height pixels gives roof type, slope, orientation and superstructures. I took raw
  tiles, not the WMS hillshade, because I need metres I can do arithmetic on
  ([blocks 4–5](walkthrough.md#4-the-height-model-dsm-and-dgm-tiles)).
- **Orthophoto `lb2024`** (WMTS, 0.1 m) for appearance — material, solar PV, vegetation, condition —
  via a CLIP zero-shot classifier ([block 6](walkthrough.md#6-the-orthophoto),
  [block 11](walkthrough.md#11-appearance-via-clip-zero-shot)).

Rejected: Sentinel-2 (10 m means ~1.4 px per roof — useless for detection, though its NIR is the
right tool for city-scale green screening), basemap.at (no advantage over the city ortho here),
ECOSTRESS thermal (~70 m per pixel spans many buildings), street-level imagery (occludes flat
roofs; not needed for top-down attributes). The main trade-off I accepted: the DEM and the ortho
have different capture dates, so on redeveloped land the height model can predate the building. I
detect the catchable case (near-zero height under a real footprint) and flag it rather than emit a
confident wrong roof ([block 7](walkthrough.md#7-roof-detection-the-height-mask)).

## What these sources can and cannot recover

Reliably recoverable: roof outline and area (a real height-based detection — nDSM > 2 m inside the
footprint's outer ring, no segmentation model), roof type from RANSAC facets, slope, orientation,
height, superstructures down to ~1 m, and — with the confidence visible — material and solar PV.

Recoverable only after fixing a systematic failure: material. The first single-prompt CLIP
vocabulary called terracotta zero times out of ten on a red-tile Gründerzeit block and put "flat
gravel" on 35° roofs. Prompt ensembles, a slate class (Vienna mansards are commonly slate; without
it "metal" absorbed every dark roof) and a structure prior that removes flat gravel when a facet
pitches ≥ 20° took my measured accuracy from 5/10 to 8/10
([block 11](walkthrough.md#11-appearance-via-clip-zero-shot)).

Not recoverable, and shipped as such: green roof (the RGB ortho has no NIR band, so there is no
real NDVI — advisory only, a proper answer needs the CIR ortho or the 2023 RGBI LiDAR), thermal /
insulation quality (no open source resolves a single roof), and the identity of small
superstructures (0.5 m DSM finds a 2 m² object but nothing open can name it). I flag these rather
than fake them.

## Alignment, and scaling from 10 to thousands

Alignment costs nothing here by choice of sources: everything is measured in EPSG:31256 (metres),
and Vienna's ortho is a true orthophoto, so a roof mask derived from the height grid lands on the
correct roof pixels in the photo — I verify this visually in
[block 6](walkthrough.md#6-the-orthophoto). With a normal aerial photo, tall buildings lean and I
would need an explicit layover correction; picking the true ortho made that problem disappear.

Scaling is by construction, because the unit of work is a bounding box, not a point: tile the city
into boxes, run boxes in parallel, drop the demo sampling and process every footprint. The
expensive inputs are cached and shared per box (DEM tiles, ortho tiles), footprints are filtered
server-side by bbox so only needed features cross the wire, and per building the work is
milliseconds of geometry plus four CLIP calls — batchable on one GPU at thousands of roofs per
hour. The seams to watch at scale are data seams, not compute: DEM sheet currency varies across the
city, so the temporal-gap flag from block 7 becomes a routine data-quality signal, and FMZK lag on
new construction argues for unioning it with OSM. For a formal accuracy number I would stratify a
labelled sample by roof type instead of sampling uniformly
([block 3](walkthrough.md#3-building-selection)).

## How the confidence scores work

Every attribute carries its own score in [0, 1], and each is derived from evidence, not asserted:
plane coverage for type, circular concentration R for orientation, valid-pixel fraction for area
and slope, CLIP softmax of the reported class for the appearance attributes. Two different
currencies hide in that list — a softmax is a per-answer score, coverage is a
quality-of-evidence proxy — so each record carries a `confidence_kind` map naming which is which,
and `clip_raw` stores the full distributions so a consumer can see when a call was close
([block 12](walkthrough.md#12-record-assembly)).

There is deliberately no single per-building score. A roof can have solid geometry and shaky
appearance at the same time; a consumer filtering for reliable roof types needs a different
threshold than one hunting confirmed solar. Per-attribute confidence lets each consumer set its own
bar — e.g. trust `type` at ≥ 0.8, require PV ≥ 0.7 before sending a surveyor, treat `green_roof`
as a lead to verify, never as a fact.

## Use of AI

- **Gemini Flash 3.5** — deep research on candidate data sources and existing roof-attribute
  models before I committed to the design.
- **Claude Sonnet 5** — catching bugs and errors in code as I wrote it; my replacement for Stack
  Overflow.
- **Claude Opus 5** — ideation partner; lookups such as the non-obvious Stadt Wien endpoints; small
  reference implementations (e.g. the RANSAC core); and the walkthrough document, which we built
  block by block with my intervention on every block (about 3 hours of joint work).
- **Claude Fable 5** — final polishing pass, test cases, and repo hygiene.

The pipeline design, the source decisions, and the final code are mine; every AI contribution went
through my review before it landed.
