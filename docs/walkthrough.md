# Pipeline walkthrough, block by block

This is the long version. The one-page design summary the task asks for is in
[`writeup.md`](writeup.md); this document goes through the pipeline one stage at a time — what each
block does, why I built it that way, and where it breaks. All figures come from the real Karlsplatz
run that produced `outputs/roof_attributes.json`.

```
0  The fusion idea               5  Surfaces: nDSM, slope, aspect     10  Geometric attributes
1  Input -> Config -> bbox       6  Orthophoto fetch                  11  CLIP appearance
2  Footprints (WFS / FMZK)       7  Roof detection (height mask)      12  Record assembly
3  Building selection            8  RANSAC planes -> roof type        13  Outputs & reproducibility
4  DEM tiles (DSM + DGM)         9  Superstructures
```

---

## 0. The fusion idea

No single open source sees a roof completely, and every design decision in this project follows
from that.

| Source | Sees | Blind to |
| --- | --- | --- |
| Footprint (FMZK) | where the building is | anything about the roof itself |
| Height model (nDSM) | the roof's 3-D shape | colour, material, texture |
| Orthophoto | colour and texture | height — flat and pitched look identical from above |

So the pipeline fuses three sources: the footprint anchors where, the height model gives structure,
the photo gives appearance, and each attribute is extracted from the source that can actually see
it. The three sources converge on the roof mask (footprint outer ring ∧ nDSM > 2 m), which fixes
the exact pixel set everything downstream reads. From there the mask + height drive RANSAC into the
structure attributes (type, slope, orientation, area, superstructures), and the mask + orthophoto
drive CLIP into the appearance attributes (material, PV, green, condition). The fusion also runs the
other way once: the structure result constrains the appearance vocabulary (block 11).

The honest failure cases are the seams between sources, most sharply when the height model and the
photo were captured years apart (block 7).

![The three-source fusion](../figures/writeup/00_fusion.png)

---

## 1. Input, config, bounding box

The entry point is `python -m roofkit --location karlsplatz`. The unit of input is a rectangle, not
a point. `Config.at("karlsplatz")` looks up the SW and NE corners of a box in lon/lat:

```python
"karlsplatz": (16.3676, 48.1978, 16.3689, 48.1987)   # west, south, east, north
```

Every knob that shapes a run lives in the same `Config` dataclass: `n_buildings`, the random `seed`,
the ortho year and zoom, and the physical thresholds the later blocks use — `roof_min_h = 2.0 m`,
`flat_thresh = 10°`, `plane_tol = 0.3 m`, `min_facet_px = 30`, `gravel_max_slope = 20°`. Nothing
that affects a result is buried deeper in the code.

Why a box and not a point: a point forces the code to guess which building you meant and how far to
look. A box is explicit and reproducible, and it is the natural unit for the real goal — you scale
to a city by tiling it into boxes, not by clicking points. The three presets stress different
conditions: `karlsplatz` (Gründerzeit blocks, DEM and ortho agree), `sonnwendviertel` (new
construction, DEM older than the photo — the temporal-gap case), `museumsquartier` (mixed).

On coordinate systems, one rule I hold throughout: EPSG:4326 to talk to the outside world,
EPSG:31256 (metres) to measure, EPSG:3857 to display. A degree of longitude at 48° N is not a fixed
number of metres, so measuring area or setting a 0.3 m tolerance in degrees would be meaningless.

What breaks: a box that is too large pulls many DEM tiles; too small and there is nothing to sample.
More subtly, a wrong constant in `Config` degrades every building at once with no error raised —
set `roof_min_h` too high and low buildings just vanish. That is exactly why every threshold is
named there and nowhere else.

![The study area is a box, not a point](../figures/writeup/01_bbox.png)

---

## 2. Footprints from the WFS

`fetch_footprints(...)` turns the box into trusted building outlines. The FMZK
(Flächen-Mehrzweckkarte, Vienna's official large-scale base map) publishes its building layer
`FMZKGEBOGD` as open data over WFS. The function converts the box corners to metres before building
the request, asks the server for `srsName=EPSG:31256` with a bbox filter, asserts the CRS actually
came back as 31256, filters to `F_KLASSE == 11` (above-ground buildings) and drops polygons under
60 m².

On the Karlsplatz box: the WFS returns 37 polygons, the class filter drops 12 non-building features
(canopies, sub-surface, boundaries), the area filter drops 11 slivers, 14 real buildings survive.

![FMZK footprints, kept vs dropped](../figures/writeup/02_footprints.png)

I chose FMZK over OpenStreetMap deliberately. FMZK is cadastral-grade, complete across the city,
carries a consistent class schema, and has a clear CC BY licence. OSM is crowd-sourced and patchy,
and has no building-class field. For a component meant to scale to thousands of buildings, a
uniform authoritative prior beats a patchy one. I also filter server-side by bbox instead of
downloading the whole city layer — only the features I need cross the wire.

The CRS assert matters more than it looks. WFS servers are allowed to ignore `srsName` and return
their native CRS. If this one silently handed back 4326, every area, the 60 m² cutoff and the 0.3 m
plane tolerance would be computed in degrees, wrong, with no error anywhere. One line converts that
silent corruption into a loud failure.

What breaks: the two filters are blunt. The 60 m² cutoff drops a genuinely tiny building and keeps a
large canopy mis-tagged as class 11. And the block trusts FMZK's classification — a building the
city mis-coded never enters the pipeline. A slightly wrong outline is recoverable (block 7 re-detects
the roof from height inside it); a missing building is not. For a full-city run I would union FMZK
with OSM so each covers the other's gaps.

---

## 3. Building selection

`pick_buildings(gdf, n, seed)` is a seeded random sample: 10 of the 14 survivors on this box.

![The seeded selection](../figures/writeup/03_selection.png)

The task asks for 8–10 buildings, so sampling is a demo affordance — at city scale you process every
footprint. But I sample uniformly at random instead of hand-picking ten photogenic roofs, on
purpose: cherry-picking would inflate the apparent quality and hide the cases the method struggles
with. Whatever mix of flat, pitched, clean and messy roofs the area contains is what gets scored.

`seed=42` fixes which ten are drawn, so a reviewer who clones and runs gets the same selection, the
same JSON, the same figures. Changing the seed is also a cheap robustness probe: `--seed 7` re-rolls
the sample onto different roofs.

One honest subtlety: `sample(random_state=seed)` picks rows by position, so byte-identical results
depend on the WFS returning features in a stable order. It holds in practice, but the correct claim
is "reproducible given a stable upstream feature order", not an unconditional guarantee. And a
uniform sample can miss the one interesting building in the box — for a formal evaluation I would
stratify by type instead.

---

## 4. The height model: DSM and DGM tiles

Two raster surfaces:

- DSM (`dom`, 0.5 m) — the surface as the laser scanner saw it: roofs, trees, everything.
- DGM (`dgm`, 1 m) — the bare ground with buildings and vegetation stripped out.

Their difference is height above ground; the subtraction is block 5. Getting the pixels is three
functions. `dem_sheets_for(bounds)` queries the official 1:5000 sheet index (`MZKBLATT5000OGD`,
another WFS layer) for which tiles cover the bounds — on this box it returns two (`35_4`, `45_2`)
because the buildings straddle a sheet boundary, which is the whole argument for querying instead of
hard-coding tile names. `_sheet_tif` downloads and caches one tile. `get_dem` mosaics the tiles,
crops to the buildings plus a 40 m margin, and converts nodata to NaN.

![DSM vs DGM over the ten buildings](../figures/writeup/04_dem_tiles.png)

I took the raw-value GeoTIFFs, not the WMS hillshade Vienna also serves. I need to do arithmetic in
metres — DSM − DGM, "is this pixel above 2 m", "fit a plane within 0.3 m" — and a hillshade is a
picture of the terrain, not a measurement.

The download itself needed work: Vienna's DEM server drops connections mid-stream, so `curl` runs
with retry + resume and a time ceiling, and before trusting the file I check it contains the ZIP
end-of-central-directory marker. A truncated download fails here with a clear message instead of
surfacing later as a baffling "not a zip file" somewhere in rasterio.

What breaks: (1) DSM and DGM are on different grids (0.5 m vs 1 m — 418×370 vs 209×185 on this box),
so they cannot be subtracted directly; block 5 regrids first. (2) If nodata were not converted to
NaN, holes would read as real elevations and poison every height statistic. (3) The DEM has a
capture date; where it predates the ortho, the height under a new building is simply wrong — block 7
flags that case.

---

## 5. Surfaces: nDSM, slope, aspect

`compute_surfaces` regrids the 1 m DGM onto the 0.5 m DSM grid with bilinear resampling, then

```
ndsm = dsm - ground          # height above terrain
gy, gx = np.gradient(dsm, 0.5)
slope_deg  = degrees(arctan(hypot(gx, gy)))
aspect_deg = degrees(arctan2(-gx, gy)) % 360
```

![nDSM, slope and aspect](../figures/writeup/05_surfaces.png)

In the figure: buildings stand at their true height in the nDSM (the tallest pixel in view is a
59 m spire), flat roof interiors are dark in the slope map while building edges light up, and each
pitched facet takes a compass colour in the aspect map — a gable shows as two opposite colours.

One resampling rule, stated once: interpolate continuous fields, never interpolate categories.
Height is continuous, so the DGM gets bilinear. A roof mask (block 7) is yes/no, so it gets
nearest-neighbour — a bilinear'd mask would invent fractional "0.5 roof" pixels along every edge.
Getting this backwards is a classic geospatial bug.

The per-pixel slope and aspect are raw material for the summary statistics in block 10. They are
deliberately not how roof type is decided — per-pixel slope is noisy, and type needs the grouped,
robust view RANSAC gives.

What breaks: at building walls the DSM steps vertically and the gradient explodes — the bright
outlines around every footprint are 60°+ edge artefacts, not roof pitch. I handle that by measuring
over the roof interior with medians and by using RANSAC for type. And aspect on a flat roof is pure
noise (arctan2 of two near-zero numbers), which is why flat roofs report `orientation_deg = null`
rather than a confident random bearing.

---

## 6. The orthophoto

`fetch_ortho` reprojects the chosen buildings to EPSG:3857, pads their bounds by 25 m, and pulls
Vienna's WMTS tiles:

```
https://maps.wien.gv.at/wmts/lb2024/farbe/google3857/{z}/{y}/{x}.jpeg
```

`contextily.bounds2img(..., zoom=20)` stitches them into one 1792×1792 image at ~0.1 m/px, and I
build an affine transform so every pixel maps to a metre coordinate. Footprints came over WFS
because I needed geometry; the photo comes over WMTS because I need pixels.

![Orthophoto and the alignment check](../figures/writeup/06_ortho.png)

Vienna publishes a true orthophoto — corrected with the surface model so every pixel is placed as if
seen straight down. That matters: on a normal aerial photo a tall building leans, and a footprint
overlay lands on the facade instead of the roof. The right panel is the check: FMZK footprints land
on the roofs. This is what lets block 7 take a mask derived from height and sample the correct roof
pixels in the photo, with no layover correction.

I pin the year (`lb2024`) instead of the `lb` "latest" alias, which silently advances every year and
would make results drift between runs. Zoom 20 (~0.1 m/px) resolves roof texture and solar panels;
higher zooms multiply download time with no information gain.

What breaks: the ortho is in 3857, a display CRS whose scale is distorted at Vienna's latitude, so I
never measure on it — area comes from the 31256 nDSM grid, the ortho is only sampled for colour.
Shadows and tree overhang are baked into a single flight date. And the ortho's capture date can
differ from the DEM's — the temporal seam again.

---

## 7. Roof detection: the height mask

This is how the roof is actually detected. No segmentation model, no training — the roof extent is
decided by the height data.

`roof_from_height` rasterises the footprint's outer ring (interior holes filled) onto the DSM grid,
then keeps the pixels that are inside it and more than 2 m above ground:

```python
roof = full & isfinite(ndsm) & (ndsm > 2.0)
```

If that leaves fewer than 30 pixels (a genuinely low building, or a DEM hole) it falls back to the
bare footprint so the building still produces a record. The kept pixels are polygonised and
simplified at 0.6 m (above the 0.5 m pixel size, so the raster staircase collapses to clean edges);
area is the pixel count × 0.25 m². Two helpers carry the mask across to the photo: reproject to the
ortho grid with nearest-neighbour (mask = category, block 5 rule), then crop and black out
everything off-roof so CLIP sees the roof and nothing else.

![Height-derived roof detection](../figures/writeup/07_roofmask.png)

The figure shows a typical building: the footprint interior coloured by height, the > 2 m rule
removing the light wells and edge overhang, and the resulting outline on the photo. The headline
number is deliberately unspectacular — 245 m² footprint, 238 m² roof, 97% kept — because on a
well-mapped Gründerzeit block FMZK is already good, and the mask's visible work is trimming the
residual light wells.

Why fill the footprint's holes and let height decide? Because I hit a real false negative. FMZK maps
a ~36 m² light well at the centre of one roof, but a solar array is physically mounted over that
opening. The original mask trusted the hole, so the panel pixels were removed despite sitting at
11 m, and `solar_pv` could never fire. Rasterising the outer ring and letting the > 2 m test
arbitrate fixes it: a genuinely open courtyard still drops out (nDSM ≈ 0), a covered opening stays.
After the fix the panels reach CLIP and PV fires (blocks 8 and 11). This is also why roof area can
exceed footprint area in the output — the record notes say so.

Why the mask matters even when it only removes 3%: it is a real detector, not a copy of the
footprint (where the footprint disagrees with reality, height wins); it defines the exact pixel set
that RANSAC, the superstructure search, the geometry medians and the CLIP crop all consume; and it
needs no model while handing RANSAC its 3-D point cloud for free. I considered SAM-style learned
segmentation and rejected it: the task is attribute extraction, not instance masks, and a learned
mask is flat 2-D — it would cost a model dependency and give back less.

What breaks: (1) the temporal gap — on redeveloped land the nDSM under a real new building is ~0,
the fallback keeps the bare footprint, and the near-zero height is the flag that the DEM predates
the building; I surface it instead of emitting a confident wrong roof. (2) A tree crossing the
footprint edge can be caught as roof; bounded, because only inside-footprint pixels are eligible.
(3) The 2 m threshold is a knob — above head height is a defensible ground-vs-roof cut, and it lives
in the config.

---

## 8. RANSAC plane fitting and roof type

Block 7 produced roof pixels that each carry a height. Treat every pixel as a 3-D point (x, y, z)
and ask: how many planes does this roof break into, and which way does each face?

Three layers. `fit_plane` is a least-squares plane through a point set. `ransac_plane` repeats 200
times: pick 3 random points, fit the plane they define, count how many of all points fall within
0.3 m of it; keep the plane with the most agreement. That is what makes it robust — a chimney
doesn't fit the roof plane, so it simply isn't counted, instead of dragging the fit off the way it
would in one global least-squares fit. `fit_roof_planes` runs this sequentially: bank the dominant
plane, remove its pixels, repeat up to 6 times. Each facet records its slope, aspect and area (all
exported in the JSON as `roof.planes`), and `coverage` is the fraction of roof pixels assigned to
some plane.

`roof_type_from_planes` reads the type off the facet set: drop slivers under 5 m²; if the steepest
facet is under 10°, flat; otherwise count pitched facets — 1 mono-pitch, 2 gable, 3 hipped,
4 hipped/pyramidal, more complex. The type confidence is the coverage, not a hand-set number.

![RANSAC facets on the solar building](../figures/writeup/08_ransac.png)

The figure is the solar building: RANSAC peels it into six facets — four hip faces (aspects 12°,
198°, 281°, 109°), a small extra facet, and a near-flat 36 m² patch at 7°, which is the covered
light-well platform the panels sit on, kept by the block 7 fix. The grey ~10% is unexplained: ridge
lines, the base of the solar mounting, edge noise. Six facets exceeds the four-facet bucket, so a
conceptually hipped roof lands in `complex` at coverage 0.90. I left that in deliberately — it shows
the taxonomy's over-segmentation edge and, at the same time, the coverage staying honestly high
because most of the roof really is planar.

The coverage number is a real measurement: a crisp gable comes out ~0.95, a roof cluttered with
superstructures scores lower because it genuinely is harder to describe as planes, and that lower
number is information the consumer should have.

What breaks: (1) five type buckets is coarse and `complex` is the catch-all. (2) The 10° flat/pitch
boundary is a hard edge — a roof hovering near it can flip on DSM noise; per-facet typing would fix
this and is on the more-time list. (3) Greedy peeling can split one noisy plane or merge two
near-coplanar ones; the 5 m² sliver filter and 30-pixel minimum keep it in check. (4) `max_planes=6`
means a truly baroque roof is summarised, not fully parsed — acceptable, the goal is a type, not a
CAD model.

---

## 9. Superstructures

Once the planes are fitted, anything sitting above them is by definition an object on the roof.
`plane_base_image` evaluates, for every roof pixel, the height of its nearest fitted plane; the
residual is actual nDSM minus that. On a bare facet the residual is ≈ 0. Threshold at > 1 m, label
the connected blobs, and classify each by geometry: under 3 m² and over 1.2 m tall is a chimney,
6 m² and up is a large superstructure, the rest dormer_or_small. Two guards: blobs under 1 m² are
noise, and anything more than 8 m above its plane is not a rooftop object but an adjacent taller
wing of the building poking into the mask.

![Height-above-plane residual with detections](../figures/writeup/09_superstructures.png)

The right panel is the residual, and it is almost entirely dark — visual proof that block 8's planes
describe this roof well. Only the genuine protrusions light up: four chimneys, three clustered along
the ridge, one on the lower wing.

Measuring against the plane rather than the ground is the point. A chimney on a six-storey building
sits at ~20 m absolute — the same height as the roof next door. What makes it a chimney is being
1.5 m above its own roof. The residual sees that, so the detector works identically on a bungalow
and a tower. I did not try to find these in the photo instead: the ortho cannot tell a dark HVAC
unit from a stain or shadow, and it has no sense of "sticks up".

What breaks: (1) the 8 m guard discards a genuinely tall stair tower or plant room — I chose
under-reporting over fabricating a giant chimney, but it is a real miss on industrial roofs.
(2) Two chimneys 30 cm apart merge into one blob, as nearly happens on the ridge here. (3) The class
is geometry-only — a 1.5 m, 2 m² bump is called a chimney whether it is a chimney, a vent stack or a
skylight housing. Naming each object from its ortho crop with CLIP is the natural upgrade, and the
reason it is deferred is resolution: a 1–3 m² object is ~15 px in this ortho, far below what
CLIP ViT-B/32 (224×224 input) can use. At this scale the height residual is the stronger signal.

---

## 10. Geometric attributes

Height is the median nDSM over the mask; slope is the median per-pixel slope. Median, not mean, on
purpose: the mask always catches a few pathological pixels — a chimney spiking the height, a wall
edge reading 80° — and a median shrugs them off.

Orientation needs more care, because compass bearings wrap: 350° and 10° are 20° apart, but their
arithmetic mean is 180°, pointing backwards. So each pixel's aspect becomes a unit vector, the
vectors are summed, and the direction of the sum is the mean orientation. The length of that sum,
normalised by the pixel count, is R ∈ [0, 1]: if every pixel faces the same way the vectors stack up
and R → 1; if they face all around the compass they cancel and R → 0. R is the orientation
confidence — measured, not invented.

![Aspect as a compass rose, R high vs R low](../figures/writeup/10_orientation.png)

Both roofs in the figure are hipped, and that is why they are shown together. The top one has a
dominant south-southwest facing: the aspect mass bunches, the arrow is long, R = 0.68, and
"orientation 184°" is worth something. The bottom one has two opposite lobes that cancel: R = 0.10,
and the reported 271° is near-meaningless — correctly so, because a symmetric roof has no single
facing. The type classifier cannot tell these two apart; R can. That is why orientation ships with
its own confidence instead of being folded into the type.

Two more terms so nothing is reported bare: `valid_frac`, the fraction of mask pixels with finite
height (the area confidence — drops when the DEM has holes), and `size_ok = clip(npix/50, 0, 1)`, a
penalty on tiny roofs where a handful of pixels cannot support a stable median. Slope confidence is
their product. And for a flat roof the record sets `orientation_deg = null` with confidence 0.0 —
reporting "no orientation" is more honest than reporting noise.

What breaks: per-pixel aspect from a DSM gradient scatters ±30° even on a clean facet, so R rarely
approaches 1 on real roofs — which is fine, as long as the consumer actually reads it. And median
height is the roof surface's typical height, not eaves or ridge height specifically.

---

## 11. Appearance via CLIP zero-shot

Material, panels, vegetation and condition are appearance: you have to look at the roof. This is the
only learned component, and it is zero-shot — no training data, no labels, no roof model to
maintain. CLIP scores the masked roof crop against groups of text prompts; the softmax inside a
group is the confidence; the full distribution is stored in `clip_raw`.

The first version used one sentence per class and a five-class vocabulary (terracotta, metal,
gravel, glass, green). It failed in a systematic way: on a Gründerzeit block full of red tile roofs
it called terracotta zero times out of ten, and it labelled two steep roofs "flat gravel" — a
material that physically cannot sit on a 35° pitch. I checked the actual crops against the
classifier output building by building and fixed it with three changes:

1. Prompt ensembles. Each class is now several phrasings, and the class embedding is the
   renormalised mean of their text embeddings. The standard CLIP trick; it removes the dependence on
   one sentence's wording.
2. A slate class. Vienna mansards are commonly dark slate or fibre-cement. Without that class in the
   vocabulary, "metal" was absorbing every dark pitched roof, which is exactly what a closed-world
   softmax does with a missing option.
3. A structure prior. If any facet from block 8 pitches ≥ 20°, `flat_gravel` is removed from the
   prompt list before scoring — the one place the fusion runs backwards, structure constraining
   appearance. The record notes say when this happened.

Scored against the crops (my visual read of the ten roofs as ground truth): 8/10 correct, up from
5/10. Both true tile roofs now come out `terracotta_tile`. The two remaining misses are honest
ambiguities — one heavily shadowed tile roof reads as slate (tile is the visible runner-up in
`clip_raw`), and one dark flat-ish roof reads as slate instead of bitumen.

![CLIP scores on the solar building](../figures/writeup/11_clip.png)

What the confidence number is, and is not. The softmax is a relative ranking over the prompts I
offered, nothing more:

- Closed-world: it answers "which of these sentences fits best", never "what is this". If the true
  material is not in the vocabulary, the scores split confidently among wrong options — the missing
  slate class demonstrated exactly that.
- Not calibrated: 0.59 does not mean 59% of such calls are correct. There is no validation set
  behind it. Treat appearance confidences as ordinal.
- Sharpened: CLIP's raw cosine similarities sit in a narrow band and are multiplied by the logit
  scale (100×) before softmax, so printed margins look bigger than they are. This is why the record
  stores the full `clip_raw` distribution, not just the winner — a 0.55 with a 0.35 runner-up and a
  0.55 with a 0.05 runner-up are very different calls.
- Prompt-sensitive and out-of-distribution: a top-down aerial crop with a black mask border is
  unlike CLIP's training data, and the score depends on the exact wording. Ensembling softens the
  wording dependence; it does not remove the OOD problem.
- The PV/green flags are a hard 0.5 cut on a soft number: a roof at 0.55 and one at 0.49 are nearly
  the same evidence with opposite booleans. The stored confidence is what tells you a call sat on
  the fence — the solar building's PV fired at 0.55.

Per-attribute trust, because these are not equal: solar PV is the most reliable call here (panels
have a distinctive signature); material is usable with the margin visible; green roof is advisory
only, structurally — the ortho is RGB with no NIR band, so there is no real NDVI and CLIP guesses
vegetation from colour; condition ("good vs weathered") is subjective and confounded by sun angle,
and I weight it lowest.

Torch is imported lazily, so the geometry paths and the unit tests never load a deep-learning stack.

---

## 12. Record assembly

`extract_building` runs blocks 7–11 for one footprint and fuses the results into one
self-describing JSON record: the file should need no external document to interpret. Two parallel,
same-keyed dicts — `roof` with the answers, `confidence` with one number per answer — plus
`clip_raw` (full distributions), `roof.planes` (the facet list), `source_used` (data lineage on
every record, not just the run), and `notes` (the standing caveats, including the no-NIR green-roof
warning and the outer-ring area behaviour).

One thing I refuse to blur: "confidence" holds two different currencies, and the record says which
is which. The appearance confidences are the CLIP softmax of the reported class — a per-answer
probability-like score. The geometry confidences are evidence-support proxies: plane coverage for
type, circular R for orientation, valid-pixel fraction for area and slope. A 0.90 coverage means
"this roof is well described by planes", not "90% chance the label is exactly right". A
`confidence_kind` map ships in the record (`p_reported_class` vs `evidence_support`) so a program
reading the raw JSON knows too.

Getting the appearance semantics consistent flushed out a real bug: originally `solar_pv` and
`green_roof` stored P(yes) while `condition` stored the probability of the reported class, which
produced `green_roof: false` with confidence 0.00 — reads as "no confidence", actually meant "very
sure it is not green". All appearance confidences now answer the same question, "how sure am I of
the value I wrote", and P(yes) is still recoverable from `clip_raw`.

There is deliberately no single overall confidence per building. A roof can have rock-solid geometry
and shaky appearance; collapsing eight numbers into one would destroy exactly the information a
downstream consumer needs to threshold per attribute.

The record for the solar building, abridged:

```jsonc
{
  "building_id": 4003631581,
  "roof": { "type": "complex", "material": "terracotta_tile", "slope_deg": 42.7,
            "height_m": 14.2, "area_m2": 452.2, "solar_pv": true,
            "planes": [ { "slope_deg": 40.4, "aspect_deg": 12.0, "area_m2": 84.0 }, ... ],
            "superstructures": [ { "class": "chimney", "area_m2": 2.5, "height_m": 1.5 }, ... ] },
  "confidence": { "type": 0.9, "material": 0.55, "solar_pv": 0.55, "orientation": 0.09, ... },
  "confidence_kind": { "type": "evidence_support", "material": "p_reported_class", ... },
  "clip_raw": { "material": { "terracotta_tile": 0.55, "metal": 0.23, "slate": 0.14, ... }, ... }
}
```

---

## 13. Outputs and reproducibility

`run(config)` builds the scene once (footprints, sample, DEM, surfaces, ortho), loops every chosen
building through `extract_building`, and writes `outputs/roof_attributes.json`, the overview figure,
and a static 3-panel detail per building for the first few (the interactive versions live in the
notebook).

![All ten buildings with attributes](../figures/attributes_overlay.png)

Red dotted is the FMZK footprint (the prior), cyan is the height-derived roof outline (the
detection), and the label is each roof's type/material with PV/green flags and the superstructure
count. The solar building carries its `+PV` only because of the block 7 mask fix.

What makes a stranger able to reproduce the numbers:

- One `Config` dataclass holds the box, the seed and every threshold.
- The selection is seeded, and the ortho year is pinned (`lb2024`, not the moving `lb` alias).
- DEM tiles download resumably and are integrity-checked before use.
- `requirements.txt` is pinned, torch resolves to the CPU wheel, and an 8-test suite covers the
  geometry paths without needing the DL stack — CI runs it on every push.
- The notebook mirrors the package (prompts and scoring are imported from `roofkit.appearance`, one
  source of truth), and I verified the notebook run and the CLI run produce byte-identical JSON.
- All text files are UTF-8. The repo's earliest files were UTF-16, which silently disabled
  `.gitignore` and mis-rendered the README; I found it with a hex dump and converted everything.

What breaks: the first run needs live network access to the Stadt Wien endpoints (cached afterwards).
And the JSON and figures are build products — after any change to the mask, prompts or confidence
logic they must be regenerated, as they were after every change described above.

---

# Decisions, dead-ends, and what I ruled out

## Deliberate choices

| Decision | Why | Rejected alternative | Block |
| --- | --- | --- | --- |
| Study area = bounding box | explicit, reproducible, the unit of city-scale tiling | click a point, guess the building | 1 |
| Three-source fusion | no single source sees where + structure + appearance | one source, accept the blind spots | 0 |
| 31256 to measure / 3857 to display / 4326 to export | metres for real geometry | measure in the ortho's 3857 | 1, 5 |
| Raw DEM GeoTIFFs, not WMS hillshade | need metres for arithmetic | pretty relief tiles | 4 |
| Query the sheet index for tile names | box straddled 2 sheets and it just worked | hard-code sheet IDs | 4 |
| Resumable curl + zip check | Vienna's server drops mid-stream | plain one-shot download | 4 |
| Pin ortho year `lb2024` | same run, same photo, forever | `lb` latest alias | 6 |
| True orthophoto | roof pixels sit on the footprint | manual layover correction | 6 |
| `F_KLASSE == 11` + 60 m² filter | keep real buildings (37 -> 14 here) | every FMZK polygon; or OSM | 2 |
| Bilinear for elevation, nearest for masks | never interpolate a category | one resampling everywhere | 5 |
| Height mask as the detector | explainable, training-free, gives 3-D free | learned segmentation (SAM) | 7 |
| Sequential RANSAC | robust to clutter; coverage = type confidence | one least-squares plane | 8 |
| CLIP zero-shot, ensembled prompts | no labelled roof data exists; softmax = usable confidence | colour rules; train a classifier | 11 |
| Slope prior on the material vocabulary | structure vetoes physically impossible materials | let CLIP rank everything always | 8, 11 |
| Circular statistics for orientation | bearings wrap at 360° | arithmetic mean of degrees | 10 |
| Per-attribute earned confidence | coverage / softmax / valid-frac are real evidence | one hand-set score per building | 12 |
| Lazy torch import | tests and geometry never touch the DL stack | import at module load | 11 |

## Dead-ends (what I tried, what broke, what I did instead)

- HSV colour heuristics for material and PV. First attempt. PV fired on 8 of 10 roofs (false
  positives on dark bitumen and shadow); material collapsed to a meaningless "concrete grey".
  Replaced with CLIP zero-shot, which cut PV to a sensible 2/10. (Block 11)
- Single-prompt CLIP vocabulary. Zero terracotta calls in ten buildings on a red-tile block, and
  "flat gravel" on 35° roofs. Fixed with prompt ensembles, a slate class, and the ≥ 20° slope prior;
  measured 5/10 -> 8/10 against the crops. (Block 11)
- FMZK courtyard contamination. The footprint encloses courtyards, so early reads averaged roof and
  empty yard together. Fixed by the nDSM > 2 m mask. (Block 7)
- Trusting FMZK's interior holes. A solar array mounted over a mapped light well was masked out
  despite sitting at 11 m, guaranteeing `solar_pv = false`. Fixed by rasterising the outer ring and
  letting height arbitrate openings; the panels now reach CLIP and PV fires. (Block 7)
- SAM / learned segmentation — considered, rejected: the task is attribute extraction, a height
  threshold is explainable and training-free, and it gives structure for free. (Block 7)
- DSM/ortho temporal gap on redeveloped land (Sonnwendviertel): the height model predates the photo,
  so a new building is missing from the nDSM. Detected via near-zero height under a real footprint
  and flagged, not corrected. (Block 7)
- QuickGELU weight mismatch. CLIP loaded with the wrong activation variant and silently degraded
  scores behind a warning. Fixed by matching the model name to the OpenAI weights
  (`ViT-B-32-quickgelu`). (Block 11)
- Inconsistent confidence semantics producing `green_roof: false (0.00)`. Unified to probability of
  the reported class. (Block 12)
- The UTF-16 `.gitignore`. Diagnosed with a hex dump, converted the repo to UTF-8. (Block 13)

## Not recoverable from these sources

- Green roof: the RGB ortho has no NIR band, so there is no true NDVI. Shipped as advisory only; a
  real answer needs Vienna's CIR ortho or the 2023 RGBI LiDAR.
- Thermal / insulation: the only open thermal source (ECOSTRESS) is ~70 m per pixel — one pixel
  spans many buildings. Not resolvable per roof, and I don't pretend otherwise.
- Sub-metre superstructure identity: small vents and thin chimneys blur away at 0.5 m DSM
  resolution; geometry finds them, nothing open can name them.

## Future ideas (scoped, with reasons they are not in this build)

- Per-superstructure CLIP: height localises each object, the ortho crop names it. Deferred because
  a 1–3 m² object is ~15 px at 0.1 m/px, far below CLIP's useful input; it becomes worthwhile with
  drone or oblique imagery where a chimney is 100+ px.
- Union FMZK with OSM footprints so each covers the other's gaps, FMZK as the trusted base.
- Per-facet roof typing to replace the global 10° flat/pitch threshold.
- Stratified sampling for a formal evaluation, so rare classes (the one PV roof) are not missed by a
  uniform draw. Production processes every footprint anyway.
