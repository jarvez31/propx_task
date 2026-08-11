"""Orchestration: config in, one record per building out (task-shaped JSON) + an overview figure."""
import json
import os

from . import appearance, attributes, data, planes, roof_mask, surfaces, viz
from .config import Config

# Two different notions of "confidence" live in the record; label which is which so a reader of the
# raw JSON isn't misled. "p_reported_class" = probability of the value we reported (CLIP softmax of the
# winner). "evidence_support" = a fit / data-completeness proxy (coverage, circular R, valid-pixel
# fraction) — high means well-supported, NOT literally P(the value is correct).
CONFIDENCE_KIND = {
    "area": "evidence_support",  "type": "evidence_support",
    "orientation": "evidence_support", "slope": "evidence_support",
    "material": "p_reported_class", "solar_pv": "p_reported_class",
    "green_roof": "p_reported_class", "condition": "p_reported_class",
}


def _source_line(ortho_year):
    return (f"Stadt Wien OGD: orthophoto lb{ortho_year} (0.1 m) + ALS DSM/DGM 0.5 m (nDSM) + FMZK footprints; "
            f"appearance via CLIP ViT-B/32 zero-shot; structure via RANSAC roof-plane fitting")


def build_scene(config: Config) -> data.Scene:
    footprints = data.fetch_footprints(config.west, config.south, config.east, config.north)
    chosen = data.pick_buildings(footprints, config.n_buildings, config.seed)
    bounds = tuple(chosen.total_bounds)
    sheets = data.dem_sheets_for(bounds)
    dsm_arr, dsm_tf = data.get_dem("dom", sheets, bounds)
    dgm_arr, dgm_tf = data.get_dem("dgm", sheets, bounds)
    ndsm, slope_deg, aspect_deg = surfaces.compute_surfaces(dsm_arr, dsm_tf, dgm_arr, dgm_tf)
    img, ortho_tf, extent = data.fetch_ortho(chosen, config.ortho_year, config.zoom)
    return data.Scene(chosen, dsm_tf, ndsm, slope_deg, aspect_deg, img, ortho_tf, extent)


def extract_building(geom, fmzk_id, scene: data.Scene, config: Config):
    """All attributes for one building. Returns (record, roof_outline)."""
    roof, outline, roof_area = roof_mask.roof_from_height(
        geom, scene.dsm_transform, scene.ndsm, config.roof_min_h, config.min_facet_px)

    facets, coverage = planes.fit_roof_planes(
        roof, scene.ndsm, scene.dsm_transform, config.plane_tol, config.min_facet_px)
    rtype, type_conf = planes.roof_type_from_planes(facets, coverage, config.flat_thresh)
    supers = planes.find_superstructures(roof, scene.ndsm, scene.dsm_transform, facets)
    g = attributes.geometric_attrs(roof, scene.ndsm, scene.slope_deg, scene.aspect_deg)

    on_ortho = roof_mask.roof_mask_on_ortho(roof, scene.dsm_transform, scene.img, scene.ortho_transform)
    roof_img = roof_mask.crop_to_roof(scene.img, on_ortho)

    # Fusion consistency: structure constrains appearance. A gravel/bitumen build-up cannot sit on a
    # steep pitch, so on clearly pitched roofs CLIP only ranks the materials that are physically
    # possible there — otherwise a red tile roof in shadow can win as "flat_gravel" at 35 deg.
    material_prompts = appearance.MATERIAL_PROMPTS
    steep = any(p["slope_deg"] >= config.gravel_max_slope and p["area_m2"] >= 5 for p in facets)
    if steep:
        material_prompts = [p for p in material_prompts if p[0] != "flat_gravel"]

    if roof_img is not None:
        material = appearance.clip_scores(roof_img, material_prompts)
        best_material = max(material, key=material.get)
        pv    = appearance.clip_scores(roof_img, appearance.PV_PROMPTS)
        green = appearance.clip_scores(roof_img, appearance.GREEN_PROMPTS)
        cond  = appearance.clip_scores(roof_img, appearance.COND_PROMPTS)
    else:
        material, best_material = {"unknown": 0.0}, "unknown"
        pv = {"yes": 0.0}
        green = {"yes": 0.0}
        cond = {"good": 0.0, "weathered": 0.0}

    record = {
        "building_id": int(fmzk_id),
        "source_used": _source_line(config.ortho_year),
        "roof": {
            "polygon": attributes.outline_lonlat(outline),
            "area_m2": round(roof_area, 1),
            "footprint_area_m2": round(float(geom.area), 1),
            "type": rtype,
            "material": best_material,
            "orientation_deg": g["orientation_deg"] if rtype != "flat" else None,
            "slope_deg": g["slope_deg"],
            "height_m": g["height_m"],
            "n_planes": len(facets),
            "planes": [{"slope_deg": p["slope_deg"], "aspect_deg": p["aspect_deg"],
                        "area_m2": p["area_m2"]} for p in facets],
            "solar_pv": bool(pv["yes"] > 0.5),
            "green_roof": bool(green["yes"] > 0.5),
            "condition": "weathered" if cond["weathered"] > 0.5 else "good",
            "superstructures": supers,
        },
        "confidence": {
            "area": round(g["valid_frac"], 2),
            "type": type_conf,
            "material": round(material[best_material], 2),
            "orientation": round(g["R"], 2) if rtype != "flat" else 0.0,
            "slope": round(g["valid_frac"] * g["size_ok"], 2),
            "solar_pv": round(max(pv.values()), 2),        # confidence in the REPORTED class (yes or no),
            "green_roof": round(max(green.values()), 2),   # so False+0.00 can't happen; consistent with condition
            "condition": round(max(cond.values()), 2),
        },
        "confidence_kind": CONFIDENCE_KIND,             # what each confidence number actually means
        "clip_raw": {                                   # full zero-shot distributions, not just the winner
            "material":  {k: round(v, 3) for k, v in material.items()},
            "pv":        {k: round(v, 3) for k, v in pv.items()},
            "green":     {k: round(v, 3) for k, v in green.items()},
            "condition": {k: round(v, 3) for k, v in cond.items()},
        },
        "notes": ("roof outline height-derived (nDSM>2m) inside the footprint's outer ring, so roof area can "
                  "exceed footprint area where a mapped courtyard hole is actually covered; area/slope/"
                  "orientation in EPSG:31256; type & superstructures from RANSAC planes; material/pv/green/"
                  "condition from CLIP (RGB only -> green roof has no NIR, treat as advisory). confidence_kind "
                  "says whether each confidence is P(reported class) or an evidence-support proxy "
                  "(coverage/R/valid-frac)."
                  + (" material vocabulary constrained by structure: flat_gravel excluded because a facet "
                     f"pitches >= {config.gravel_max_slope:.0f} deg." if steep else "")),
    }
    return record, outline


def run(config: Config | None = None) -> dict:
    config = config or Config()
    scene = build_scene(config)
    print(f"{len(scene.chosen_buildings)} buildings | DSM {scene.ndsm.shape} | ortho {scene.img.shape}")

    roof_attributes, outlines = {}, {}
    for geom, fmzk_id in zip(scene.chosen_buildings.geometry, scene.chosen_buildings.FMZK_ID):
        record, outline = extract_building(geom, fmzk_id, scene, config)
        roof_attributes[int(fmzk_id)] = record
        outlines[int(fmzk_id)] = outline

    os.makedirs(os.path.dirname(config.out_json) or ".", exist_ok=True)
    with open(config.out_json, "w") as f:
        json.dump(roof_attributes, f, indent=2)

    fig_dir = os.path.dirname(config.fig_path) or "."
    os.makedirs(fig_dir, exist_ok=True)
    viz.overview_figure(scene, roof_attributes, outlines, config.fig_path, config.ortho_year)

    # a few per-building detail panels for the deliverable (interactive views live in the notebook)
    for geom, fmzk_id in list(zip(scene.chosen_buildings.geometry, scene.chosen_buildings.FMZK_ID))[:3]:
        fmzk_id = int(fmzk_id)
        viz.building_panel(scene, geom, fmzk_id, roof_attributes[fmzk_id], config,
                           os.path.join(fig_dir, f"building_{fmzk_id}.png"))

    print(f"saved {config.out_json} and {fig_dir}/ figures")
    return roof_attributes
