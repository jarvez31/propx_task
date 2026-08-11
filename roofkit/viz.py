"""Figures: an all-buildings overview and per-building detail panels for the deliverable."""
import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from . import planes, roof_mask


def overview_figure(scene, roof_attributes, outlines, fig_path, ortho_year=2024):
    order = list(scene.chosen_buildings.FMZK_ID.astype(int))
    buildings_3857 = scene.chosen_buildings.to_crs(epsg=3857)
    outlines_3857 = gpd.GeoSeries([outlines[f] for f in order], crs=31256).to_crs(3857)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(scene.img, extent=[scene.extent[0], scene.extent[1], scene.extent[2], scene.extent[3]])
    buildings_3857.plot(ax=ax, facecolor="none", edgecolor="red",  linewidth=1.0, linestyle=":")   # footprint
    outlines_3857.plot(ax=ax, facecolor="none", edgecolor="cyan", linewidth=1.6)                    # roof outline
    for geom_3857, fmzk_id in zip(outlines_3857.geometry, order):
        r = roof_attributes[fmzk_id]["roof"]; c = geom_3857.centroid
        tag = f"{r['type']}/{r['material']}"
        if r["solar_pv"]:   tag += "+PV"
        if r["green_roof"]: tag += "+green"
        tag += f"\n{len(r['superstructures'])}obj"
        ax.annotate(tag, (c.x, c.y), color="yellow", fontsize=7, ha="center", va="center")
    ax.set_title(f"roof attributes (CLIP + RANSAC planes) — {len(order)} buildings, lb{ortho_year}\n"
                 f"red dotted = FMZK footprint, cyan = nDSM>2m roof outline")
    ax.set_axis_off()
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def building_panel(scene, geom, fmzk_id, record, config, out_path):
    """Static 3-panel detail for one building: ortho crop, nDSM + superstructures, slope."""
    roof, _, _ = roof_mask.roof_from_height(geom, scene.dsm_transform, scene.ndsm,
                                            config.roof_min_h, config.min_facet_px)
    facets, _ = planes.fit_roof_planes(roof, scene.ndsm, scene.dsm_transform,
                                       config.plane_tol, config.min_facet_px)
    residual = np.zeros(roof.shape, "float32")
    if facets:
        residual[roof] = scene.ndsm[roof] - planes.plane_base_image(
            roof, scene.ndsm, scene.dsm_transform, facets)[roof]

    on_ortho = roof_mask.roof_mask_on_ortho(roof, scene.dsm_transform, scene.img, scene.ortho_transform)
    ys, xs = np.where(on_ortho)
    r0, r1 = max(ys.min() - 10, 0), min(ys.max() + 11, scene.img.shape[0])
    c0, c1 = max(xs.min() - 10, 0), min(xs.max() + 11, scene.img.shape[1])
    yy, xx = np.where(roof)
    dr0, dr1 = max(yy.min() - 6, 0), min(yy.max() + 7, roof.shape[0])
    dc0, dc1 = max(xx.min() - 6, 0), min(xx.max() + 7, roof.shape[1])
    sub = lambda A: A[dr0:dr1, dc0:dc1]

    r, c = record["roof"], record["confidence"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(scene.img[r0:r1, c0:c1]); ax[0].set_title("ortho"); ax[0].axis("off")
    im = ax[1].imshow(sub(scene.ndsm), cmap="terrain"); plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[1].contour(sub(roof), levels=[0.5], colors="red", linewidths=1)
    ax[1].contour(sub(residual), levels=[1.0], colors="magenta", linewidths=1)
    ax[1].set_title("nDSM + superstructures"); ax[1].axis("off")
    im = ax[2].imshow(sub(scene.slope_deg), cmap="viridis"); plt.colorbar(im, ax=ax[2], fraction=0.046)
    ax[2].set_title("slope (deg)"); ax[2].axis("off")
    fig.suptitle(f"FMZK {fmzk_id} — {r['type']} (c{c['type']}) / {r['material']} (c{c['material']}) | "
                 f"slope {r['slope_deg']}° | {len(r['superstructures'])} superstructures", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
