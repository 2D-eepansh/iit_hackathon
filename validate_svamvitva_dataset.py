"""
Pre-training dataset validation script for SVAMITVA multi-class segmentation.

Exits with code 0 on PASS, code 1 on FAIL.

Usage:
    python validate_svamvitva_dataset.py
"""

import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit only these paths if the layout changes
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("data/Raz/PB_training_dataSet_shp_file")
SHP_DIR   = DATA_ROOT / "shp-file"

REQUIRED_SHPS: dict[str, int] = {
    "Road.shp":               1,
    "Railway.shp":            2,
    "Bridge.shp":             3,
    "Built_Up_Area_typ.shp":  4,   # actual on-disk name (truncated)
}

CLASS_NAMES = {
    0: "Background",
    1: "Road",
    2: "Railway",
    3: "Bridge",
    4: "Built-Up Area",
}

PATCH_SIZE      = 512
N_TEST_PATCHES  = 20
DEBUG_OUT       = Path("debug_overlay.png")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_FAILURES: list[str] = []
_PASSES:   list[str] = []

BOLD  = "\033[1m"
RED   = "\033[91m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
RESET = "\033[0m"

SEP   = "─" * 70


def _ok(label: str, detail: str = "") -> None:
    msg = f"  {GREEN}✓{RESET} {label}" + (f"  [{detail}]" if detail else "")
    print(msg)
    _PASSES.append(label)


def _fail(label: str, detail: str = "") -> None:
    msg = f"  {RED}✗{RESET} {label}" + (f"  [{detail}]" if detail else "")
    print(msg)
    _FAILURES.append(f"{label}: {detail}" if detail else label)


def _header(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{SEP}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{SEP}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Raster Verification
# ─────────────────────────────────────────────────────────────────────────────

def step1_raster_verification() -> list:
    """Open every TIFF in DATA_ROOT; return list of (path, opened_src_meta)."""
    import rasterio

    _header("STEP 1 — RASTER VERIFICATION")

    tif_paths = sorted(
        list(DATA_ROOT.glob("*.tif")) + list(DATA_ROOT.glob("*.tiff"))
    )

    if not tif_paths:
        _fail("Raster files found", f"No .tif / .tiff files in {DATA_ROOT}")
        return []

    valid_rasters = []

    for p in tif_paths:
        print(f"\n  {p.name}")
        try:
            with rasterio.open(str(p)) as src:
                meta = {
                    "path":   p,
                    "crs":    src.crs,
                    "bounds": src.bounds,
                    "shape":  (src.height, src.width),
                    "dtype":  src.dtypes[0],
                    "bands":  src.count,
                    "transform": src.transform,
                }
            print(f"    CRS:    {meta['crs']}")
            print(f"    Bounds: {meta['bounds']}")
            print(f"    Shape:  {meta['shape'][0]}×{meta['shape'][1]}")
            print(f"    Dtype:  {meta['dtype']}   Bands: {meta['bands']}")

            ok = True
            if meta["crs"] is None:
                _fail(f"{p.name} — CRS", "CRS is None"); ok = False
            else:
                _ok(f"{p.name} — CRS", str(meta["crs"]))

            if meta["bands"] < 3:
                _fail(f"{p.name} — band count", f"Got {meta['bands']}, need ≥3"); ok = False
            else:
                _ok(f"{p.name} — band count", f"{meta['bands']} bands")

            if meta["dtype"] not in ("uint8", "uint16"):
                _fail(f"{p.name} — dtype", f"Got {meta['dtype']}, expected uint8/uint16"); ok = False
            else:
                _ok(f"{p.name} — dtype", meta["dtype"])

            if ok:
                valid_rasters.append(meta)

        except Exception as exc:
            _fail(f"{p.name} — open failed", str(exc))

    return valid_rasters


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Shapefile Verification
# ─────────────────────────────────────────────────────────────────────────────

def step2_shp_verification() -> dict[str, object]:
    """Load each required SHP; return dict of {shp_name: gdf}."""
    import geopandas as gpd

    _header("STEP 2 — SHAPEFILE VERIFICATION")

    loaded: dict[str, object] = {}

    for shp_name in REQUIRED_SHPS:
        shp_path = SHP_DIR / shp_name
        print(f"\n  {shp_name}")

        if not shp_path.exists():
            _fail(f"{shp_name} — file exists", f"Not found: {shp_path}")
            continue

        try:
            gdf = gpd.read_file(str(shp_path))
            print(f"    CRS:          {gdf.crs}")
            print(f"    Geometry:     {gdf.geom_type.unique().tolist()}")
            print(f"    Feature count: {len(gdf)}")

            if len(gdf) == 0:
                _fail(f"{shp_name} — features", "0 features — file is empty")
                continue

            if gdf.crs is None:
                _fail(f"{shp_name} — CRS", "CRS is None")
                continue

            _ok(f"{shp_name}", f"{len(gdf)} features  CRS={gdf.crs.to_epsg()}")
            loaded[shp_name] = gdf

        except Exception as exc:
            _fail(f"{shp_name} — load failed", str(exc))

    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CRS Alignment Check
# ─────────────────────────────────────────────────────────────────────────────

def step3_crs_alignment(raster_metas: list, shp_gdfs: dict) -> dict:
    """Reproject all SHPs to first raster's CRS; return reprojected GDFs."""
    from pyproj import CRS as ProjCRS

    _header("STEP 3 — CRS ALIGNMENT")

    if not raster_metas:
        _fail("CRS alignment", "No valid rasters to align to")
        return {}

    raster_crs = raster_metas[0]["crs"]
    print(f"\n  Target CRS (from raster): {raster_crs}")

    reprojected: dict[str, object] = {}

    for shp_name, gdf in shp_gdfs.items():
        original_crs = gdf.crs
        try:
            if original_crs == raster_crs:
                reprojected[shp_name] = gdf
                _ok(f"{shp_name}", f"Already in target CRS")
            else:
                gdf_reproj = gdf.to_crs(raster_crs)
                reprojected[shp_name] = gdf_reproj
                _ok(
                    f"{shp_name}",
                    f"Reprojected  {original_crs.to_epsg()} → {raster_crs.to_epsg()}",
                )
        except Exception as exc:
            _fail(f"{shp_name} — reprojection failed", str(exc))

    return reprojected


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Spatial Overlap Check
# ─────────────────────────────────────────────────────────────────────────────

def step4_spatial_overlap(raster_metas: list, reprojected_gdfs: dict) -> bool:
    """
    For each raster × each SHP, confirm bounding-box intersection exists.
    Returns True if at least one (raster, shp) pair overlaps.
    """
    from shapely.geometry import box

    _header("STEP 4 — SPATIAL OVERLAP CHECK")

    if not raster_metas:
        _fail("Spatial overlap", "No valid rasters")
        return False

    all_rasters_valid = True

    for rmeta in raster_metas:
        b = rmeta["bounds"]
        raster_box = box(b.left, b.bottom, b.right, b.top)
        print(f"\n  Raster: {rmeta['path'].name}")
        print(f"    Bounds: ({b.left:.2f}, {b.bottom:.2f}, {b.right:.2f}, {b.top:.2f})")

        overlap_count = 0
        for shp_name, gdf in reprojected_gdfs.items():
            tb = gdf.total_bounds   # (minx, miny, maxx, maxy)
            shp_box = box(tb[0], tb[1], tb[2], tb[3])
            overlaps = raster_box.intersects(shp_box)

            if overlaps:
                intersection = raster_box.intersection(shp_box)
                print(
                    f"    {GREEN}✓{RESET} {shp_name:<30}"
                    f"  overlap ≈ {intersection.area:.2f} sq-units"
                )
                overlap_count += 1
            else:
                # Sparse layers (Railway, Bridge) may legitimately not cover all tiles
                print(f"    {CYAN}–{RESET} {shp_name:<30}  no intersection (sparse layer)")

        if overlap_count == 0:
            _fail(
                f"{rmeta['path'].name} — feature coverage",
                "ZERO shapefiles overlap this raster — check dataset pairing / CRS",
            )
            all_rasters_valid = False
        else:
            _ok(
                f"{rmeta['path'].name} — feature coverage",
                f"{overlap_count}/{len(reprojected_gdfs)} layer(s) overlap",
            )

    return all_rasters_valid


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Patch Rasterization Test
# ─────────────────────────────────────────────────────────────────────────────

def step5_patch_rasterization(raster_metas: list, reprojected_gdfs: dict) -> Counter:
    """Randomly sample patches and rasterize SHP geometries; count classes."""
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window, transform as window_transform
    from shapely.geometry import box

    _header("STEP 5 — PATCH RASTERIZATION TEST")

    if not raster_metas or not reprojected_gdfs:
        _fail("Patch rasterization", "Missing rasters or SHPs")
        return Counter()

    class_counts: Counter = Counter()
    patches_ok = 0
    patches_fail = 0

    rng = np.random.default_rng(seed=42)

    patches_per_raster = max(1, N_TEST_PATCHES // len(raster_metas))

    for rmeta in raster_metas:
        p = rmeta["path"]
        h, w = rmeta["shape"]
        raster_crs = rmeta["crs"]

        if h < PATCH_SIZE or w < PATCH_SIZE:
            _fail(f"{p.name}", f"Raster too small for {PATCH_SIZE}px patch")
            continue

        with rasterio.open(str(p)) as src:
            for _ in range(patches_per_raster):
                try:
                    y = int(rng.integers(0, h - PATCH_SIZE))
                    x = int(rng.integers(0, w - PATCH_SIZE))
                    window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                    win_transform = window_transform(window, src.transform)

                    # Build patch bounding box in raster CRS
                    from rasterio.windows import bounds as window_bounds
                    left, bottom, right, top = window_bounds(window, src.transform)
                    patch_box = box(left, bottom, right, top)

                    # Rasterize all SHP layers into one mask
                    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                    for shp_name, gdf in reprojected_gdfs.items():
                        class_id = int(REQUIRED_SHPS[shp_name])
                        clipped = gdf[gdf.intersects(patch_box)]
                        if len(clipped) == 0:
                            continue
                        # Filter None / empty geometries to avoid rasterize errors
                        valid_geoms = [
                            geom for geom in clipped.geometry
                            if geom is not None and not geom.is_empty
                        ]
                        if not valid_geoms:
                            continue
                        shapes = [(geom, class_id) for geom in valid_geoms]
                        burned = rasterize(
                            shapes,
                            out_shape=(PATCH_SIZE, PATCH_SIZE),
                            transform=win_transform,
                            fill=0,
                            dtype=np.uint8,
                        )
                        # Higher class ID wins where polygons overlap
                        mask = np.where(burned > 0, burned, mask)

                    unique_vals = np.unique(mask).tolist()
                    for v in unique_vals:
                        class_counts[int(v)] += int((mask == v).sum())

                    patches_ok += 1

                except Exception as exc:
                    patches_fail += 1
                    print(f"    ⚠  Patch failed: {exc}")

    print(f"\n  Patches sampled: {patches_ok}  |  Failed: {patches_fail}")
    print()

    # ── Class distribution summary ─────────────────────────────────────────
    total = sum(class_counts.values()) or 1
    print(f"  {'Class':<6} {'Name':<18} {'Pixels':>12}  {'%':>6}  Bar")
    print(f"  {'─'*6} {'─'*18} {'─'*12}  {'─'*6}  {'─'*40}")

    for cid in range(5):
        count = class_counts.get(cid, 0)
        pct   = 100.0 * count / total
        bar   = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        flag  = f"  {GREEN}✓{RESET}" if count > 0 else f"  {RED}✗{RESET}"
        print(f"{flag} {cid:<5} {CLASS_NAMES[cid]:<18} {count:>12,}  {pct:>5.1f}%  {bar[:40]}")

    # ── Validation ─────────────────────────────────────────────────────────
    only_background = all(cid == 0 for cid in class_counts)
    if only_background:
        _fail("Feature rasterization", "Only class 0 (background) found — rasterization is broken")
    else:
        feature_ids = [c for c in class_counts if c != 0 and class_counts[c] > 0]
        _ok("Feature rasterization", f"Feature classes with pixels: {feature_ids}")

    invalid_vals = [v for v in class_counts if v not in range(5)]
    if invalid_vals:
        _fail("Mask value range", f"Values outside 0–4 detected: {invalid_vals}")
    else:
        _ok("Mask value range", "All values in {0, 1, 2, 3, 4}")

    return class_counts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Geometric Alignment Debug Image
# ─────────────────────────────────────────────────────────────────────────────

def step6_debug_overlay(raster_metas: list, reprojected_gdfs: dict) -> None:
    """Save a debug overlay of one patch for visual CRS alignment verification."""
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.features import rasterize
    from rasterio.windows import Window, transform as window_transform
    from rasterio.windows import bounds as window_bounds
    from shapely.geometry import box

    _header("STEP 6 — GEOMETRIC ALIGNMENT DEBUG IMAGE")

    if not raster_metas or not reprojected_gdfs:
        _fail("Debug overlay", "Missing rasters or SHPs — skipping")
        return

    rmeta = raster_metas[0]
    p     = rmeta["path"]
    h, w  = rmeta["shape"]

    if h < PATCH_SIZE or w < PATCH_SIZE:
        _fail("Debug overlay", "Raster too small")
        return

    rng = np.random.default_rng(seed=0)
    y   = int(rng.integers(0, h - PATCH_SIZE))
    x   = int(rng.integers(0, w - PATCH_SIZE))

    try:
        with rasterio.open(str(p)) as src:
            window       = Window(x, y, PATCH_SIZE, PATCH_SIZE)
            win_transform = window_transform(window, src.transform)
            image        = src.read([1, 2, 3], window=window)  # (3, H, W)

        left, bottom, right, top = window_bounds(window, rmeta["transform"])
        patch_box = box(left, bottom, right, top)

        # Build mask
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        for shp_name, gdf in reprojected_gdfs.items():
            class_id = int(REQUIRED_SHPS[shp_name])
            clipped  = gdf[gdf.intersects(patch_box)]
            if len(clipped) == 0:
                continue
            valid_geoms = [
                geom for geom in clipped.geometry
                if geom is not None and not geom.is_empty
            ]
            if not valid_geoms:
                continue
            shapes = [(geom, class_id) for geom in valid_geoms]
            burned = rasterize(
                shapes,
                out_shape=(PATCH_SIZE, PATCH_SIZE),
                transform=win_transform,
                fill=0,
                dtype=np.uint8,
            )
            mask = np.where(burned > 0, burned, mask)

        # Visualise
        img_display = image.transpose(1, 2, 0).astype(np.float32)
        img_display = (img_display - img_display.min()) / (
            img_display.max() - img_display.min() + 1e-6
        )

        COLORS = {
            0: [0.0, 0.0, 0.0],  # background  — transparent
            1: [1.0, 0.0, 0.0],  # road        — red
            2: [0.0, 1.0, 0.0],  # railway     — green
            3: [0.0, 0.0, 1.0],  # bridge      — blue
            4: [1.0, 1.0, 0.0],  # built-up    — yellow
        }
        mask_rgb = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
        for cid, col in COLORS.items():
            mask_rgb[mask == cid] = col

        blended = 0.6 * img_display + 0.4 * mask_rgb

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_display);  axes[0].set_title("TIFF patch");        axes[0].axis("off")
        axes[1].imshow(mask, cmap="tab10", vmin=0, vmax=4)
        axes[1].set_title("Rasterized mask (classes 0-4)"); axes[1].axis("off")
        plt.colorbar(axes[1].images[0], ax=axes[1])
        axes[2].imshow(blended);      axes[2].set_title("Overlay");           axes[2].axis("off")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=col, label=f"{cid} — {CLASS_NAMES[cid]}")
            for cid, col in COLORS.items()
        ]
        axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.suptitle(
            f"Debug overlay  |  {p.name}  |  window ({x},{y})+{PATCH_SIZE}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(str(DEBUG_OUT), dpi=120, bbox_inches="tight")
        plt.close(fig)

        _ok("Debug overlay saved", str(DEBUG_OUT.resolve()))
        print(f"    Open {DEBUG_OUT} to visually verify CRS alignment.")

    except Exception as exc:
        _fail("Debug overlay", str(exc))
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Final Readiness Report
# ─────────────────────────────────────────────────────────────────────────────

def step7_final_report() -> bool:
    """Print PASS / FAIL summary and return True if all checks passed."""
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  SVAMITVA DATASET READINESS REPORT{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")

    if _PASSES:
        for p in _PASSES:
            print(f"  {GREEN}✓{RESET}  {p}")

    if _FAILURES:
        print()
        for f in _FAILURES:
            print(f"  {RED}✗{RESET}  {f}")

    print(f"\n{'─'*70}")
    passed = len(_FAILURES) == 0

    if passed:
        print(f"\n  {GREEN}{BOLD}✓  READY FOR TRAINING{RESET}")
        print(f"\n  Run:  python train.py\n")
    else:
        print(f"\n  {RED}{BOLD}✗  DO NOT TRAIN — FIX {len(_FAILURES)} ISSUE(S) ABOVE{RESET}\n")

    print(f"{'═'*70}\n")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  SVAMITVA PRE-TRAINING VALIDATION{RESET}")
    print(f"{BOLD}  Data root : {DATA_ROOT.resolve()}{RESET}")
    print(f"{BOLD}  SHP dir   : {SHP_DIR.resolve()}{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")

    # Guard: data root must exist
    if not DATA_ROOT.exists():
        print(f"\n{RED}FATAL: DATA_ROOT does not exist: {DATA_ROOT.resolve()}{RESET}\n")
        sys.exit(1)

    if not SHP_DIR.exists():
        print(f"\n{RED}FATAL: SHP_DIR does not exist: {SHP_DIR.resolve()}{RESET}\n")
        sys.exit(1)

    # Run validation steps
    raster_metas   = step1_raster_verification()
    raw_shp_gdfs   = step2_shp_verification()
    reproj_gdfs    = step3_crs_alignment(raster_metas, raw_shp_gdfs)
    overlap_ok     = step4_spatial_overlap(raster_metas, reproj_gdfs)
    class_counts   = step5_patch_rasterization(raster_metas, reproj_gdfs)
    step6_debug_overlay(raster_metas, reproj_gdfs)
    all_passed     = step7_final_report()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
