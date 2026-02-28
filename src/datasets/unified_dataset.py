"""Unified multi-source dataset merging PB + CG orthomosaics for 4-class supervision.

Combines multiple dataset roots (each with its own TIFFs and SHP directory)
into a single training/validation dataset.  Maintains memory-safe windowed
raster reading, on-the-fly CRS reprojection, and minority-aware centroid
patch sampling from the original ``MultiClassDataset``.

Classes:
    0 = Background
    1 = Road
    2 = Bridge
    3 = Built-Up Area
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as A
import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import (
    Window,
    bounds as window_bounds,
    transform as window_transform,
)
from torch.utils.data import Dataset

# Re-export transforms so callers can keep one import path
from src.datasets.multiclass_dataset import get_train_transform, get_val_transform


# ── Per-source configuration ────────────────────────────────────────────────

# Each entry maps SHP filenames → class id.  The actual filenames on disk
# differ between PB (truncated) and CG (full) datasets.

PB_CLASS_MAPPING: dict[str, int] = {
    "Road.shp": 1,
    "Bridge.shp": 2,
    "Built_Up_Area_typ.shp": 3,
}

CG_CLASS_MAPPING: dict[str, int] = {
    "Road.shp": 1,
    "Bridge.shp": 2,
    "Built_Up_Area_type.shp": 3,
}

# Default dataset sources — each dict describes one orthomosaic collection.
# ``tiff_dir`` contains the rasters, ``shp_dir`` the corresponding shapefiles.
DEFAULT_SOURCES: list[dict] = [
    {
        "name": "PB",
        "tiff_dir": "data/Raz/PB_training_dataSet_shp_file",
        "shp_dir": "data/Raz/PB_training_dataSet_shp_file/shp-file",
        "class_mapping": PB_CLASS_MAPPING,
    },
    {
        "name": "CG-2",
        "tiff_dir": "data/Raz/Training_dataSet_2",
        "shp_dir": "data/Raz/CG_shp-file/shp-file",
        "class_mapping": CG_CLASS_MAPPING,
    },
    {
        "name": "CG-3",
        "tiff_dir": "data/Raz/Training_dataSet_3",
        "shp_dir": "data/Raz/CG_shp-file/shp-file",
        "class_mapping": CG_CLASS_MAPPING,
    },
]

CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Road",
    2: "Bridge",
    3: "Built-Up Area",
}


# ── Helper dataclass for per-TIFF bookkeeping ───────────────────────────────

class _TiffEntry:
    """Lightweight container for one TIFF and its associated metadata."""

    __slots__ = (
        "path", "source_name", "shp_dir", "class_mapping",
        "crs", "transform", "bounds", "height", "width",
        "layers", "centroids",
    )

    def __init__(
        self,
        path: Path,
        source_name: str,
        shp_dir: Path,
        class_mapping: dict[str, int],
    ) -> None:
        self.path = path
        self.source_name = source_name
        self.shp_dir = shp_dir
        self.class_mapping = class_mapping
        # Populated later
        self.crs = None
        self.transform = None
        self.bounds = None
        self.height: int = 0
        self.width: int = 0
        self.layers: dict[int, gpd.GeoDataFrame] = {}
        self.centroids: list[tuple[int, int, int]] = []


# ── Unified dataset ─────────────────────────────────────────────────────────

class UnifiedMultiClassDataset(Dataset):
    """Unified multi-source dataset for PB + CG multi-class segmentation.

    Merges TIFFs from multiple dataset roots into a single sample list.
    Each TIFF retains its own SHP layers (reprojected to the raster CRS).
    Supports train/val splitting by TIFF filename and minority-aware
    centroid-based patch sampling.
    """

    def __init__(
        self,
        sources: list[dict] | None = None,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        patch_size: int = 512,
        patches_per_image: int = 50,
        positive_ratio_threshold: float = 0.03,
        positive_sampling_prob: float = 0.7,
        train_tiffs: list[str] | None = None,
        val_tiffs: list[str] | None = None,
        debug_sampling: bool = False,
    ) -> None:
        """
        Args:
            sources: List of dataset source dicts, each with keys
                ``name``, ``tiff_dir``, ``shp_dir``, ``class_mapping``.
                Defaults to ``DEFAULT_SOURCES`` (PB + CG-2 + CG-3).
            split: ``'train'`` or ``'val'``.
            transform: Albumentations pipeline.
            patch_size: Spatial crop size in pixels.
            patches_per_image: Patches sampled per TIFF per epoch.
            positive_ratio_threshold: Min foreground ratio to accept a patch.
            positive_sampling_prob: Prob of retrying low-positive patches.
            train_tiffs: Explicit list of TIFF *stems* for training split.
            val_tiffs: Explicit list of TIFF *stems* for validation split.
            debug_sampling: Print sampling diagnostics.
        """
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_ratio_threshold = positive_ratio_threshold
        self.positive_sampling_prob = positive_sampling_prob
        self.debug_sampling = debug_sampling
        self._debug_counter = 0

        if sources is None:
            sources = DEFAULT_SOURCES

        # ── 1. Collect all TIFFs across sources ───────────────────────────────
        all_entries: list[_TiffEntry] = []
        for src_cfg in sources:
            tiff_dir = Path(src_cfg["tiff_dir"])
            shp_dir = Path(src_cfg["shp_dir"])
            name = src_cfg["name"]
            class_mapping = src_cfg["class_mapping"]

            if not shp_dir.exists():
                print(f"⚠️  SHP dir missing for {name}: {shp_dir}")
                continue

            tiff_paths = sorted(
                list(tiff_dir.glob("*.tif")) + list(tiff_dir.glob("*.tiff"))
            )
            for tp in tiff_paths:
                all_entries.append(
                    _TiffEntry(tp, name, shp_dir, class_mapping)
                )

        # ── 2. Filter by split (train_tiffs / val_tiffs) ─────────────────────
        if train_tiffs is not None or val_tiffs is not None:
            train_set = set(train_tiffs or [])
            val_set = set(val_tiffs or [])
            if split == "train" and train_set:
                all_entries = [e for e in all_entries if e.path.stem in train_set]
            elif split == "val" and val_set:
                all_entries = [e for e in all_entries if e.path.stem in val_set]

        # ── 3. Open each TIFF (metadata only) and load SHP layers ────────────
        valid_entries: list[_TiffEntry] = []
        for entry in all_entries:
            try:
                with rasterio.open(str(entry.path)) as src:
                    entry.crs = src.crs
                    entry.transform = src.transform
                    entry.bounds = src.bounds
                    entry.height = src.height
                    entry.width = src.width

                    if entry.height < patch_size or entry.width < patch_size:
                        print(f"⚠️  Skipping {entry.path.name} (too small)")
                        continue
            except Exception as exc:
                print(f"⚠️  Cannot open {entry.path.name}: {exc!s:.60s} — skipped")
                continue

            # Load + reproject SHP layers to this TIFF's CRS
            entry.layers = self._load_layers(
                entry.shp_dir, entry.class_mapping, entry.crs
            )
            valid_entries.append(entry)

        if not valid_entries:
            raise ValueError("No valid TIFFs found across all sources")

        self.entries = valid_entries

        # ── 4. Print inventory ────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  UnifiedMultiClassDataset  split={split}  patch={patch_size}")
        print(f"{'─'*60}")
        for i, e in enumerate(self.entries):
            overlap_cls = self._overlapping_classes(e)
            print(
                f"  [{i}] {e.source_name:5s}  {e.path.name[:50]:50s}  "
                f"CRS={e.crs}  {e.width}×{e.height}  "
                f"classes={overlap_cls}"
            )
        print(f"  Total TIFFs: {len(self.entries)}")
        print(f"{'─'*60}\n")

        # ── 5. Precompute centroids for minority-aware sampling ───────────────
        self._precompute_feature_centroids()

    # ── SHP loading (per-TIFF CRS) ──────────────────────────────────────────

    @staticmethod
    def _load_layers(
        shp_dir: Path,
        class_mapping: dict[str, int],
        target_crs,
    ) -> dict[int, gpd.GeoDataFrame]:
        """Read SHP files and reproject to *target_crs*."""
        layers: dict[int, gpd.GeoDataFrame] = {}
        for shp_name, class_id in class_mapping.items():
            shp_path = shp_dir / shp_name
            if not shp_path.exists():
                continue
            gdf = gpd.read_file(shp_path)
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            layers[class_id] = gdf
        return layers

    @staticmethod
    def _overlapping_classes(entry: _TiffEntry) -> list[int]:
        """Return class ids with ≥1 feature overlapping the TIFF bounds."""
        if entry.bounds is None:
            return []
        rb = entry.bounds
        overlap = []
        for cid, gdf in entry.layers.items():
            if len(gdf) == 0:
                continue
            sx1, sy1, sx2, sy2 = gdf.total_bounds
            if sx1 < rb.right and sx2 > rb.left and sy1 < rb.top and sy2 > rb.bottom:
                overlap.append(cid)
        return sorted(overlap)

    # ── Centroid pre-computation ─────────────────────────────────────────────

    def _precompute_feature_centroids(self) -> None:
        """Index per-TIFF centroids for minority-aware sampling."""
        self._centroids: dict[int, list[tuple[int, int, int]]] = {}

        for idx, entry in enumerate(self.entries):
            centroids: list[tuple[int, int, int]] = []
            inv_t = ~entry.transform
            b = entry.bounds
            h, w = entry.height, entry.width

            for cid, gdf in entry.layers.items():
                if len(gdf) == 0:
                    continue
                gdf_overlap = gdf.cx[b.left:b.right, b.bottom:b.top]
                for geom in gdf_overlap.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    cx, cy = geom.centroid.x, geom.centroid.y
                    col, row = inv_t * (cx, cy)
                    row, col = int(row), int(col)
                    if 0 <= row < h and 0 <= col < w:
                        centroids.append((cid, row, col))
            self._centroids[idx] = centroids

        total = sum(len(v) for v in self._centroids.values())
        print(f"✓ Pre-indexed {total} feature centroids across {len(self.entries)} TIFFs")

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tiff_idx = idx % len(self.entries)
        entry = self.entries[tiff_idx]

        with rasterio.open(str(entry.path)) as src:
            height, width = src.height, src.width

            if self.split == "train":
                image, mask = self._sample_train_patch(
                    src, entry, tiff_idx, height, width
                )
            else:
                image, mask = self._sample_val_patch(src, entry, height, width, idx)

        # Augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask

    # ── Patch sampling helpers ───────────────────────────────────────────────

    def _sample_train_patch(
        self, src, entry: _TiffEntry, tiff_idx: int, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a training patch with minority-aware logic."""
        ps = self.patch_size
        max_attempts = 10
        image = mask = None

        for _ in range(max_attempts):
            feat_list = self._centroids.get(tiff_idx, [])
            if feat_list and np.random.rand() < 0.5:
                _cls, cy, cx = feat_list[np.random.randint(len(feat_list))]
                jitter = ps // 4
                cy += np.random.randint(-jitter, jitter + 1)
                cx += np.random.randint(-jitter, jitter + 1)
                y = max(0, min(cy - ps // 2, height - ps))
                x = max(0, min(cx - ps // 2, width - ps))
            else:
                y = np.random.randint(0, height - ps + 1)
                x = np.random.randint(0, width - ps + 1)

            win = Window(x, y, ps, ps)
            image = src.read([1, 2, 3], window=win).transpose(1, 2, 0).astype(np.uint8)
            mask = self._rasterize_patch(win, src.transform, entry.layers)

            pos_ratio = float((mask > 0).sum()) / float(ps * ps)
            if self.debug_sampling and self._debug_counter % 100 == 0:
                print(f"[{entry.source_name}] positive_ratio: {pos_ratio:.6f}")
            self._debug_counter += 1

            if pos_ratio >= self.positive_ratio_threshold:
                break
            if np.random.rand() < self.positive_sampling_prob:
                continue
            break

        if image is None or mask is None:
            raise RuntimeError(f"Failed to sample patch from {entry.path.name}")
        return image, mask

    def _sample_val_patch(
        self, src, entry: _TiffEntry, height: int, width: int, idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic random patch for validation (seeded by sample index).

        Each ``idx`` maps to a unique, reproducible random position so that
        validation patches cover the full TIFF extent rather than only the
        centre crop.
        """
        ps = self.patch_size
        rng = np.random.RandomState(seed=idx)
        y = rng.randint(0, height - ps + 1)
        x = rng.randint(0, width - ps + 1)
        win = Window(x, y, ps, ps)
        image = src.read([1, 2, 3], window=win).transpose(1, 2, 0).astype(np.uint8)
        mask = self._rasterize_patch(win, src.transform, entry.layers)
        return image, mask

    # ── Rasterization (shared, identical to original) ────────────────────────

    @staticmethod
    def _rasterize_patch(
        window: Window,
        raster_transform,
        layers: dict[int, gpd.GeoDataFrame],
        patch_size: int | None = None,
    ) -> np.ndarray:
        """Rasterize SHP layers for a window patch."""
        ps = patch_size or window.width
        mask = np.zeros((ps, ps), dtype=np.uint8)
        ptf = window_transform(window, raster_transform)

        for class_id, gdf in layers.items():
            if len(gdf) == 0:
                continue
            try:
                minx, miny, maxx, maxy = window_bounds(window, raster_transform)
                gdf_clip = gdf.cx[minx:maxx, miny:maxy]
                if len(gdf_clip) == 0:
                    continue
                shapes = ((geom, class_id) for geom in gdf_clip.geometry)
                rasterized = rasterize(
                    shapes,
                    out_shape=(ps, ps),
                    transform=ptf,
                    fill=0,
                    dtype=np.uint8,
                )
                mask = np.maximum(mask, rasterized)
            except Exception as e:
                print(f"Warning: rasterize error class {class_id}: {e}")
        return mask

    # ── Utility ──────────────────────────────────────────────────────────────

    def get_tiff_stem(self, idx: int) -> str:
        return self.entries[idx % len(self.entries)].path.stem
