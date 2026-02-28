"""Memory-safe multi-class dataset with windowed raster reading."""

from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import geopandas as gpd
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform, bounds as window_bounds
from torch.utils.data import Dataset


class MultiClassDataset(Dataset):
    """Memory-safe multi-class dataset with windowed raster reading."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        patch_size: int = 512,
        patches_per_image: int = 50,
        positive_ratio_threshold: float = 0.03,
        positive_sampling_prob: float = 0.7,
        debug_sampling: bool = False,
    ) -> None:
        """
        Initialize MultiClassDataset with automatic SHP pairing and CRS alignment.

        Supports two dataset layouts:
          PB flat layout:  data_root/*.tif + data_root/shp-file/*.shp
          Inria layout:    data_root/split/images/*.tif + external shp dir

        Args:
            data_root: Root directory containing ORI rasters (or Inria split root).
            split: 'train' or 'val' — only used for Inria layout.
            transform: Albumentations transform pipeline.
            patch_size: Crop size for patch sampling.
            patches_per_image: Number of random patch samples drawn per image per epoch.
                Effective dataset length = len(images) * patches_per_image.
            positive_ratio_threshold: Minimum positive pixel ratio to accept a patch.
            positive_sampling_prob: Probability of resampling low-positive patches.
            debug_sampling: Print positive ratio debug info.
        """
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_ratio_threshold = positive_ratio_threshold
        self.positive_sampling_prob = positive_sampling_prob
        self.debug_sampling = debug_sampling
        self._debug_counter = 0

        self.data_root = Path(data_root)
        data_root_lower = str(self.data_root).lower()

        # ── Detect PB flat layout ──────────────────────────────────────────────
        # PB layout: TIFFs directly in data_root, SHPs in data_root/shp-file/
        # Class mapping uses truncated Built_Up_Area_typ.shp filename
        if "pb_training" in data_root_lower:
            self.images_dir = self.data_root
            self.shp_dir = self.data_root / "shp-file"
            self.class_mapping = {
                "Road.shp": 1,
                "Railway.shp": 2,
                "Bridge.shp": 3,
                "Built_Up_Area_typ.shp": 4,   # actual filename on disk (truncated)
            }
        else:
            # Inria / AerialImageDataset layout: split/images subdir
            self.images_dir = self.data_root / split / "images"
            self.shp_dir = Path("data/Raz/CG_shp-file/shp-file")
            self.class_mapping = {
                "Road.shp": 1,
                "Railway.shp": 2,
                "Bridge.shp": 3,
                "Built_Up_Area_type.shp": 4,
            }

        if not self.shp_dir.exists():
            raise FileNotFoundError(
                f"SHP directory not found: {self.shp_dir.resolve()}"
            )

        # ── Collect TIFF / PNG files ───────────────────────────────────────────
        self.image_paths = sorted(
            list(self.images_dir.glob("*.tif"))
            + list(self.images_dir.glob("*.tiff"))
            + list(self.images_dir.glob("*.png"))
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No raster files found in {self.images_dir}")

        # ── Get CRS from first raster ──────────────────────────────────────────
        with rasterio.open(str(self.image_paths[0])) as src:
            self.raster_crs = src.crs
            self.raster_transform = src.transform
            self.raster_bounds = src.bounds

        # ── Load and reproject SHP layers once ────────────────────────────────
        self.layers: dict[int, gpd.GeoDataFrame] = {}
        missing_shps = []
        for shp_name, class_id in self.class_mapping.items():
            shp_path = self.shp_dir / shp_name
            if shp_path.exists():
                gdf = gpd.read_file(shp_path)
                if gdf.crs != self.raster_crs:
                    gdf = gdf.to_crs(self.raster_crs)
                self.layers[class_id] = gdf
            else:
                missing_shps.append(str(shp_path))

        if missing_shps:
            print(f"⚠️  Warning: SHP files not found (will be skipped):")
            for p in missing_shps:
                print(f"     {p}")

        # ── Spatial overlap validation ─────────────────────────────────────────
        self._validate_spatial_overlap()

        # ── Precompute feature locations for minority-aware sampling ──────────
        self._precompute_feature_centroids()

    def _validate_spatial_overlap(self) -> None:
        """Warn if no loaded SHP layer spatially overlaps the first raster."""
        if not self.layers:
            print("⚠️  No SHP layers loaded — masks will be all background (class 0).")
            return

        rb = self.raster_bounds
        overlapping = []
        for class_id, gdf in self.layers.items():
            if len(gdf) == 0:
                continue
            sx1, sy1, sx2, sy2 = gdf.total_bounds
            if sx1 < rb.right and sx2 > rb.left and sy1 < rb.top and sy2 > rb.bottom:
                overlapping.append(class_id)

        if overlapping:
            print(f"✓ SHP spatial overlap confirmed for classes: {overlapping}")
        else:
            print(
                f"⚠️  WARNING: No SHP layer overlaps raster bounds {rb}.\n"
                f"   Masks will be all background. Check dataset pairing."
            )

    def _precompute_feature_centroids(self) -> None:
        """Precompute pixel-space centroids of all SHP features per TIFF.

        Stores a dict mapping TIFF index → list of (class_id, row, col)
        tuples.  Used by minority-aware patch sampling in ``__getitem__``.
        No raster pixel data is read — only metadata transforms.
        """
        self._feature_centroids: dict[int, list[tuple[int, int, int]]] = {}

        for tiff_idx, img_path in enumerate(self.image_paths):
            centroids: list[tuple[int, int, int]] = []
            with rasterio.open(str(img_path)) as src:
                h, w = src.height, src.width
                inv_transform = ~src.transform     # geographic → pixel
                b = src.bounds

                for class_id, gdf in self.layers.items():
                    if len(gdf) == 0:
                        continue
                    # Spatial filter: features overlapping this TIFF
                    gdf_overlap = gdf.cx[b.left:b.right, b.bottom:b.top]
                    for geom in gdf_overlap.geometry:
                        if geom is None or geom.is_empty:
                            continue
                        cx, cy = geom.centroid.x, geom.centroid.y
                        col, row = inv_transform * (cx, cy)
                        row, col = int(row), int(col)
                        if 0 <= row < h and 0 <= col < w:
                            centroids.append((class_id, row, col))

            self._feature_centroids[tiff_idx] = centroids

        total = sum(len(v) for v in self._feature_centroids.values())
        print(f"✓ Pre-indexed {total} feature centroids for minority-aware sampling")

    def __len__(self) -> int:
        """Return total number of patches: num_images * patches_per_image."""
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and multi-class mask pair using windowed reading.

        Args:
            idx: Index of sample (mapped to image via idx % num_images).

        Returns:
            Tuple of (image, mask) tensors.
            image: (3, H, W) normalized
            mask: (H, W) long dtype with class IDs {0,1,2,3,4}
        """
        # Map global index → which image to read from
        idx = idx % len(self.image_paths)
        img_path = self.image_paths[idx]

        # Open raster (do NOT read full image)
        with rasterio.open(str(img_path)) as src:
            height = src.height
            width = src.width

            if height < self.patch_size or width < self.patch_size:
                raise ValueError(
                    f"Image {img_path.name} smaller than patch_size={self.patch_size}"
                )

            # Sample window
            if self.split == "train":
                max_attempts = 10
                image_window = None
                mask = None

                for attempt in range(max_attempts):
                    # ── Minority-aware sampling ──────────────────────────
                    # 50 % chance: centre patch on a random feature centroid
                    # 50 % chance: purely random patch (original behaviour)
                    feat_list = self._feature_centroids.get(idx, [])
                    if feat_list and np.random.rand() < 0.5:
                        _cls, cy, cx = feat_list[
                            np.random.randint(len(feat_list))
                        ]
                        # Spatial jitter avoids always centering exactly
                        jitter = self.patch_size // 4
                        cy += np.random.randint(-jitter, jitter + 1)
                        cx += np.random.randint(-jitter, jitter + 1)
                        y = max(0, min(cy - self.patch_size // 2,
                                       height - self.patch_size))
                        x = max(0, min(cx - self.patch_size // 2,
                                       width - self.patch_size))
                    else:
                        y = np.random.randint(0, height - self.patch_size + 1)
                        x = np.random.randint(0, width - self.patch_size + 1)

                    image_window = Window(x, y, self.patch_size, self.patch_size)

                    # Read image patch
                    image = src.read([1, 2, 3], window=image_window)
                    image = image.transpose(1, 2, 0).astype(np.uint8)

                    # Rasterize mask patch
                    mask = self._rasterize_patch(image_window, src.transform)

                    # Check positive ratio
                    positive_ratio = float((mask > 0).sum()) / float(
                        self.patch_size * self.patch_size
                    )

                    if self.debug_sampling and (self._debug_counter % 100 == 0):
                        print(f"positive_ratio: {positive_ratio:.6f}")
                    self._debug_counter += 1

                    if positive_ratio >= self.positive_ratio_threshold:
                        break

                    if np.random.rand() < self.positive_sampling_prob:
                        continue
                    break

                if image is None or mask is None:
                    raise RuntimeError("Failed to sample patch")

            else:
                # Validation: center crop
                y = (height - self.patch_size) // 2
                x = (width - self.patch_size) // 2
                image_window = Window(x, y, self.patch_size, self.patch_size)

                image = src.read([1, 2, 3], window=image_window)
                image = image.transpose(1, 2, 0).astype(np.uint8)

                mask = self._rasterize_patch(image_window, src.transform)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert mask to torch.long
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask

    def _rasterize_patch(
        self, window: Window, raster_transform
    ) -> np.ndarray:
        """
        Rasterize SHP layers for a window patch.

        Args:
            window: Rasterio window object.
            raster_transform: Affine transform of raster.

        Returns:
            Mask array (H, W) with class IDs {0,1,2,3,4}.
        """
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        # Compute window transform for this patch
        patch_transform = window_transform(window, raster_transform)

        # Rasterize each class
        for class_id, gdf in self.layers.items():
            if len(gdf) == 0:
                continue

            try:
                # Get geometries within bounds
                minx, miny, maxx, maxy = window_bounds(window, raster_transform)
                gdf_clipped = gdf.cx[minx:maxx, miny:maxy]

                if len(gdf_clipped) == 0:
                    continue

                # Rasterize
                shapes = ((geom, class_id) for geom in gdf_clipped.geometry)
                rasterized = rasterize(
                    shapes,
                    out_shape=(self.patch_size, self.patch_size),
                    transform=patch_transform,
                    fill=0,
                    dtype=np.uint8,
                )
                mask = np.maximum(mask, rasterized)

            except Exception as e:
                print(f"Warning: Rasterization error for class {class_id}: {e}")
                continue

        return mask

    def get_city_name(self, idx: int) -> str:
        """Extract city name from filename."""
        filename = self.image_paths[idx].stem
        city = "".join([c for c in filename if not c.isdigit()]).rstrip("_-")
        return city if city else filename


def get_train_transform(image_size: int = 512) -> A.Compose:
    """Get training augmentation pipeline."""
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=10,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transform(image_size: int = 512) -> A.Compose:
    """Get validation transform pipeline."""
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def create_city_split(
    dataset: MultiClassDataset,
    val_cities: Optional[list[str]] = None,
    val_ratio: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Create train/val split by city."""
    city_to_indices: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        city = dataset.get_city_name(idx)
        if city not in city_to_indices:
            city_to_indices[city] = []
        city_to_indices[city].append(idx)

    cities = sorted(city_to_indices.keys())

    if val_cities is None:
        num_val_cities = max(1, int(len(cities) * val_ratio))
        val_cities = cities[-num_val_cities:]

    train_indices = []
    val_indices = []

    for city in cities:
        if city in val_cities:
            val_indices.extend(city_to_indices[city])
        else:
            train_indices.extend(city_to_indices[city])

    return train_indices, val_indices
