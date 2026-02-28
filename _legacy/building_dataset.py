"""Dataset class for aerial building segmentation."""

from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class BuildingDataset(Dataset):
    """Dataset for loading aerial images and building segmentation masks."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        use_processed: bool = True,
        positive_ratio_threshold: float = 0.03,
        positive_sampling_prob: float = 0.7,
        patch_size: int = 512,
        debug_sampling: bool = False,
    ) -> None:
        """
        Initialize BuildingDataset.

        Args:
            data_root: Root directory containing AerialImageDataset.
            split: 'train' or 'test'.
            transform: Albumentations transform pipeline.
            use_processed: Use PNG from processed/ instead of TIFF from raw/.
            positive_ratio_threshold: Minimum positive ratio to accept a patch.
            positive_sampling_prob: Probability of resampling low-positive patches.
            patch_size: Crop size for training and evaluation.
            debug_sampling: Print sampling stats occasionally when True.
        """
        self.split = split
        self.transform = transform
        self.positive_ratio_threshold = positive_ratio_threshold
        self.positive_sampling_prob = positive_sampling_prob
        self.patch_size = patch_size
        self.debug_sampling = debug_sampling
        self._debug_counter = 0
        
        # Determine data path
        if use_processed:
            data_path = Path(data_root).parent / "processed" / Path(data_root).name
            if not data_path.exists():
                print(f"Processed data not found at {data_path}, falling back to raw")
                data_path = Path(data_root)
        else:
            data_path = Path(data_root)
        
        self.data_root = data_path
        self.images_dir = self.data_root / split / "images"
        self.masks_dir = self.data_root / split / "gt"

        # Get all image files (prefer PNG, fallback to TIF)
        png_files = sorted(list(self.images_dir.glob("*.png")))
        tif_files = sorted(list(self.images_dir.glob("*.tif")) + list(self.images_dir.glob("*.tiff")))
        
        self.image_paths = png_files if png_files else tif_files
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get image and mask pair.

        Args:
            idx: Index of sample.

        Returns:
            Tuple of (image, mask) tensors.
            image: (3, H, W) normalized
            mask: (1, H, W) binary {0, 1}
        """
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.masks_dir / img_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Convert mask from 255 to 1
        mask = (mask > 127).astype(np.float32)

        height, width = mask.shape[:2]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"Image {img_path.name} smaller than patch_size={self.patch_size}"
            )

        if self.split == "train":
            max_attempts = 10
            image_crop = None
            mask_crop = None

            for attempt in range(max_attempts):
                y = np.random.randint(0, height - self.patch_size + 1)
                x = np.random.randint(0, width - self.patch_size + 1)

                image_crop = image[y:y + self.patch_size, x:x + self.patch_size]
                mask_crop = mask[y:y + self.patch_size, x:x + self.patch_size]

                building_ratio = float(mask_crop.sum()) / float(self.patch_size * self.patch_size)

                if self.debug_sampling and (self._debug_counter % 100 == 0):
                    print(f"building_ratio: {building_ratio:.6f}")
                self._debug_counter += 1

                if building_ratio >= self.positive_ratio_threshold:
                    break

                if np.random.rand() < self.positive_sampling_prob:
                    continue
                break

            image = image_crop
            mask = mask_crop
        else:
            y = (height - self.patch_size) // 2
            x = (width - self.patch_size) // 2
            image = image[y:y + self.patch_size, x:x + self.patch_size]
            mask = mask[y:y + self.patch_size, x:x + self.patch_size]

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Add channel dimension to mask
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        return image, mask
    
    def get_city_name(self, idx: int) -> str:
        """
        Extract city name from filename.
        
        Assumes format: cityname123.ext
        
        Args:
            idx: Sample index.
            
        Returns:
            City name prefix.
        """
        filename = self.image_paths[idx].stem
        # Extract city name (remove trailing digits)
        city = "".join([c for c in filename if not c.isdigit()]).rstrip("_-")
        return city if city else filename


def get_train_transform(image_size: int = 512) -> A.Compose:
    """
    Get training augmentation pipeline.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations compose transform.
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 512) -> A.Compose:
    """
    Get validation transform pipeline.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations compose transform.
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_city_split(
    dataset: BuildingDataset,
    val_cities: Optional[list[str]] = None,
    val_ratio: float = 0.2,
) -> tuple[list[int], list[int]]:
    """
    Create train/val split by city to avoid data leakage.
    
    Args:
        dataset: BuildingDataset instance.
        val_cities: Specific city names for validation. If None, auto-select by ratio.
        val_ratio: Ratio of cities to use for validation.
        
    Returns:
        Tuple of (train_indices, val_indices).
    """
    # Group indices by city
    city_to_indices: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        city = dataset.get_city_name(idx)
        if city not in city_to_indices:
            city_to_indices[city] = []
        city_to_indices[city].append(idx)
    
    cities = sorted(city_to_indices.keys())
    
    # Select validation cities
    if val_cities is None:
        num_val_cities = max(1, int(len(cities) * val_ratio))
        val_cities = cities[-num_val_cities:]  # Last N cities for reproducibility
    
    # Split indices
    train_indices = []
    val_indices = []
    
    for city in cities:
        if city in val_cities:
            val_indices.extend(city_to_indices[city])
        else:
            train_indices.extend(city_to_indices[city])
    
    return train_indices, val_indices
