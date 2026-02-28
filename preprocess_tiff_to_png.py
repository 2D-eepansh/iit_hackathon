"""Preprocess TIFF images to PNG for faster I/O during training."""

import os
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm


def convert_tiff_to_png(
    data_root: str,
    splits: tuple[str, ...] = ("train",),
    subdirs: tuple[str, ...] = ("images", "gt"),
    output_root: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Convert TIFF images to PNG format.

    Args:
        data_root: Root directory containing raw data.
        splits: Data splits to process.
        subdirs: Subdirectories within each split.
        output_root: Output directory. If None, creates 'processed' alongside 'raw'.
        force: Force reconversion even if PNG exists.
    """
    data_root = Path(data_root)
    
    if output_root is None:
        output_root = data_root.parent / "processed" / data_root.name
    else:
        output_root = Path(output_root)
    
    print(f"Converting TIFF to PNG")
    print(f"Source: {data_root}")
    print(f"Target: {output_root}")
    print("-" * 80)
    
    total_converted = 0
    total_skipped = 0
    
    for split in splits:
        for subdir in subdirs:
            input_dir = data_root / split / subdir
            output_dir = output_root / split / subdir
            
            if not input_dir.exists():
                print(f"Skipping {input_dir} (not found)")
                continue
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all TIFF files
            tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
            
            if not tiff_files:
                print(f"No TIFF files in {input_dir}")
                continue
            
            print(f"\nProcessing {split}/{subdir} ({len(tiff_files)} files)...")
            
            for tiff_path in tqdm(tiff_files, desc=f"{split}/{subdir}"):
                png_path = output_dir / (tiff_path.stem + ".png")
                
                if png_path.exists() and not force:
                    total_skipped += 1
                    continue
                
                # Read and write
                img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(str(png_path), img)
                total_converted += 1
    
    print("\n" + "=" * 80)
    print(f"Conversion complete!")
    print(f"  Converted: {total_converted}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Output: {output_root}")
    print("=" * 80)


def main() -> None:
    """Main entry point."""
    convert_tiff_to_png(
        data_root="data/raw/AerialImageDataset",
        splits=("train", "test"),
        subdirs=("images", "gt"),
        force=False,
    )


if __name__ == "__main__":
    main()
