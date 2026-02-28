#!/usr/bin/env python3
"""Dataset structure audit for multi-class segmentation training."""

import os
from pathlib import Path
from collections import defaultdict

import rasterio
import geopandas as gpd
import numpy as np


def audit_dataset():
    """Perform comprehensive dataset audit."""
    
    raz_root = Path("data/Raz")
    
    print("=" * 80)
    print("SECTION 1 — FOLDER OVERVIEW")
    print("=" * 80)
    
    if not raz_root.exists():
        print(f"ERROR: {raz_root} does not exist!")
        return
    
    folders = sorted([d for d in raz_root.iterdir() if d.is_dir()])
    print(f"\nTotal folders in data/RAZ/: {len(folders)}\n")
    
    folder_stats = {}
    for folder in folders:
        tiff_files = list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
        shp_files = list(folder.glob("*.shp"))
        # Also check for nested shp-file folder
        nested_shp = list(folder.glob("shp-file/*.shp"))
        shp_files.extend(nested_shp)
        other_files = [f for f in folder.iterdir() if f.is_file() and not f.suffix in ['.tif', '.tiff', '.shp', '.dbf', '.shx', '.prj', '.cpg']]
        
        print(f"Folder: {folder.name}")
        print(f"  TIFF files: {len(tiff_files)}")
        print(f"  SHP files: {len(shp_files)}")
        print(f"  Other files: {len(other_files)}")
        print(f"  Total items: {len(list(folder.iterdir()))}")
        print()
        
        folder_stats[folder.name] = {
            'tiff_count': len(tiff_files),
            'shp_count': len(shp_files),
            'tiff_files': tiff_files,
            'shp_files': shp_files,
        }
    
    print("\n" + "=" * 80)
    print("SECTION 2 — RASTER ANALYSIS")
    print("=" * 80)
    
    all_tiff_files = []
    tiff_metadata = {}
    
    for folder_name, stats in folder_stats.items():
        tiff_files = stats['tiff_files']
        if len(tiff_files) == 0:
            continue
        
        print(f"\n>>> Folder: {folder_name}\n")
        
        for tiff_path in tiff_files:
            try:
                with rasterio.open(tiff_path) as src:
                    height, width = src.height, src.width
                    band_count = src.count
                    dtype = src.dtypes[0]
                    crs = src.crs
                    transform = src.transform
                    
                    # Read data to get value ranges
                    data = src.read()
                    min_val = float(np.min(data))
                    max_val = float(np.max(data))
                    
                    # Determine if likely mask or ORI
                    is_likely_mask = False
                    if band_count == 1:
                        unique_vals = len(np.unique(data))
                        if unique_vals <= 10 or (min_val >= 0 and max_val <= 255 and dtype in ['uint8', 'int32']):
                            is_likely_mask = True
                    
                    metadata = {
                        'path': str(tiff_path),
                        'folder': folder_name,
                        'width': width,
                        'height': height,
                        'bands': band_count,
                        'dtype': dtype,
                        'crs': str(crs) if crs else 'None',
                        'transform': str(transform),
                        'min_value': min_val,
                        'max_value': max_val,
                        'is_likely_mask': is_likely_mask,
                    }
                    tiff_metadata[tiff_path.name] = metadata
                    all_tiff_files.append(tiff_path.name)
                    
                    # Print formatted output
                    file_type = "MASK" if is_likely_mask else "ORI/UNKNOWN"
                    print(f"File: {tiff_path.name} [{file_type}]")
                    print(f"  Shape: {height} × {width}")
                    print(f"  Bands: {band_count}")
                    print(f"  Dtype: {dtype}")
                    print(f"  CRS: {crs if crs else 'None (UNDEFINED!)'}")
                    print(f"  Pixel Range: [{min_val}, {max_val}]")
                    if band_count == 1:
                        print(f"  Unique Values: {len(np.unique(data))}")
                    print()
                    
            except Exception as e:
                print(f"ERROR reading {tiff_path.name}: {e}\n")
    
    print("\n" + "=" * 80)
    print("SECTION 3 — SHP ANALYSIS")
    print("=" * 80)
    
    shp_metadata = {}
    for folder_name, stats in folder_stats.items():
        shp_files = stats['shp_files']
        if len(shp_files) == 0:
            continue
        
        print(f"\n>>> Folder: {folder_name}\n")
        
        for shp_path in shp_files:
            try:
                gdf = gpd.read_file(shp_path)
                crs = gdf.crs
                geom_types = gdf.geometry.type.unique()
                feature_count = len(gdf)
                columns = list(gdf.columns)
                
                metadata = {
                    'path': str(shp_path),
                    'folder': folder_name,
                    'crs': str(crs) if crs else 'None',
                    'geometry_types': list(geom_types),
                    'feature_count': feature_count,
                    'columns': columns,
                }
                shp_metadata[shp_path.name] = metadata
                
                print(f"File: {shp_path.name}")
                print(f"  CRS: {crs if crs else 'None (UNDEFINED!)'}")
                print(f"  Geometry Types: {', '.join(geom_types)}")
                print(f"  Feature Count: {feature_count}")
                print(f"  Columns: {', '.join(columns)}")
                print()
                
            except Exception as e:
                print(f"ERROR reading {shp_path.name}: {e}\n")
    
    print("\n" + "=" * 80)
    print("SECTION 4 — CRITICAL ISSUES")
    print("=" * 80)
    print()
    
    issues = []
    
    # Check for CRS mismatches
    print("1. CRS Alignment Check:")
    tiff_crs_set = set([m['crs'] for m in tiff_metadata.values() if m['crs'] != 'None'])
    shp_crs_set = set([m['crs'] for m in shp_metadata.values() if m['crs'] != 'None'])
    
    if len(tiff_crs_set) > 1:
        print(f"   WARNING: Multiple TIFF CRS detected: {tiff_crs_set}")
        issues.append("Multiple TIFF CRS")
    elif len(tiff_crs_set) == 0:
        print("   ERROR: No CRS found in TIFFs!")
        issues.append("No TIFF CRS")
    else:
        print(f"   OK: All TIFFs use same CRS: {list(tiff_crs_set)[0]}")
    
    if len(shp_crs_set) > 1:
        print(f"   WARNING: Multiple SHP CRS detected: {shp_crs_set}")
        issues.append("Multiple SHP CRS")
    elif len(shp_crs_set) == 0:
        print("   WARNING: No CRS found in SHPs")
        issues.append("No SHP CRS")
    else:
        print(f"   OK: All SHPs use same CRS: {list(shp_crs_set)[0]}")
    
    if tiff_crs_set and shp_crs_set and tiff_crs_set != shp_crs_set:
        print(f"   ERROR: TIFF CRS {list(tiff_crs_set)[0]} != SHP CRS {list(shp_crs_set)[0]}")
        issues.append("CRS mismatch between TIFF and SHP")
    elif tiff_crs_set and shp_crs_set:
        print("   OK: TIFF and SHP CRS match")
    
    print()
    
    # Check for dimension consistency
    print("2. Raster Dimension Consistency:")
    tiff_shapes = {}
    for name, metadata in tiff_metadata.items():
        shape = (metadata['height'], metadata['width'])
        if shape not in tiff_shapes:
            tiff_shapes[shape] = []
        tiff_shapes[shape].append(name)
    
    if len(tiff_shapes) == 1:
        shape = list(tiff_shapes.keys())[0]
        print(f"   OK: All {len(all_tiff_files)} TIFFs have consistent dimensions: {shape[0]} × {shape[1]}")
    else:
        print(f"   WARNING: {len(tiff_shapes)} different dimensions detected:")
        for shape, files in tiff_shapes.items():
            print(f"      {shape[0]} × {shape[1]}: {len(files)} files")
        issues.append("Inconsistent raster dimensions")
    
    print()
    
    # Check for undefined CRS
    print("3. CRS Definition Check:")
    undefined_crs = [name for name, m in tiff_metadata.items() if m['crs'] == 'None']
    if undefined_crs:
        print(f"   ERROR: {len(undefined_crs)} TIFFs have undefined CRS: {undefined_crs}")
        issues.append("Undefined TIFF CRS")
    else:
        print(f"   OK: All TIFFs have defined CRS")
    
    undefined_shp_crs = [name for name, m in shp_metadata.items() if m['crs'] == 'None']
    if undefined_shp_crs:
        print(f"   WARNING: {len(undefined_shp_crs)} SHPs have undefined CRS: {undefined_shp_crs}")
        issues.append("Undefined SHP CRS")
    else:
        print(f"   OK: All SHPs have defined CRS")
    
    print()
    
    # Check for presence of key components
    print("4. Dataset Component Check:")
    ori_count = sum(1 for m in tiff_metadata.values() if not m['is_likely_mask'])
    mask_count = sum(1 for m in tiff_metadata.values() if m['is_likely_mask'])
    print(f"   Likely ORI images: {ori_count}")
    print(f"   Likely masks: {mask_count}")
    print(f"   SHP files: {len(shp_metadata)}")
    
    if ori_count == 0:
        print("   ERROR: No ORI imagery detected!")
        issues.append("No ORI images found")
    if len(shp_metadata) == 0:
        print("   ERROR: No SHP files found!")
        issues.append("No SHP files found")
    
    print()
    
    print("\n" + "=" * 80)
    print("SECTION 5 — REQUIRED FIXES")
    print("=" * 80)
    print()
    
    if not issues:
        print("No critical issues detected. Dataset appears ready for inspection in training code.")
    else:
        print(f"Found {len(issues)} issue(s):\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print()
        print("FIXES REQUIRED:")
        if "No TIFF CRS" in issues or "Undefined TIFF CRS" in issues:
            print("  → Add CRS information to TIFFs using rasterio.open with 'r+' mode")
        if "CRS mismatch" in issues:
            print("  → Reproject SHP files to match TIFF CRS before rasterization")
        if "No ORI images" in issues:
            print("  → Verify correct folder contains ORI imagery")
        if "No SHP files" in issues:
            print("  → Verify SHP files are in data/RAZ/CG_shp-file/shp_file/")
    
    print()
    print("\n" + "=" * 80)
    print("SECTION 6 — TRAINING READINESS VERDICT")
    print("=" * 80)
    print()
    
    # Final verdict
    can_rasterize_now = len(shp_metadata) > 0 and all(m['crs'] != 'None' for m in shp_metadata.values())
    tiff_ready = all(m['crs'] != 'None' for m in tiff_metadata.values()) and len(tiff_shapes) == 1
    
    print("1. Do we need to rasterize SHP?")
    print(f"   {'YES' if len(shp_metadata) > 0 else 'NO'} — SHP files present: {len(shp_metadata) > 0}")
    
    print("\n2. Are TIFF rasters usable?")
    if tiff_ready:
        print("   YES — All TIFFs have CRS and consistent dimensions")
    else:
        print("   NO — TIFFs have undefined CRS or inconsistent dimensions")
    
    print("\n3. Is CRS alignment safe?")
    if can_rasterize_now or (not undefined_crs and not undefined_shp_crs):
        print("   YES — All files have defined CRS")
    else:
        print("   NO — Some files have undefined CRS")
    
    print("\n4. Is directory structure clean?")
    if len(list(raz_root.glob("*.tif"))) == 0 and len(list(raz_root.glob("*.shp"))) == 0:
        print("   YES — All files in subfolders, no strays in root")
    else:
        print("   NO — Files found in RAZ root directory")
    
    print("\n5. Is dataset ready for training?")
    if not issues:
        print("   YES — All checks passed. Ready to proceed with training code.")
    else:
        print(f"   FIX FIRST — {len(issues)} issue(s) must be resolved first.")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    audit_dataset()
