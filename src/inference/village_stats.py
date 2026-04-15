"""
Village Infrastructure Statistics — SVAMITVA Survey Report Generator.

Converts segmentation mask + geospatial metadata into a structured
village assessment report with physical units (meters, sq meters, counts).

Usage:
    from src.inference.village_stats import VillageReport
    report = VillageReport.from_mask(mask, pixel_size_m=0.3, village_name="Nadala")
    print(report.summary())
    report.save_json("outputs/village_report.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Class IDs ─────────────────────────────────────────────────────────────────
_BG      = 0
_ROAD    = 1
_BRIDGE  = 2
_BUILTUP = 3
_WATER   = 4


@dataclass
class ClassStats:
    pixel_count:  int   = 0
    area_m2:      float = 0.0
    coverage_pct: float = 0.0
    n_components: int   = 0


@dataclass
class RoadStats(ClassStats):
    total_length_m:  float = 0.0   # Approximate: perimeter / 2 of skeleton
    avg_width_m:     float = 0.0   # area / skeleton_length
    n_segments:      int   = 0


@dataclass
class BuildingStats(ClassStats):
    n_buildings:            int   = 0
    avg_building_area_m2:   float = 0.0
    building_density_per_ha: float = 0.0
    rooftop_count:          int   = 0


@dataclass
class WaterStats(ClassStats):
    n_water_bodies:  int = 0
    largest_body_m2: float = 0.0


@dataclass
class BridgeStats(ClassStats):
    n_bridges:          int   = 0
    bridge_locations:   list  = field(default_factory=list)  # [(cx, cy), ...]


@dataclass
class VillageReport:
    village_name:     str
    timestamp:        str
    pixel_size_m:     float
    total_area_m2:    float
    total_area_ha:    float

    road:    RoadStats    = field(default_factory=RoadStats)
    builtup: BuildingStats = field(default_factory=BuildingStats)
    water:   WaterStats   = field(default_factory=WaterStats)
    bridge:  BridgeStats  = field(default_factory=BridgeStats)

    processing_time_s: float = 0.0
    confidence_coverage_pct: float = 0.0   # % pixels with pred conf > 0.7

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        pixel_size_m: float = 0.3,
        village_name: str = "Unknown Village",
        confidence_map: Optional[np.ndarray] = None,
    ) -> "VillageReport":
        t0 = time.time()
        H, W = mask.shape
        total_px  = H * W
        px_area   = pixel_size_m * pixel_size_m
        total_m2  = total_px * px_area

        def pct(px): return round(100.0 * px / max(total_px, 1), 2)
        def m2(px):  return round(px * px_area, 1)

        # ── Road ──────────────────────────────────────────────────────────────
        road_bin = (mask == _ROAD).astype(np.uint8)
        road_px  = int(road_bin.sum())
        road_stats = RoadStats(
            pixel_count=road_px,
            area_m2=m2(road_px),
            coverage_pct=pct(road_px),
        )
        if road_px > 0:
            # Approximate road length: skeletonise and count skeleton pixels
            try:
                from skimage.morphology import skeletonize
                skel = skeletonize(road_bin.astype(bool))
                skel_px = int(skel.sum())
                road_stats.total_length_m = round(skel_px * pixel_size_m, 1)
                # Average width = area / length
                if skel_px > 0:
                    road_stats.avg_width_m = round(road_px * pixel_size_m / skel_px, 2)
            except ImportError:
                # Fallback: estimate from area
                road_stats.total_length_m = round(road_px * pixel_size_m / 3.0, 1)

            n_lbl, _, st, _ = cv2.connectedComponentsWithStats(road_bin, connectivity=8)
            road_stats.n_components = max(0, n_lbl - 1)
            road_stats.n_segments   = sum(
                1 for i in range(1, n_lbl) if st[i, cv2.CC_STAT_AREA] >= 100
            )

        # ── Built-Up / Buildings ───────────────────────────────────────────────
        bu_bin = (mask == _BUILTUP).astype(np.uint8)
        bu_px  = int(bu_bin.sum())
        n_lbl_bu, labels_bu, st_bu, _ = cv2.connectedComponentsWithStats(bu_bin, connectivity=8)
        valid_buildings = [
            i for i in range(1, n_lbl_bu)
            if 50 <= st_bu[i, cv2.CC_STAT_AREA] <= 50000
        ]
        n_buildings   = len(valid_buildings)
        avg_build_m2  = (
            round(np.mean([st_bu[i, cv2.CC_STAT_AREA] for i in valid_buildings]) * px_area, 1)
            if valid_buildings else 0.0
        )

        # Rooftop count: buildings area 100-8000px, aspect < 6, solidity > 0.4
        rooftop_count = 0
        for i in valid_buildings:
            area = st_bu[i, cv2.CC_STAT_AREA]
            if area < 100 or area > 8000:
                continue
            bw = max(st_bu[i, cv2.CC_STAT_WIDTH],  1)
            bh = max(st_bu[i, cv2.CC_STAT_HEIGHT], 1)
            if max(bw, bh) / min(bw, bh) > 6.0:
                continue
            if area / (bw * bh) < 0.40:
                continue
            rooftop_count += 1

        # Building density per hectare
        total_ha       = total_m2 / 10000.0
        build_density  = round(n_buildings / max(total_ha, 0.01), 1)

        builtup_stats = BuildingStats(
            pixel_count=bu_px,
            area_m2=m2(bu_px),
            coverage_pct=pct(bu_px),
            n_components=max(0, n_lbl_bu - 1),
            n_buildings=n_buildings,
            avg_building_area_m2=avg_build_m2,
            building_density_per_ha=build_density,
            rooftop_count=rooftop_count,
        )

        # ── Water ──────────────────────────────────────────────────────────────
        water_bin = (mask == _WATER).astype(np.uint8)
        water_px  = int(water_bin.sum())
        n_lbl_w, _, st_w, _ = cv2.connectedComponentsWithStats(water_bin, connectivity=8)
        valid_water  = [i for i in range(1, n_lbl_w) if st_w[i, cv2.CC_STAT_AREA] >= 200]
        largest_body = (
            round(max(st_w[i, cv2.CC_STAT_AREA] for i in valid_water) * px_area, 1)
            if valid_water else 0.0
        )
        water_stats = WaterStats(
            pixel_count=water_px,
            area_m2=m2(water_px),
            coverage_pct=pct(water_px),
            n_components=max(0, n_lbl_w - 1),
            n_water_bodies=len(valid_water),
            largest_body_m2=largest_body,
        )

        # ── Bridge ─────────────────────────────────────────────────────────────
        bridge_bin = (mask == _BRIDGE).astype(np.uint8)
        bridge_px  = int(bridge_bin.sum())
        bridge_locs: list = []
        n_lbl_br, labels_br, st_br, centroids_br = cv2.connectedComponentsWithStats(
            bridge_bin, connectivity=8
        )
        valid_bridges = [i for i in range(1, n_lbl_br) if st_br[i, cv2.CC_STAT_AREA] >= 30]
        if valid_bridges:
            bridge_locs = [
                {"x_px": int(centroids_br[i][0]), "y_px": int(centroids_br[i][1]),
                 "area_m2": round(st_br[i, cv2.CC_STAT_AREA] * px_area, 1)}
                for i in valid_bridges
            ]
        bridge_stats = BridgeStats(
            pixel_count=bridge_px,
            area_m2=m2(bridge_px),
            coverage_pct=pct(bridge_px),
            n_components=len(valid_bridges),
            n_bridges=len(valid_bridges),
            bridge_locations=bridge_locs,
        )

        # ── Confidence coverage ────────────────────────────────────────────────
        conf_pct = 0.0
        if confidence_map is not None:
            conf_pct = round(float((confidence_map > 0.7).mean() * 100), 1)

        report = cls(
            village_name=village_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            pixel_size_m=pixel_size_m,
            total_area_m2=round(total_m2, 1),
            total_area_ha=round(total_m2 / 10000.0, 2),
            road=road_stats,
            builtup=builtup_stats,
            water=water_stats,
            bridge=bridge_stats,
            processing_time_s=round(time.time() - t0, 2),
            confidence_coverage_pct=conf_pct,
        )
        return report

    def summary(self) -> str:
        """Formatted human-readable summary card."""
        sep = "=" * 60
        lines = [
            sep,
            f"  SVAMITVA VILLAGE SURVEY REPORT",
            f"  Village: {self.village_name}",
            f"  Generated: {self.timestamp}",
            sep,
            f"  Total Area: {self.total_area_ha:.2f} ha ({self.total_area_m2:,.0f} m2)",
            f"",
            f"  ROAD NETWORK",
            f"    Total length:   ~{self.road.total_length_m:,.0f} m",
            f"    Coverage:        {self.road.coverage_pct:.1f}% of area",
            f"    Road segments:   {self.road.n_segments}",
            f"",
            f"  BUILT-UP AREA",
            f"    Area:            {self.builtup.area_m2:,.0f} m2 ({self.builtup.coverage_pct:.1f}%)",
            f"    Buildings:       {self.builtup.n_buildings}",
            f"    Probable rooftops: {self.builtup.rooftop_count}",
            f"    Density:         {self.builtup.building_density_per_ha:.1f} buildings/ha",
            f"",
            f"  WATER BODIES",
            f"    Total area:      {self.water.area_m2:,.0f} m2 ({self.water.coverage_pct:.1f}%)",
            f"    Water bodies:    {self.water.n_water_bodies}",
            f"",
            f"  BRIDGES",
            f"    Detected:        {self.bridge.n_bridges}",
            f"    Total area:      {self.bridge.area_m2:,.0f} m2",
        ]
        if self.bridge.bridge_locations:
            for i, loc in enumerate(self.bridge.bridge_locations[:5]):
                lines.append(f"      Bridge {i+1}: ({loc['x_px']}, {loc['y_px']}) "
                             f"~{loc['area_m2']:.0f} m2")
        lines += [
            f"",
            f"  SYSTEM PERFORMANCE",
            f"    Automated coverage: {self.confidence_coverage_pct:.1f}% high-confidence",
            f"    Requires review:    {100-self.confidence_coverage_pct:.1f}% of pixels",
            f"    Processing time:    {self.processing_time_s:.2f}s",
            sep,
        ]
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        def _serialise(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialise(v) for k, v in asdict(obj).items()}
            return obj

        with open(out, "w") as f:
            json.dump(_serialise(self), f, indent=2)
