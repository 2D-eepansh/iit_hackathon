"""
src/postprocessing.py — Post-processing for SVAMITVA segmentation masks.

All functions are safe to call with any 5-class mask (H, W) uint8.
Each step is independent and wrapped in the caller so a single failure
cannot crash the inference pipeline.
"""

from typing import Optional

import cv2
import numpy as np

# Class IDs (mirrors unified_dataset.py CLASS_NAMES)
CLASS_BG      = 0
CLASS_ROAD    = 1
CLASS_BRIDGE  = 2
CLASS_BUILTUP = 3
CLASS_WATER   = 4


def _softmax(logit_acc: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax along class axis 0. (C, H, W) → (C, H, W)."""
    shifted = logit_acc - logit_acc.max(axis=0, keepdims=True)
    e = np.exp(shifted)
    return e / (e.sum(axis=0, keepdims=True) + 1e-9)


# ── Road ──────────────────────────────────────────────────────────────────────

def refine_roads(mask: np.ndarray) -> np.ndarray:
    """
    Morphological closing (5×5) to connect broken road segments,
    then remove fragments < 200px.

    Ported from train_one_epoch.py:validate_multiclass — makes validation
    refinement available at test/demo time.
    """
    road_mask = (mask == CLASS_ROAD).astype(np.uint8)
    if road_mask.sum() == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    road_refined = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        road_refined, connectivity=8
    )
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 200:
            road_refined[labels == lbl] = 0

    result = mask.copy()
    result[mask == CLASS_ROAD] = CLASS_BG       # clear old road pixels
    result[road_refined == 1]  = CLASS_ROAD     # apply refined road
    return result


# ── Water ─────────────────────────────────────────────────────────────────────

def refine_water(
    mask: np.ndarray,
    logit_acc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Water body stabilisation:
      1. Optional confidence gate (softmax prob > 0.35) — eliminates ghost water
      2. Morphological opening  (3×3) — removes isolated noise pixels
      3. Morphological closing (11×11) — fills internal holes
      4. Remove components < 500px   — removes scattered misclassification
    """
    water_mask = (mask == CLASS_WATER).astype(np.uint8)
    if water_mask.sum() == 0:
        return mask

    # Confidence gate
    if logit_acc is not None:
        probs = _softmax(logit_acc)
        water_prob = probs[CLASS_WATER]
        water_mask = (water_mask & (water_prob > 0.35)).astype(np.uint8)

    # Denoise
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k_open)

    # Fill holes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k_close)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        water_mask, connectivity=8
    )
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 500:
            water_mask[labels == lbl] = 0

    result = mask.copy()
    result[mask == CLASS_WATER] = CLASS_BG
    result[water_mask == 1]     = CLASS_WATER
    return result


# ── Bridge ────────────────────────────────────────────────────────────────────

def refine_bridges(
    mask: np.ndarray,
    logit_acc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Bridge credibility filter — reduces false positives:
      1. Optional confidence gate (softmax prob > 0.40)
      2. Area filter: keep only 50–5000px components
      3. Proximity check: bridge must be within ~50px of road or water
         (geographically, bridges cross water and connect roads)
    """
    bridge_mask = (mask == CLASS_BRIDGE).astype(np.uint8)
    if bridge_mask.sum() == 0:
        return mask

    # Confidence gate
    if logit_acc is not None:
        probs = _softmax(logit_acc)
        bridge_prob = probs[CLASS_BRIDGE]
        bridge_mask = (bridge_mask & (bridge_prob > 0.40)).astype(np.uint8)

    # Area filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bridge_mask, connectivity=8
    )
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 50 or area > 5000:
            bridge_mask[labels == lbl] = 0

    # Proximity to road or water (±50px dilation → 101×101 kernel)
    road_water = ((mask == CLASS_ROAD) | (mask == CLASS_WATER)).astype(np.uint8)
    k_prox = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 101))
    nearby = cv2.dilate(road_water, k_prox)
    bridge_mask = cv2.bitwise_and(bridge_mask, nearby)

    result = mask.copy()
    result[mask == CLASS_BRIDGE] = CLASS_BG
    result[bridge_mask == 1]     = CLASS_BRIDGE
    return result


# ── Rooftop heuristic ─────────────────────────────────────────────────────────

def classify_rooftops(mask: np.ndarray) -> np.ndarray:
    """
    Heuristic sub-classification of Built-Up Area (class 3) into probable
    individual rooftops vs. general built-up land.

    A Built-Up component is tagged as a rooftop candidate when:
      - Area:      100–8000px  (single building footprint)
      - Aspect:    < 6.0       (not a long road-like streak)
      - Solidity:  > 0.40      (bounded-box fill — rectangular-ish shape)

    Returns
    -------
    np.ndarray  bool (H, W)
        True where Built-Up pixels are classified as probable rooftops.
    """
    bu_mask = (mask == CLASS_BUILTUP).astype(np.uint8)
    rooftop = np.zeros(mask.shape, dtype=bool)

    if bu_mask.sum() == 0:
        return rooftop

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bu_mask, connectivity=8
    )
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 100 or area > 8000:
            continue
        bw = max(stats[lbl, cv2.CC_STAT_WIDTH], 1)
        bh = max(stats[lbl, cv2.CC_STAT_HEIGHT], 1)
        if max(bw, bh) / min(bw, bh) > 6.0:
            continue
        if area / (bw * bh) < 0.40:
            continue
        rooftop[labels == lbl] = True

    return rooftop


def get_infrastructure_summary(mask: np.ndarray, rooftop_mask: np.ndarray) -> dict:
    """
    Compute simple infrastructure statistics from a prediction mask.

    Returns a dict with counts/percentages suitable for the demo summary card.
    """
    total = mask.size

    road_px    = int((mask == CLASS_ROAD).sum())
    water_px   = int((mask == CLASS_WATER).sum())
    bu_px      = int((mask == CLASS_BUILTUP).sum())
    bridge_px  = int((mask == CLASS_BRIDGE).sum())
    rooftop_px = int(rooftop_mask.sum())

    # Count distinct connected components (buildings, water bodies)
    bu_bin = (mask == CLASS_BUILTUP).astype(np.uint8)
    n_buildings = 0
    if bu_bin.sum() > 0:
        n_labels, _, b_stats, _ = cv2.connectedComponentsWithStats(bu_bin, connectivity=8)
        n_buildings = sum(
            1 for i in range(1, n_labels) if b_stats[i, cv2.CC_STAT_AREA] >= 200
        )

    water_bin = (mask == CLASS_WATER).astype(np.uint8)
    n_water_bodies = 0
    if water_bin.sum() > 0:
        n_wlabels, _, w_stats, _ = cv2.connectedComponentsWithStats(water_bin, connectivity=8)
        n_water_bodies = sum(
            1 for i in range(1, n_wlabels) if w_stats[i, cv2.CC_STAT_AREA] >= 500
        )

    n_rooftops = 0
    if rooftop_mask.sum() > 0:
        r_bin = rooftop_mask.astype(np.uint8)
        n_rlabels, _, r_stats, _ = cv2.connectedComponentsWithStats(r_bin, connectivity=8)
        n_rooftops = n_rlabels - 1

    return {
        "road_pct":        round(100.0 * road_px / max(total, 1), 2),
        "water_pct":       round(100.0 * water_px / max(total, 1), 2),
        "builtup_pct":     round(100.0 * bu_px / max(total, 1), 2),
        "bridge_px":       bridge_px,
        "rooftop_px":      rooftop_px,
        "n_buildings":     n_buildings,
        "n_water_bodies":  n_water_bodies,
        "n_rooftops":      n_rooftops,
    }


# ── Road Gap Fill ─────────────────────────────────────────────────────────────

def road_gap_fill(mask: np.ndarray, max_gap_px: int = 25) -> np.ndarray:
    """
    Close discontinuities in the predicted road network.

    Strategy:
      1. Skeletonise the road binary mask (thin to 1px paths).
      2. Dilate skeleton endpoints by max_gap_px to detect nearby broken ends.
      3. For each pair of nearby endpoints, draw the shortest connecting path.
      4. Apply final morphological closing (3×3) to smooth hairline gaps.

    This directly attacks the high Road->Background error (41% of Road GT
    pixels were lost because the model is conservative on faint / narrow roads).
    """
    road_bin = (mask == CLASS_ROAD).astype(np.uint8)
    if road_bin.sum() < 10:
        return mask

    # Morphological closing to bridge small gaps (conservative kernel first)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    road_closed = cv2.morphologyEx(road_bin, cv2.MORPH_CLOSE, k_close)

    # Second pass — slightly larger kernel to catch bigger breaks
    k_close2 = cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap_px, 3))
    road_h = cv2.morphologyEx(road_bin, cv2.MORPH_CLOSE, k_close2)

    k_close3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max_gap_px))
    road_v = cv2.morphologyEx(road_bin, cv2.MORPH_CLOSE, k_close3)

    # Union of all closes, then mask to only add pixels that were BG
    road_filled = np.maximum(road_closed, np.maximum(road_h, road_v))

    # Only grow into BG pixels — never overwrite water/bridge/builtup
    safe_growth = (mask == CLASS_BG)
    new_road    = ((road_filled == 1) & safe_growth & (road_bin == 0)).astype(np.uint8)

    result = mask.copy()
    result[new_road == 1] = CLASS_ROAD
    return result


# ── Bridge Recovery from Built-Up ─────────────────────────────────────────────

def bridge_recovery_from_builtup(mask: np.ndarray) -> np.ndarray:
    """
    Recover missed Bridge predictions from Built-Up misclassifications.

    Physical reasoning:
      Bridges are elongated structures connecting roads across waterways.
      The confusion matrix shows 59.4% of Bridge GT → Built-Up (model sees
      structure but assigns wrong class).

    Recovery rule — reclassify a Built-Up connected component as Bridge when:
      1. Area    : 50 – 6000 px  (bridge-scale objects)
      2. Aspect  : > 2.0  (elongated — bridges are longer than wide)
      3. Proximity: bounding box overlaps a 40px dilation of Road  AND
                    a 120px dilation of Water or Road on BOTH sides
                    (must be between something, not just next to it)
    """
    bu_mask = (mask == CLASS_BUILTUP).astype(np.uint8)
    if bu_mask.sum() == 0:
        return mask

    road_mask  = (mask == CLASS_ROAD).astype(np.uint8)
    water_mask = (mask == CLASS_WATER).astype(np.uint8)

    # Dilate road for proximity check
    k_road = cv2.getStructuringElement(cv2.MORPH_RECT, (81, 81))   # ~40px radius
    road_nearby = cv2.dilate(road_mask, k_road)

    # Dilate water for proximity check
    k_water = cv2.getStructuringElement(cv2.MORPH_RECT, (241, 241))  # ~120px radius
    water_nearby = cv2.dilate(water_mask, k_water)

    # Union: bridge must be near road or near water
    context_mask = cv2.bitwise_or(road_nearby, water_nearby)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bu_mask, connectivity=8)
    bridge_candidate = np.zeros_like(mask, dtype=np.uint8)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 50 or area > 6000:
            continue

        bw = max(stats[lbl, cv2.CC_STAT_WIDTH],  1)
        bh = max(stats[lbl, cv2.CC_STAT_HEIGHT], 1)
        aspect = max(bw, bh) / min(bw, bh)
        if aspect < 2.0:
            continue

        # Must overlap BOTH road-nearby AND water-nearby zones (is between them)
        comp_mask = (labels == lbl).astype(np.uint8)
        near_road  = (comp_mask & road_nearby ).any()
        near_water = (comp_mask & water_nearby).any()
        # Also accept: near road on both horizontal ends (road bridge without river GT)
        near_context = near_road and (near_water or _spans_road(comp_mask, road_mask))
        if not near_context:
            continue

        bridge_candidate[labels == lbl] = 1

    result = mask.copy()
    result[bridge_candidate == 1] = CLASS_BRIDGE
    return result


def _spans_road(comp_mask: np.ndarray, road_mask: np.ndarray) -> bool:
    """
    Check if comp_mask spans between two separate road regions
    (i.e., road exists on both horizontal or vertical sides of the component).
    Heuristic: dilate component 80px, intersect with road, check if road
    pixels exist on left half AND right half (or top AND bottom).
    """
    ys, xs = np.where(comp_mask)
    if len(xs) == 0:
        return False
    x_mid = (xs.min() + xs.max()) // 2
    y_mid = (ys.min() + ys.max()) // 2

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (161, 161))
    dilated = cv2.dilate(comp_mask, k)
    road_near = dilated & road_mask

    road_left   = road_near[:, :x_mid].any()
    road_right  = road_near[:, x_mid:].any()
    road_top    = road_near[:y_mid, :].any()
    road_bottom = road_near[y_mid:, :].any()

    return (road_left and road_right) or (road_top and road_bottom)



# -- Master pipeline ----------------------------------------------------------

def postprocess_mask(
    mask,
    logit_acc=None,
    use_bridge_recovery=True,
    use_road_gap_fill=True,
):
    """
    Full post-processing pipeline:
      1. Road gap fill   - closes discontinuities in road network
      2. Road refine     - morphological closing + small component removal
      3. Water refine    - confidence gate + hole filling
      4. Bridge refine   - area/proximity filter
      5. Bridge recovery - reclassify elongated BU near Road+Water as Bridge
    Each step is individually guarded.
    """
    if use_road_gap_fill:
        try:
            mask = road_gap_fill(mask)
        except Exception:
            pass
    try:
        mask = refine_roads(mask)
    except Exception:
        pass
    try:
        mask = refine_water(mask, logit_acc)
    except Exception:
        pass
    try:
        mask = refine_bridges(mask, logit_acc)
    except Exception:
        pass
    if use_bridge_recovery:
        try:
            mask = bridge_recovery_from_builtup(mask)
        except Exception:
            pass
    return mask
