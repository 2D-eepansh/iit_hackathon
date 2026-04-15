"""
app.py — Demo UI for SVAMITVA AI Infrastructure Mapping System.

Launch:
    cd <project_root>
    streamlit run demo_ui/app.py

This is a VISUALIZATION INTERFACE for AI model outputs — demo purposes only.
Remove /demo_ui to cleanly detach from the core system.
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Resolve project root so imports work regardless of cwd ───────────────────
DEMO_DIR = Path(__file__).parent
ROOT     = DEMO_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DEMO_DIR))

from inference_wrapper import (  # noqa: E402
    CLASS_NAMES,
    CLASS_COLORS,
    ROOFTOP_COLOR,
    classify_rooftops,
    colorize_confidence,
    colorize_with_rooftops,
    create_overlay,
    get_class_stats,
    get_infrastructure_summary,
    model_info,
    predict_image,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Infrastructure Mapping",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Base dark theme ---- */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"] { background: #161b22; }

/* ---- Hide hamburger & footer ---- */
#MainMenu, footer, header { visibility: hidden; }

/* ---- Hero section ---- */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeInDown 0.7s ease both;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #238636 0%, #1a7f37 100%);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    line-height: 1.15;
}
.hero p {
    font-size: 1.05rem;
    color: #8b949e;
    margin: 0;
}

/* ---- Cards ---- */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    animation: fadeIn 0.5s ease both;
}
.card-title {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 0.8rem;
}

/* ---- Stat bar ---- */
.stat-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.6rem;
}
.stat-label {
    color: #c9d1d9;
    font-size: 0.88rem;
    min-width: 110px;
}
.stat-bar-wrap {
    flex: 1;
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.stat-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}
.stat-pct {
    color: #8b949e;
    font-size: 0.82rem;
    min-width: 42px;
    text-align: right;
}

/* ---- Meta chip ---- */
.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.8rem;
    color: #8b949e;
    margin-right: 0.5rem;
    margin-bottom: 0.4rem;
}
.meta-chip span { color: #e6edf3; font-weight: 600; }

/* ---- Image panel ---- */
.img-caption {
    text-align: center;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-top: 0.5rem;
    padding-bottom: 0.2rem;
}

/* ---- Legend ---- */
.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.4rem;
}
.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    flex-shrink: 0;
}
.legend-name {
    font-size: 0.85rem;
    color: #c9d1d9;
}

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploadDropzone"] {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #58a6ff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 16px rgba(31,111,235,0.35) !important;
    width: 100% !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
[data-testid="stImage"] img {
    border-radius: 8px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.5);
}
.stToggle { margin-top: 0.2rem; }

/* ─ Divider ─ */
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def _cached_model_info():
    """Trigger lazy model load and cache the info dict."""
    return model_info()


def _to_display(arr_float_or_uint8) -> np.ndarray:
    """Ensure RGB uint8 for st.image."""
    if arr_float_or_uint8.dtype != np.uint8:
        arr = np.clip(arr_float_or_uint8 * 255, 0, 255).astype(np.uint8)
    else:
        arr = arr_float_or_uint8
    return arr


def _pil_to_rgb(uploaded) -> np.ndarray:
    """Convert any uploaded file to uint8 RGB numpy array."""
    img = Image.open(uploaded).convert("RGB")
    return np.array(img)


# ── Hero section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛰️ &nbsp; Demo Mode &nbsp; — Visualization Interface</div>
    <h1>AI Infrastructure Mapping System</h1>
    <p>Upload an aerial or satellite image &nbsp;→&nbsp; Get semantic segmentation output</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin: 0 0 1.2rem;'>", unsafe_allow_html=True)

# ── Model info (load once) ────────────────────────────────────────────────────
info = _cached_model_info()

st.markdown(f"""
<div class="card" style="animation-delay:.05s">
<div class="card-title">Model Status</div>
<div>
  <span class="meta-chip">Architecture <span>DeepLabV3+ / ResNet-50</span></span>
  <span class="meta-chip">Checkpoint <span>{info['checkpoint']}</span></span>
  <span class="meta-chip">Epoch <span>{info['epoch']}</span></span>
  <span class="meta-chip">Best mIoU <span>{info['best_miou']:.4f}</span></span>
  <span class="meta-chip">Device <span>{info['device'].upper()}</span></span>
  <span class="meta-chip">Ensemble <span>{'ON (ep43+ep80)' if info.get('ensemble') else 'OFF'}</span></span>
  <span class="meta-chip">Bias Tuned <span>{'YES' if info.get('bias_tuned') else 'DEFAULT'}</span></span>
</div>
<div style="margin-top:0.6rem; font-size:0.78rem; color:#8b949e;">
  Pipeline: {info.get('pipeline', 'Standard')}
</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="card" style="animation-delay:.1s"><div class="card-title">Upload Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="Drag & drop or click to browse",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    label_visibility="visible",
)

st.markdown("</div>", unsafe_allow_html=True)

# ── Main workflow ─────────────────────────────────────────────────────────────

if uploaded is not None:
    # Convert to numpy
    image_rgb = _pil_to_rgb(uploaded)
    h_orig, w_orig = image_rgb.shape[:2]

    # Downscale very large images for demo speed (keep aspect ratio)
    MAX_DIM = 2048
    if max(h_orig, w_orig) > MAX_DIM:
        scale  = MAX_DIM / max(h_orig, w_orig)
        nh, nw = int(h_orig * scale), int(w_orig * scale)
        img_pil   = Image.fromarray(image_rgb).resize((nw, nh), Image.LANCZOS)
        image_rgb = np.array(img_pil)
        h_orig, w_orig = nh, nw

    # Preview
    st.markdown('<div class="card" style="animation-delay:.15s"><div class="card-title">Preview</div>', unsafe_allow_html=True)
    prev_col, _ = st.columns([2, 1])
    with prev_col:
        st.image(image_rgb, caption="Uploaded image", use_container_width=True)
        st.markdown(f'<p class="img-caption">{uploaded.name} &nbsp;·&nbsp; {w_orig}×{h_orig}px</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Run inference
    run_col, opt_col, _ = st.columns([1, 1, 1])
    with run_col:
        run = st.button("Run Inference", use_container_width=True)
    with opt_col:
        use_tta = st.toggle("TTA (3x slower, +IoU)", value=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "result" not in st.session_state:
        st.session_state.result = None
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    # Reset result on new upload
    if st.session_state.last_file != uploaded.name:
        st.session_state.result    = None
        st.session_state.last_file = uploaded.name

    if run:
        with st.spinner("Running calibrated ensemble inference…"):
            t0           = time.time()
            mask, conf_map = predict_image(image_rgb, use_tta=use_tta, return_confidence=True)
            rooftop_mask = classify_rooftops(mask)
            elapsed      = time.time() - t0

        st.session_state.result = {
            "mask":         mask,
            "rooftop_mask": rooftop_mask,
            "conf_map":     conf_map,
            "elapsed":      elapsed,
        }

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.result is not None:
        mask         = st.session_state.result["mask"]
        rooftop_mask = st.session_state.result.get("rooftop_mask")
        conf_map     = st.session_state.result.get("conf_map")
        elapsed      = st.session_state.result["elapsed"]
        stats        = get_class_stats(mask, rooftop_mask)

        colored  = colorize_with_rooftops(mask, rooftop_mask)
        overlay  = _to_display(create_overlay(image_rgb, mask, rooftop_mask=rooftop_mask))

        # ── Output panels ────────────────────────────────────────────────────
        st.markdown('<div class="card" style="animation-delay:.2s"><div class="card-title">Segmentation Output</div>', unsafe_allow_html=True)

        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            show_overlay = st.toggle("Show overlay blend", value=True)
        with col_opts2:
            show_conf = st.toggle("Show confidence map", value=False)

        n_panels = 2 + int(show_overlay) + int(show_conf and conf_map is not None)
        cols_out = st.columns(n_panels)

        panel_idx = 0
        with cols_out[panel_idx]:
            st.image(image_rgb, use_container_width=True)
            st.markdown('<p class="img-caption">Original Image</p>', unsafe_allow_html=True)
        panel_idx += 1

        with cols_out[panel_idx]:
            st.image(colored, use_container_width=True)
            st.markdown('<p class="img-caption">Predicted Mask</p>', unsafe_allow_html=True)
        panel_idx += 1

        if show_overlay:
            with cols_out[panel_idx]:
                st.image(overlay, use_container_width=True)
                st.markdown('<p class="img-caption">Overlay (a=0.45)</p>', unsafe_allow_html=True)
            panel_idx += 1

        if show_conf and conf_map is not None:
            conf_rgb = colorize_confidence(conf_map)
            high_conf_pct = float((conf_map > 0.7).mean() * 100)
            with cols_out[panel_idx]:
                st.image(conf_rgb, use_container_width=True)
                st.markdown(
                    f'<p class="img-caption">Confidence ({high_conf_pct:.0f}% high)</p>',
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Stats + Legend ────────────────────────────────────────────────────
        left, right = st.columns([3, 1])

        with left:
            st.markdown('<div class="card" style="animation-delay:.25s"><div class="card-title">Class Distribution</div>', unsafe_allow_html=True)

            rows_html = ""
            for cls_name, s in stats.items():
                hex_col = "#{:02x}{:02x}{:02x}".format(*s["color"])
                bar_w   = max(0.5, s["pct"])
                rows_html += f"""
                <div class="stat-row">
                  <div class="stat-label">{cls_name}</div>
                  <div class="stat-bar-wrap">
                    <div class="stat-bar-fill" style="width:{bar_w:.1f}%;background:{hex_col}"></div>
                  </div>
                  <div class="stat-pct">{s['pct']:.1f}%</div>
                </div>"""

            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card" style="animation-delay:.3s"><div class="card-title">Legend</div>', unsafe_allow_html=True)

            legend_html = ""
            for cls_name, s in stats.items():
                hex_col = "#{:02x}{:02x}{:02x}".format(*s["color"])
                border  = " border:1px solid #555;" if s["color"] == [0, 0, 0] else ""
                legend_html += f"""
                <div class="legend-row">
                  <div class="legend-dot" style="background:{hex_col};{border}"></div>
                  <div class="legend-name">{cls_name}</div>
                </div>"""

            st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Village Survey Statistics ─────────────────────────────────────────
        infra = get_infrastructure_summary(mask, rooftop_mask if rooftop_mask is not None
                                           else np.zeros_like(mask, dtype=bool))
        n_classes_detected = sum(1 for s in stats.values() if s["pct"] > 0.1)

        # Physical unit estimate (assuming 0.3m/px for demo images)
        PIXEL_SIZE_M = 0.3
        px_area = PIXEL_SIZE_M * PIXEL_SIZE_M
        road_px     = int((mask == 1).sum())
        water_px    = int((mask == 4).sum())
        bu_px       = int((mask == 3).sum())
        bridge_px   = int((mask == 2).sum())
        total_area_ha = round(mask.size * px_area / 10000.0, 2)
        road_len_m    = round(road_px * PIXEL_SIZE_M, 0)
        water_area_m2 = round(water_px * px_area, 0)
        bu_area_m2    = round(bu_px * px_area, 0)
        conf_pct = round(float((conf_map > 0.7).mean() * 100), 1) if conf_map is not None else 0.0
        review_pct = round(100.0 - conf_pct, 1)

        st.markdown(f"""
        <div class="card" style="animation-delay:.35s">
        <div class="card-title">Village Infrastructure Assessment</div>

        <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:1rem;">
          <span class="meta-chip">Inference time <span>{elapsed:.2f}s</span></span>
          <span class="meta-chip">Total area <span>~{total_area_ha} ha</span></span>
          <span class="meta-chip">Classes detected <span>{n_classes_detected}</span></span>
          <span class="meta-chip">Automated coverage <span>{conf_pct:.0f}%</span></span>
          <span class="meta-chip">Needs review <span>{review_pct:.0f}%</span></span>
        </div>

        <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:0.8rem;">

          <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:0.9rem;">
            <div style="color:#ff5555;font-size:0.75rem;font-weight:700;letter-spacing:.08em;margin-bottom:.4rem;">ROAD NETWORK</div>
            <div style="font-size:1.3rem;font-weight:800;color:#e6edf3;">~{road_len_m:,.0f} m</div>
            <div style="font-size:0.8rem;color:#8b949e;margin-top:.2rem;">approx road length</div>
            <div style="font-size:0.82rem;color:#c9d1d9;margin-top:.3rem;">{infra['road_pct']:.1f}% area coverage</div>
          </div>

          <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:0.9rem;">
            <div style="color:#ffff00;font-size:0.75rem;font-weight:700;letter-spacing:.08em;margin-bottom:.4rem;">BUILT-UP AREA</div>
            <div style="font-size:1.3rem;font-weight:800;color:#e6edf3;">{infra['n_buildings']} bldgs</div>
            <div style="font-size:0.8rem;color:#8b949e;margin-top:.2rem;">~{bu_area_m2:,.0f} m2 total</div>
            <div style="font-size:0.82rem;color:#c9d1d9;margin-top:.3rem;">~{infra['n_rooftops']} probable rooftops</div>
          </div>

          <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:0.9rem;">
            <div style="color:#00c8ff;font-size:0.75rem;font-weight:700;letter-spacing:.08em;margin-bottom:.4rem;">WATER BODIES</div>
            <div style="font-size:1.3rem;font-weight:800;color:#e6edf3;">{infra['n_water_bodies']} bodies</div>
            <div style="font-size:0.8rem;color:#8b949e;margin-top:.2rem;">~{water_area_m2:,.0f} m2 total</div>
            <div style="font-size:0.82rem;color:#c9d1d9;margin-top:.3rem;">{infra.get('water_pct', 0.0):.1f}% area coverage</div>
          </div>

          <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:0.9rem;">
            <div style="color:#0055ff;font-size:0.75rem;font-weight:700;letter-spacing:.08em;margin-bottom:.4rem;">BRIDGES</div>
            <div style="font-size:1.3rem;font-weight:800;color:#e6edf3;">{infra['bridge_px']} px</div>
            <div style="font-size:0.8rem;color:#8b949e;margin-top:.2rem;">detected bridge area</div>
            <div style="font-size:0.82rem;color:#c9d1d9;margin-top:.3rem;">spatial recovery active</div>
          </div>

        </div>
        </div>
        """, unsafe_allow_html=True)

        # Download mask
        mask_pil = Image.fromarray(colored)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇  Download Segmentation Mask",
            data=buf.getvalue(),
            file_name=f"{Path(uploaded.name).stem}_mask.png",
            mime="image/png",
        )

else:
    # Placeholder when no file uploaded
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem 2rem; color:#8b949e; animation-delay:.15s">
        <div style="font-size:3.5rem; margin-bottom:1rem;">🛰️</div>
        <div style="font-size:1.1rem; font-weight:600; color:#c9d1d9; margin-bottom:0.5rem;">
            No image uploaded yet
        </div>
        <div style="font-size:0.9rem;">
            Upload a PNG, JPG, or TIFF aerial image above to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.78rem; margin-top:2rem; padding:1rem">
    This is a visualization interface for AI model outputs.
    Demo purposes only — not part of core training/inference pipeline.
    &nbsp;·&nbsp; SVAMITVA Multi-Class Segmentation
</div>
""", unsafe_allow_html=True)
