# Demo UI Instructions

## Overview

The demo UI is a Streamlit web interface for visualizing segmentation outputs. It loads the trained model, accepts uploaded aerial images, and displays the predicted segmentation mask with class statistics.

**This is a presentation layer only -- it does not affect the core training or inference pipeline.**

---

## Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

The model checkpoint must exist at `outputs/checkpoints/best_model.pth`. If not, run training first:

```bash
python train.py
```

---

## Launch

```bash
streamlit run demo_ui/app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. **Upload image**: Drag and drop or browse for a PNG, JPG, or TIFF aerial image
2. **Preview**: The uploaded image is displayed immediately
3. **Run inference**: Click the "Run Inference" button
4. **View results**: Three panels appear -- Original Image, Predicted Mask, and Overlay
5. **Class statistics**: Per-class pixel percentages are shown with color-coded bars
6. **Download**: Export the segmentation mask as a PNG

---

## Supported Formats

| Format | Extension |
|--------|-----------|
| PNG | `.png` |
| JPEG | `.jpg`, `.jpeg` |
| TIFF | `.tif`, `.tiff` |

Large images (> 2048px on longest side) are automatically downscaled for demo speed.

---

## Architecture

```
demo_ui/
  app.py                 -- Streamlit UI with dark theme
  inference_wrapper.py   -- Model loading + sliding-window inference
  assets/                -- Sample tiles for demonstration
  .streamlit/config.toml -- Server configuration
```

`inference_wrapper.py` imports from `src/models/model_factory.py` and loads the EMA checkpoint. The model is cached in memory after first load.

---

## Classes

| ID | Name | Color |
|----|------|-------|
| 0 | Background | Black |
| 1 | Road | Red |
| 2 | Bridge | Blue |
| 3 | Built-Up Area | Yellow |

---

## Removal

To cleanly remove the demo layer:

```bash
rm -rf demo_ui/
```

Core training (`train.py`) and inference (`test_inference.py`) are unaffected.
