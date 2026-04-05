# Demo UI — AI Infrastructure Mapping System

Lightweight Streamlit interface for visualizing SVAMITVA segmentation outputs.

**This is a demo layer only — isolated from the core ML pipeline.**
Delete this directory entirely to remove it without affecting training or inference.

---

## Quick Start

```bash
# From project root
cd /home/dk/ml_projects/iit_hackathon
venv/bin/streamlit run demo_ui/app.py
```

Open **http://localhost:8501** in your browser.

---

## Features

| Feature | Details |
|---|---|
| File upload | PNG / JPG / JPEG / TIFF (drag & drop or browse) |
| Live preview | Shown immediately after upload |
| Inference | Sliding-window (512px patches, 64px overlap) |
| Results | Original · Mask · Overlay side-by-side |
| Overlay toggle | Blend mask over image at α=0.45 |
| Class stats | Per-class pixel % with color bars |
| Download | Export segmentation mask as PNG |
| Model info | Checkpoint name, epoch, best mIoU, device |

---

## Architecture

```
demo_ui/
├── app.py               — Streamlit UI (dark theme, animations)
├── inference_wrapper.py — Model loading + sliding-window inference
├── assets/              — Static assets (if any)
└── README.md            — This file
```

`inference_wrapper.py` imports from:
- `src/models/model_factory.py` → `create_model()`
- `outputs/checkpoints/best_model.pth` → EMA weights (epoch 39, mIoU 0.5880)

**No core files are modified.**

---

## Classes

| ID | Name | Color |
|---|---|---|
| 0 | Background | Black `#000000` |
| 1 | Road | Red `#ff0000` |
| 2 | Bridge | Blue `#0000ff` |
| 3 | Built-Up Area | Yellow `#ffff00` |

---

## Removal

```bash
rm -rf demo_ui/
```

Core training (`train.py`) and inference (`test_inference.py`) are unaffected.
