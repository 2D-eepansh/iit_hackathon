# Submission Document

## AI/ML Hackathon -- Ministry of Panchayati Raj (Geospatial Intelligence Challenge)

**Powered by: Geo-Intel Lab, IIT Tirupati Navavishkar I-Hub Foundation**

---

## 1. Title of the Idea

**Automated Multi-Class Rural Infrastructure Segmentation from SVAMITVA Drone Orthomosaics Using Deep Learning**

---

## 2. Team/Individual Name & Affiliation

**Name**: [Your Name]
**Institution/Organization**: [Your Institution/Organization]
**Contact**: [Email / Phone]

**Supported by**: Geospatial Intelligence and Applications Laboratory, IIT Tirupati Navavishkar I-Hub Foundation

---

## 3. Problem Statement Addressed

India's SVAMITVA scheme seeks to survey and map over 660,000 villages using high-resolution drone orthomosaics for property card generation. A critical bottleneck in this workflow is the extraction of infrastructure features -- roads, bridges, built-up areas, and water bodies -- from the resulting imagery. Each village orthomosaic spans up to 235,000 x 120,000 pixels, making manual digitization prohibitively slow: a single village can take trained annotators several days. At the national scale, the labour and cost required for manual feature extraction render the current approach unsustainable.

The challenge is compounded by severe class imbalance in rural landscapes -- over 85% of pixels are background vegetation and terrain, while critical features like bridges occupy less than 0.1% of any given image. Current automated tools lack the geospatial awareness and class-specific engineering required to handle these rural mapping conditions reliably.

---

## 4. Proposed Solution

We developed an end-to-end deep learning pipeline that ingests raw georeferenced drone orthomosaics and shapefile annotations, and produces spatially accurate, GIS-ready infrastructure prediction masks at full resolution.

The system operates across four integrated stages. **Data ingestion** reads large GeoTIFF rasters using windowed access -- never loading full images into memory -- and rasterizes shapefile vector annotations to pixel masks on-the-fly with automatic CRS alignment. **Training** extracts 768x768 patches using minority-aware centroid sampling (90% centered on infrastructure features), applies spatial and photometric augmentations including bridge copy-paste for extreme scarcity, and trains a DeepLabV3+ model with a composite Focal + Dice loss specifically tuned for rural class imbalance. **Inference** processes full-resolution orthomosaics using strip-based sliding-window prediction with overlap blending, outputting GeoTIFF masks that preserve the original coordinate reference system. A calibrated ensemble of two EMA checkpoints with per-class logit bias tuning and test-time augmentation further improves detection accuracy. **Post-processing** applies domain-specific morphological refinement: road gap filling to reconnect broken road networks, bridge spatial recovery from misclassified built-up regions using proximity and geometry heuristics, and water body stabilization with confidence gating.

The system produces georeferenced masks, village infrastructure statistics in physical units (metres, square metres, hectares), and a confidence heatmap identifying regions requiring human review -- enabling a human-in-the-loop verification workflow.

---

## 5. Uniqueness and Innovation

Three aspects distinguish this solution from standard image segmentation approaches.

**Geospatial-native processing.** Unlike approaches that treat orthomosaics as regular images, our pipeline operates directly on georeferenced data throughout. Annotations are rasterized from vector shapefiles per patch with automatic CRS reprojection, and predictions are written as GeoTIFFs preserving spatial metadata -- making outputs immediately usable in existing SVAMITVA GIS workflows without manual georeferencing.

**Memory-bounded arbitrarily-large inference.** Orthomosaics of 235,000 x 120,000 pixels cannot be loaded into memory. Our strip-based sliding-window approach processes the image in horizontal strips, accumulating prediction logits only for the active strip, enabling inference on arbitrarily large rasters with bounded GPU and CPU memory (under 8 GB).

**Domain-aware post-processing pipeline.** Rather than relying solely on neural network output, we apply geospatially-informed refinement: road gap filling reconnects discontinuities in the predicted road network; bridge spatial recovery reclassifies elongated built-up components near road-water intersections as bridges; and confidence-gated water body filtering eliminates ghost predictions. This hybrid neural-plus-heuristic approach directly addresses the confusion patterns observed in the model's predictions.

**Calibrated ensemble with human-in-the-loop output.** Two EMA checkpoints are ensembled with learned per-class bias, and a confidence heatmap flags low-certainty regions for manual verification -- balancing automation throughput with quality assurance.

---

## 6. Technology Stack / Methodology

**Deep Learning Framework.** PyTorch with mixed-precision training (AMP), gradient accumulation (effective batch size 16), and gradient clipping for stable convergence on limited GPU memory.

**Model Architecture.** DeepLabV3+ with a ResNet-50 encoder (ImageNet-pretrained) via the segmentation-models-pytorch library. The Atrous Spatial Pyramid Pooling module captures multi-scale context critical for detecting both narrow road structures and broad built-up regions simultaneously.

**Loss Function.** A composite of Focal Loss (gamma=2.0, per-class alpha weighting with bridge at 3x) and class-weighted Dice Loss (road at 2x weight), balancing hard-example mining with overlap optimization on foreground classes.

**Optimization.** AdamW with differential learning rates (encoder 1e-5, decoder 1e-4), ReduceLROnPlateau scheduling, and Exponential Moving Average (decay=0.99, updated per optimizer step) for checkpoint stability against class-collapse events.

**Geospatial Stack.** rasterio for windowed raster access, geopandas and fiona for shapefile parsing, shapely for geometry operations, pyproj for CRS management.

**Inference Pipeline.** Two-model ensemble with calibrated logit bias, test-time augmentation (horizontal and vertical flips), and a five-stage morphological post-processing pipeline (road gap fill, road refinement, water stabilization, bridge filtering, bridge spatial recovery).

**Augmentation.** Albumentations pipeline with spatial transforms (flip, rotate, affine +/-45 degrees), photometric jitter, and bridge copy-paste augmentation from 30 cached bridge patches.

---

## 7. Expected Impact (Environmental, Social, Economic)

Automating infrastructure segmentation from SVAMITVA orthomosaics reduces the feature extraction workload per village from days of manual annotation to minutes of compute time. At the scale of 660,000+ villages, this translates to substantial savings in labour cost and accelerated timelines for property card generation.

The system produces outputs directly compatible with GIS tools used in SVAMITVA workflows, enabling seamless integration without additional conversion steps. The village-level infrastructure statistics -- road network length, built-up area, water body coverage, and building counts in physical units -- provide structured data for Gram Panchayat development planning, resource allocation, and progress monitoring.

By flagging low-confidence predictions for human review, the system supports a practical human-in-the-loop workflow: surveyors focus on verification and edge cases rather than exhaustive manual digitization, improving both throughput and data quality for rural property documentation.

---

## 8. Implementation Plan / Roadmap

**Phase 1 -- Current System (Completed).** Built and validated the full pipeline: unified multi-state dataset module (Punjab and Chhattisgarh), DeepLabV3+ model with class-imbalance engineering, strip-based full-resolution inference, calibrated two-model ensemble with post-processing, evaluation reporting, and a Streamlit demonstration interface. Validated on held-out village orthomosaics with 94% infrastructure detection accuracy and mean infrastructure F1 of 0.712.

**Phase 2 -- Expansion (3-6 Months).** Incorporate orthomosaic data from additional Indian states to improve geographic generalization. Add temporal analysis capabilities for monitoring infrastructure development over successive survey flights. Integrate higher-resolution drone sensors and new feature classes (agricultural boundaries, footpaths) as SVAMITVA data coverage expands. Implement active learning to prioritize annotation of high-uncertainty regions, reducing labelling cost.

**Phase 3 -- Deployment (6-12 Months).** Package the inference pipeline for production deployment using ONNX Runtime or TensorRT for optimized throughput. Develop a web-based interface for surveyors to upload village orthomosaics, receive automated predictions, and review flagged regions. Integrate with existing SVAMITVA GIS infrastructure for direct ingestion of outputs into the property card generation workflow. Establish automated quality assurance checks and periodic model retraining.

---

## 9. Required Resources

**Compute.** NVIDIA GPU with 8+ GB VRAM for training (developed on RTX 5070, 16 GB VRAM, CUDA 12.8). A single training run completes in approximately 2 hours for 80 epochs. Inference on a full village orthomosaic requires under 10 minutes on equivalent hardware.

**Data.** SVAMITVA orthomasic GeoTIFFs and shapefile annotations. Current pipeline is validated on Punjab and Chhattisgarh data (~2 GB). Expansion requires annotated orthomosaics from additional states, which the SVAMITVA programme is actively generating.

**Software.** Python 3.10+, PyTorch, rasterio, geopandas, segmentation-models-pytorch, Albumentations, and supporting geospatial libraries. All dependencies are open-source and listed in `requirements.txt`.

**Personnel.** One ML engineer for model development and optimization; one geospatial analyst for data quality and GIS integration; domain support from survey agencies for validation.

---

## 10. Scalability and Sustainability

The pipeline is designed for horizontal scalability. Additional states and village datasets are incorporated by adding entries to the data source configuration -- no code changes are required. The architecture is agnostic to geographic region, CRS system, or image resolution.

The windowed raster reading and strip-based inference approach imposes no upper limit on input image resolution -- it processes 235,000 x 120,000 pixel orthomosaics within 8 GB GPU memory. For production throughput, the ONNX export utility enables integration into TensorRT or ONNX Runtime serving frameworks.

The modular codebase (separate data, model, loss, training, inference, and post-processing modules) supports independent maintenance and extension. All components are built on open-source libraries with active communities, ensuring long-term compatibility. The system's geographic-agnostic design means it can extend beyond SVAMITVA to other government drone survey programmes.

---

## 11. Stakeholders and Collaborations

**Primary Stakeholders.** Ministry of Panchayati Raj (scheme governance and national rollout), Survey of India (survey standards, data collection protocols, and quality benchmarks), State Revenue Departments (property card generation and land record management), and Gram Panchayats (beneficiary-level administration and local infrastructure planning).

**Technical Collaborators.** Geo-Intel Lab at IIT Tirupati Navavishkar I-Hub Foundation provides the geospatial AI research context, domain expertise, and computational infrastructure. Drone survey agencies contracted under SVAMITVA are upstream data providers whose orthomosaic output directly feeds into this pipeline.

**Beneficiaries.** Rural households across India awaiting property documentation under SVAMITVA benefit from accelerated processing timelines. Field surveyors gain reduced manual annotation workload, enabling them to cover more villages per cycle. District and state administrative bodies managing land records digitization at scale benefit from structured, quantitative village infrastructure data for evidence-based planning and resource allocation.

---

## 12. Current Stage of Development

**Stage: Prototype / MVP**

The system is fully implemented and operational. The training, inference, and evaluation pipelines are reproducible end-to-end. The best model achieves a mean IoU of 0.5880 on held-out orthomosaics with calibrated ensemble inference, with 94% infrastructure detection accuracy and a mean infrastructure F1 of 0.712. A Streamlit demo interface is functional for live presentation, supporting image upload, real-time segmentation, confidence visualization, and village infrastructure statistics. The repository includes comprehensive documentation and structured evaluation reporting.

---

## 13. Supporting Materials

| Material | Location |
|----------|----------|
| Source Code | Repository root and `src/` directory |
| Model Checkpoint | `outputs/checkpoints/best_model.pth` |
| Evaluation Report | `outputs/evaluation_report.txt` and `outputs/evaluation_report.json` |
| Training Curves | `outputs/plots/` |
| Test Predictions | `outputs/test_predictions_live_demo/` |
| System Documentation | `docs/` directory (architecture, training, evaluation guides) |
| Demo Interface | `demo_ui/app.py` (Streamlit) |
| Demo Instructions | `docs/demo_instructions.md` |
| Requirements | `requirements.txt` |
