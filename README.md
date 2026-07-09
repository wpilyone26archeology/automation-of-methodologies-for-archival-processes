# Automation of Methodologies for Archival Processing

This project contains four documented pipelines for archaeological data organization and management that aim to assist in the implementation of artificial intelligence to assist experts in these processes. Each of the four pipelines come with fully documented procedures and results that were produced by implementing these processes on the L’Art à Hué artbook and a dataset of seventeenth century Vietnamese steles.

```
automation-of-methodologies-for-archival-processes/
├── authority-file/
│   ├── authorityFile.txt
│   ├── "Entity List.xlsx"
│   ├── method-for-creating-an-authority-file-database.pdf
│   └── Neo4j-access-codes.txt
├── document-preservation/
│   ├── "Dublin Core.xmp"
│   ├── method-for-appending-metadata-to-files-for-document-preservation.pdf
│   ├── "L'Art à Hue.xmp"
│   └── RenameToDCIdentifier.jsx
├── text-analysis-ocr/
│   └── method-for-performing-multilingual-ocr-and-transcribing-textual-documents.pdf
├── ai-pipeline-for-image-recognition-and-classification.pdf
├── README
├── stelemotifpipeline2.py
└── test_connection.py
```

---

## Document Preservation

"Method for Appending Metadata to Files for Document Preservation"
(method-for-appending-metadata-to-files-for-document-preservation.pdf)

### Purpose

The purpose of this process is to append metadata to image files so that they contain all
necessary information to enable automatic classification, simplified data organization, and future document reconstruction. The metadata follows the Dublin Core standard and the files should be saved in a lossless compression TIFF file format, as recommended by the Library of Congress (Dublin Core Metadata Initiative (DCMI), n.d.; Library of Congress, n.d.-b).

### Requirements

- Adobe Photoshop
- Adobe Bridge
- Excel
- Metadata Deluxe toolkit

### Files

- “Dublin Core.xmp” - blank XMP template for uploading metadata to image files in with the fifteen elements of the Dublin Core standard.
- “Rename to DC Identifier (Fixed)” Custom script - renames outputted files to the specified Dublin Core identifier

---

## Textual Analysis

"Method for Performing Multilingual OCR and Transcribing Textual Documents"
(method-for-performing-multilingual-ocr-and-transcribing-textual-documents.pdf)

### Purpose

The purpose of the textual analysis pipeline is to transcribe physical documents into a machine readable format. The outlined pipeline is compatible for multilingual uses as well.

### Requirements

- ScanTailor
- Access to LLM with OCR capabilities

### Files

Appendix C: LLM Prompts Used for OCR

---

## Authority File

"Method for Creating an Authority File Database"
(method-for-creating-an-authority-file-database)

### Purpose

The purpose of this process is to create a reference framework (authority file) that contains
entities and relationships between them. This framework should be in the form of a graphical
database and should contain necessary translations and alternate terminology for all entities.


### Requirements

- Text editor
- Spreadsheet software
- Neo4j

---

# Vietnamese Stele Motif Detection & Segmentation Pipeline

An automated computer vision pipeline for detecting and segmenting decorative motifs from digitized graphite rubbings of Vietnamese stone steles. The pipeline combines a locally-hosted vision-language model (Qwen3-VL-8B) with Meta's SAM2 segmentation model to identify and precisely outline carved motifs such as dragons, lotus flowers, cloud scrolls, and phoenixes.

---

## Background

Vietnamese stone steles present a significant challenge for standard object detection methods. The imagery in this pipeline originates from graphite rubbings of carved stone, transferred to paper, and digitally scanned. These scans exist as high-contrast monochromatic images that are visually unlike the natural photographic scenes that most pretrained detectors were trained on. The use of a large vision-language model (VLM) for detection, rather than a conventional object detector, allows the pipeline to leverage semantic and cultural understanding of the motifs rather than purely low-level visual features.

The pipeline processes images in the following stages:

1. The input image is divided into overlapping tiles
2. Each tile is sent to Qwen3-VL-8B with a culturally-informed prompt describing the target motifs
3. Detections across tiles are remapped to full-image coordinates and deduplicated via NMS
4. Each candidate detection is verified with a second VLM call on the cropped region
5. Verified bounding boxes are passed to SAM2 for pixel-precise segmentation masks
6. Results are exported as a visualized overlay, individual mask PNGs, and a JSON file


---

## Requirements

### Software

- **Python 3.10 or higher**
- **[LM Studio](https://lmstudio.ai/)** — for hosting Qwen3-VL-8B locally
- **CUDA-compatible GPU** — strongly recommended for SAM2 performance; CPU inference is supported but slow
- **Conda** — recommended for environment management

### Python Packages

Install all dependencies into your environment:

```bash
pip install openai pillow numpy matplotlib tifffile torch torchvision
```

If you are using a conda environment (recommended):

```bash
conda activate your_env_name
pip install openai pillow numpy matplotlib tifffile
```

`torch` and `torchvision` should be installed following the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) so the correct CUDA version is selected for your system.

### SAM2

Clone and install Meta's SAM2 from GitHub:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

Then download the model checkpoint. From inside the cloned `sam2/` directory:

```bash
cd checkpoints
./download_ckpts.sh
```

This downloads several checkpoint sizes. The pipeline uses `sam2.1_hiera_large.pt` by default. If you are constrained on disk space or GPU memory, `sam2.1_hiera_small.pt` is a viable alternative (update `SAM2_CHECKPOINT` and `SAM2_CONFIG` in the pipeline config accordingly).

### Qwen3-VL-8B via LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Search for and download `Qwen3-VL-8B` (or `qwen/qwen3-vl-8b:2`) from within LM Studio
3. Before running the pipeline, navigate to the **Local Server** tab in LM Studio and click **Start Server**. The server must be running and the model loaded for the pipeline to function
4. Note the exact model identifier shown in the server tab — you will need this for the config

---

## File Structure

```
your_project/
├── stele_motif_pipeline.py     ← main pipeline script
├── test_connection.py             ← optional diagnostic script
├── checkpoints/
│   └── sam2.1_hiera_large.pt
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml
├── sam2/                          ← cloned SAM2 repository
├── test_images/                   ← place your input images here
│   ├── stele_inverted.tif         ← color-inverted B&W image (used for detection)
│   └── stele_original.jpg         ← original colored image (used for visualization)
└── output/                        ← created automatically on first run
    ├── detection_overlay.png
    ├── detections.json
    ├── raw_detections.json
    └── mask_000_dragon.png
```

`stele_motif_pipeline.py` and `reference_library.py` must be in the same directory, as the pipeline imports directly from the reference library module.

---

## Configuration

Open `stele_motif_pipeline.py` and edit the CONFIG section near the top of the file:

```python
# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGE_PATH    = "stele_inverted.tif"      # path to your inverted B&W TIF
ORIGINAL_PATH = "stele_original.jpg"      # path to your original colored image

# ── LM Studio connection ──────────────────────────────────────────────────────
DASHSCOPE_API_KEY = "lm-studio"           # LM Studio does not require a real key
DASHSCOPE_BASE    = "http://localhost:1234/v1"   # default LM Studio server address
VLM_MODEL         = "qwen/qwen3-vl-8b:2" # must match exactly what LM Studio shows

# ── SAM2 ──────────────────────────────────────────────────────────────────────
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

All other parameters (tile size, overlap, confidence thresholds, NMS IoU threshold) are documented in the config section and can be tuned to suit different images.

---

## Input Image Preparation

The pipeline is designed for **color-inverted** stele scans, where the stone background appears black and carved motifs appear white. This is the preferred input for the VLM detection stage.

If your scans are not already inverted, you can invert them with any image editing tool (Photoshop, GIMP, or the Python snippet below):

```python
from PIL import Image, ImageOps
img = Image.open("stele_original.jpg").convert("RGB")
ImageOps.invert(img).save("stele_inverted.tif")
```

The original non-inverted image is used only for the final visualization and SAM2 segmentation, as SAM2 produces better masks on natural-looking imagery. If no original is provided, the pipeline falls back to using the inverted image for all stages.

---

## Usage

### 1. Verify LM Studio is running

Before any pipeline run, confirm that LM Studio's local server is active and reachable:

```bash
python test_connection.py
```

Expected output:
```
Checking available models...
Server is reachable. Models available:
  - qwen/qwen3-vl-8b:2
Testing vision call...
Vision call succeeded: <color>
```

If this fails, ensure the server is started inside LM Studio and that the model is loaded in the server tab (not just the chat tab).

### 2. Run the pipeline

```bash
# Detect a single motif type
python stele_motif_pipeline.py --image test_images/stele_inverted.tif --original test_images/stele_original.jpg --motifs dragon

# Detect multiple motif types in a single pass
python stele_motif_pipeline.py --image test_images/stele_inverted.tif --original test_images/stele_original.jpg --motifs dragon lotus cloud

# Skip the verification pass for faster (but less precise) results
python stele_motif_pipeline.py --image test_images/stele_inverted.tif --original test_images/stele_original.jpg --motifs dragon --skip-verify
```

### 3. Review outputs

Results are saved to the `output/` directory:

- `detection_overlay.png` — full-image visualization with colored masks and bounding boxes overlaid
- `mask_000_dragon.png`, `mask_001_dragon.png`, etc. — individual binary mask PNG for each detected motif
- `detections.json` — structured JSON with label, VLM confidence, SAM2 score, quality flag, bounding box, and description for each detection
- `raw_detections.json` — all detections before NMS and verification, useful for debugging

In the overlay and JSON, detections with a SAM2 score below 0.60 are flagged with ⚠ and marked `"quality": "review"` — these are worth inspecting manually before treating as confirmed.

---

## Supported Motif Types

The following motif types are supported out of the box with culturally-informed visual descriptions:

| Label | Vietnamese | Description |
|---|---|---|
| `dragon` | rồng | Sinuous serpentine creature with scaled body, clawed limbs, and flowing mane |
| `lotus` | hoa sen | Stylized floral form with radiating petals; often appears as border frieze |
| `cloud` | mây cuộn | Abstract curling C- or S-shaped scroll forms |
| `phoenix` | phụng | Bird figure with elaborate tail plumes, often paired with a dragon |
| `inscription` | văn bản | Dense panel of Chinese or Vietnamese characters in vertical columns |

To add a new motif type, add an entry to the `MOTIF_DESCRIPTIONS` dictionary in `stele_motif_pipeline.py`:

```python
MOTIF_DESCRIPTIONS["tortoise"] = (
    "Tortoise (rùa): a turtle figure, often depicted supporting a stele on its back. "
    "Identifiable by a domed shell with geometric patterning and four short limbs."
)
```

It will then be available via `--motifs tortoise`.

---

## Tuning for Different Images

A few parameters worth adjusting if detection quality is poor on a particular image:

**Too few detections (missed motifs):**
- Lower `CONFIDENCE_MIN` from `0.20` to `0.10`
- Lower `SAM2_SCORE_MIN` from `0.60` to `0.50`
- Run with `--skip-verify` to bypass the verification pass
- Reduce `TILE_SIZE` to `768` if motifs are small relative to the image

**Too many false positives:**
- Raise `CONFIDENCE_MIN` to `0.35` or higher
- Ensure the reference library has clean, unambiguous examples
- Add more specific negative guidance to the relevant entry in `MOTIF_DESCRIPTIONS`

**Large motifs being split across tile boundaries:**
- Increase `TILE_SIZE` to `1536` and `OVERLAP` to `200`

---

## Common Errors

**Connection error on tile processing:**
Confirm LM Studio's local server is started and the model is loaded in the server tab. Run `python test_connection.py` to diagnose. Ensure `DASHSCOPE_BASE` is `http://` (not `https://`) and the path is `/v1` (lowercase).

**SAM2 UserWarning about Flash Attention:**
These warnings are harmless on Windows. PyTorch's standard Windows builds do not include Flash Attention; SAM2 falls back to a standard attention kernel automatically. Segmentation will still complete correctly.

**`IndexError: arrays used as indices must be of integer (or boolean) type`:**
SAM2 returned float masks instead of boolean. This is handled in v3 with explicit `.astype(bool)` casts. If you see this in a modified version of the code, add `.astype(bool)` wherever a SAM2 mask is used as an index.

**No detections found:**
If raw_detections.json is also empty, lower `CONFIDENCE_MIN` and check that the prompt descriptions in `MOTIF_DESCRIPTIONS` match what is actually depicted on your stele. Consider adding reference crops if the library is empty.

---

## Acknowledgements

- [Meta SAM2](https://github.com/facebookresearch/sam2) — Segment Anything Model 2
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) — Qwen Vision-Language Model by Alibaba
- [LM Studio](https://lmstudio.ai/) — local model hosting and OpenAI-compatible API server
