"""
Stele Motif Detection & Segmentation Pipeline  (v2)
====================================================
Qwen3-VL (local via LM Studio) → NMS → Verification → SAM2

Changes from v1:
  - Multi-motif detection in a single VLM pass per tile
  - Richer prompts: negative guidance, medium description, spatial context
  - Verification pass: each candidate re-queried with a binary yes/no crop check
  - Better SAM2 mask selection: filters by plausible coverage area
  - Per-label NMS
  - SAM2 score used as quality signal in JSON export

Requirements (likely already installed from GroundedSAM2 / Florence work):
    pip install openai pillow numpy matplotlib tifffile

Usage:
    python stele_motif_pipeline.py --image stele_inverted.tif --original stele_original.jpg
    python stele_motif_pipeline.py --image stele_inverted.tif --original stele_original.jpg --motifs dragon lotus cloud
"""

import os
import json
import base64
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from io import BytesIO
from PIL import Image

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

IMAGE_PATH    = "test_images/10925bwi.tif"
ORIGINAL_PATH = "test_images/10925.jpg"
OUTPUT_DIR    = Path("output")

# Tiling
TILE_SIZE = 1536   # px — increase to 1536 for large motifs, decrease to 768 for fine detail
OVERLAP   = 250   # px — increase proportionally if you change TILE_SIZE

# Detection
CONFIDENCE_MIN  = 0.3   # lowered from 0.3; rely on verification + SAM2 score instead
NMS_IOU_THRESH  = 0.45    # was 0.5 initially
MIN_COVERAGE    = 0.0001  # mask must cover at least 0.01% of image area
MAX_COVERAGE    = 0.20    # mask must cover at most 20% of image area
SAM2_SCORE_MIN  = 0.60   # detections with SAM2 score below this are flagged for review

# LM Studio local server
DASHSCOPE_API_KEY = "lm-studio"
DASHSCOPE_BASE    = "http://localhost:1234/v1"
VLM_MODEL         = "qwen/qwen3-vl-8b"   # match exactly what LM Studio shows

# SAM2
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Default motifs to detect (overridden by --motifs argument)
DEFAULT_MOTIFS = ["dragon"]

# ── MOTIF VISUAL DESCRIPTIONS ────────────────────────────────────────────────
# Add or edit entries here to support new motif types.
# Each value is inserted into the detection prompt verbatim.

MOTIF_DESCRIPTIONS = {
    "dragon": (
        "Dragon (rồng): a sinuous, serpentine creature with an elongated scaled body, "
        "multiple clawed limbs, an open mouth with whiskers or flame elements, and a flowing "
        "mane or tail. May be depicted ascending, descending, coiling, or symmetrically flanking "
        "an inscription. Commonly found at the tympanum arch, flanking the central text panel, "
        "or as column border decorations. The body should show a clear S-curve or spiral form."
    ),
    "lotus": (
        "Lotus (hoa sen): a stylized floral form with multiple petals radiating outward from a "
        "central point, appearing as a full bloom, half-bloom, or bud. Often appears as a repeated "
        "border frieze, as decorative fill between other motifs, or as a base element beneath "
        "figures. Petals are typically rounded and symmetrically arranged."
    ),
    "cloud": (
        "Cloud scroll (mây cuộn): curling, C-shaped or S-shaped ribbon-like forms arranged in "
        "repeating bands or interwoven with other motifs. Often appears as background fill or "
        "as a border pattern. Distinguished from dragons by the absence of any animal features — "
        "purely abstract curvilinear forms."
    ),
    "flaming globe": (
        "flaming globe: A circular ornament depicted with surrounding flames.  "
        
    ),
    "inscription": (
        "Text inscription panel: a rectangular region densely filled with Chinese or Vietnamese "
        "characters (chữ Hán or chữ Nôm), typically arranged in vertical columns. Distinguished "
        "from decorative motifs by the presence of recognisable character forms."
    ),
}


# ── IMAGE UTILITIES ───────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def tile_image(img: Image.Image, tile_size: int = 1024, overlap: int = 150):
    W, H = img.size
    stride = tile_size - overlap
    tiles = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            tile = img.crop((x, y, x2, y2))
            tiles.append((tile, x, y))
            if x2 == W:
                break
            x += stride
        if y2 == H:
            break
        y += stride
    return tiles


def encode_tile(tile: Image.Image, quality: int = 92) -> str:
    buf = BytesIO()
    tile.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── PROMPTS ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in Vietnamese decorative arts and stone stele iconography (bia đá). "
    "You are analysing digitally scanned graphite rubbings of Vietnamese stone steles. "
    "These images are colour-inverted: the stone background appears BLACK; carved relief motifs appear WHITE. "
    "Your task is to precisely locate specific decorative motifs and return bounding box pixel coordinates "
    "as a JSON object. You must respond with only the JSON object — no preamble, no explanation, no markdown."
)


def build_detection_prompt(motifs: list[str], tile_w: int, tile_h: int) -> str:
    # Build description block for each requested motif
    desc_lines = []
    for m in motifs:
        desc = MOTIF_DESCRIPTIONS.get(m, f"{m}: identify based on your knowledge of Vietnamese stele iconography.")
        desc_lines.append(f"  - {desc}")
    motif_block = "\n".join(desc_lines)
    motif_list  = ", ".join(f'"{m}"' for m in motifs)

    return (
        f"This is a {tile_w}×{tile_h} pixel tile from a colour-inverted graphite rubbing of a "
        f"Vietnamese stone stele. The stone surface is BLACK; carved motifs appear WHITE.\n\n"

        f"MEDIUM NOTE: This is a graphite rubbing transferred to paper then digitally scanned and "
        f"colour-inverted. The image may contain paper grain, uneven rubbing pressure, and areas of "
        f"incomplete ink transfer — these are artefacts, NOT motifs. Focus only on clearly intentional "
        f"carved forms with consistent white tone and defined edges.\n\n"

        f"Detect ALL instances of these motif types:\n{motif_block}\n\n"

        f"DO NOT flag: decorative border lines or geometric frames alone, Chinese/Vietnamese text "
        f"characters (chữ Hán or chữ Nôm), paper texture, rubbing artefacts, cloud scrolls without "
        f"animal features when searching for dragons, or any region where you are not confident a "
        f"specific motif type is present.\n\n"

        f"Before assigning coordinates, consider: does this region contain a complete or clearly "
        f"partial motif with identifiable features? Only return detections you are genuinely confident about.\n\n"

        f"Return ONLY this JSON object — no other text, no markdown:\n"
        f'{{"detections": [{{"label": <one of {motif_list}>, "confidence": <0.0–1.0>, '
        f'"bbox": [x1, y1, x2, y2], "description": "<brief note on pose or position>"}}]}}\n\n'
        f"If no target motifs are visible: {{\"detections\": []}}\n"
        f"Coordinates are in pixels relative to this tile (origin top-left). "
        f"Tile dimensions: {tile_w}w × {tile_h}h."
    )


def build_verification_prompt(motif: str) -> str:
    return (
        f"This is a cropped region from a colour-inverted graphite rubbing of a Vietnamese stone stele. "
        f"Carved motifs appear WHITE on a BLACK background.\n\n"
        f"Does this crop contain a {motif} motif — a recognisable decorative figure as it would appear "
        f"in Vietnamese stone carving tradition — as opposed to text characters, border lines, or "
        f"rubbing artefacts?\n\n"
        f"Reply with only YES or NO."
    )


# ── VLM CALLS ────────────────────────────────────────────────────────────────

def call_vlm_detect(client, tile_img: Image.Image, motifs: list[str]) -> list[dict]:
    tw, th = tile_img.size
    b64    = encode_tile(tile_img)
    prompt = build_detection_prompt(motifs, tw, th)

    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",      "text": prompt},
            ]},
        ],
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw).strip()

    try:
        return json.loads(raw).get("detections", [])
    except json.JSONDecodeError:
        print(f"    [WARN] Could not parse VLM response: {raw[:200]}")
        return []


def verify_detection(client, full_img: Image.Image, det: dict) -> bool:
    """Re-query the VLM with just the detection crop for a binary yes/no confirmation."""
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    pad = 60
    W, H = full_img.size
    crop = full_img.crop((max(0, x1-pad), max(0, y1-pad),
                          min(W, x2+pad), min(H, y2+pad)))
    b64  = encode_tile(crop)
    prompt = build_verification_prompt(det["label"])

    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",      "text": prompt},
            ]}],
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("Y")
    except Exception as e:
        print(f"    [WARN] Verification call failed: {e} — keeping detection")
        return True   # keep on error rather than silently discard


# ── COORDINATE REMAPPING ──────────────────────────────────────────────────────

def remap_to_global(detections: list[dict], x_off: int, y_off: int) -> list[dict]:
    remapped = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        remapped.append({**det, "bbox": [x1+x_off, y1+y_off, x2+x_off, y2+y_off]})
    return remapped


# ── PER-LABEL NMS ─────────────────────────────────────────────────────────────

def compute_iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    """Run NMS independently for each label, then combine."""
    by_label: dict[str, list] = {}
    for det in detections:
        by_label.setdefault(det["label"], []).append(det)

    kept = []
    for label, dets in by_label.items():
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if compute_iou(best["bbox"], d["bbox"]) < iou_threshold]
    return kept


# ── SAM2 SEGMENTATION ─────────────────────────────────────────────────────────

def segment_detections(image_rgb: np.ndarray, detections: list[dict]) -> list[dict]:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  SAM2 running on: {device}")

    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    predictor  = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image_rgb)

    image_area = image_rgb.shape[0] * image_rgb.shape[1]
    results    = []

    for det in detections:
        box = np.array(det["bbox"], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)

        # Prefer masks with a plausible coverage area; fall back to highest score
        valid = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            coverage = mask.astype(bool).sum() / image_area
            if MIN_COVERAGE < coverage < MAX_COVERAGE:
                valid.append((float(score), i))

        best_idx = max(valid, key=lambda x: x[0])[1] if valid else int(np.argmax(scores))

        results.append({
            "detection":  det,
            "mask":       masks[best_idx].astype(bool),
            "sam2_score": float(scores[best_idx]),
        })
        quality = "✓" if scores[best_idx] >= SAM2_SCORE_MIN else "?"
        print(
            f"    {quality} {det['label']:12s} | "
            f"VLM conf={det['confidence']:.2f} | SAM2={scores[best_idx]:.3f}"
        )

    return results


# ── VISUALISATION ─────────────────────────────────────────────────────────────

# One colour per motif label for consistent colouring across runs
LABEL_COLORS = {
    "dragon":      "#FF4136",
    "lotus":       "#2ECC40",
    "cloud":       "#0074D9",
    "phoenix":     "#FF851B",
    "inscription": "#B10DC9",
}
FALLBACK_COLORS = ["#FFDC00", "#01FF70", "#F012BE", "#7FDBFF", "#3D9970"]


def label_color(label: str, idx: int) -> str:
    return LABEL_COLORS.get(label, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def visualize_results(base_image: Image.Image, seg_results: list[dict], output_path: str, dpi: int = 150):
    W, H = base_image.size
    fig, ax = plt.subplots(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax.imshow(base_image)

    for i, res in enumerate(seg_results):
        det   = res["detection"]
        color = label_color(det["label"], i)
        r, g, b = tuple(int(color.lstrip("#")[j:j+2], 16)/255 for j in (0, 2, 4))

        mask = res["mask"].astype(bool)
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        rgba[mask] = [r, g, b, 0.45]
        ax.imshow(rgba, interpolation="nearest")

        x1, y1, x2, y2 = det["bbox"]
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=1.5, edgecolor=color, facecolor="none"
        ))
        quality_flag = "" if res["sam2_score"] >= SAM2_SCORE_MIN else " ⚠"
        ax.text(
            x1, max(y1-6, 0),
            f"{det['label']} {det['confidence']:.2f} (SAM {res['sam2_score']:.2f}){quality_flag}",
            color="white", fontsize=5, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.7, pad=1, linewidth=0),
        )

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay → {output_path}")


def export_masks(seg_results: list[dict], output_dir: Path):
    for i, res in enumerate(seg_results):
        label    = res["detection"]["label"]
        mask_img = Image.fromarray(res["mask"].astype(bool).astype(np.uint8) * 255, mode="L")
        fname    = output_dir / f"mask_{i:03d}_{label}.png"
        mask_img.save(fname)
    print(f"  Saved {len(seg_results)} mask PNGs → {output_dir}")


def export_json(seg_results: list[dict], path: str):
    export = [{
        "label":          r["detection"]["label"],
        "confidence_vlm": r["detection"]["confidence"],
        "confidence_sam2": r["sam2_score"],
        "quality":        "high" if r["sam2_score"] >= SAM2_SCORE_MIN else "review",
        "bbox_xyxy":      r["detection"]["bbox"],
        "description":    r["detection"].get("description", ""),
    } for r in seg_results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"  Saved detections JSON → {path}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path:    str   = IMAGE_PATH,
    original_path: str   = ORIGINAL_PATH,
    motifs:        list  = None,
    output_dir:    Path  = OUTPUT_DIR,
    skip_verify:   bool  = False,
):
    from openai import OpenAI

    if motifs is None:
        motifs = DEFAULT_MOTIFS

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load images
    print(f"\n[1/6] Loading detection image: {image_path}")
    detect_img = load_image(image_path)
    W, H = detect_img.size
    print(f"      Dimensions: {W} × {H} px")

    if os.path.exists(original_path):
        print(f"      Loading original for visualisation: {original_path}")
        vis_img = load_image(original_path)
    else:
        print("      No original found — visualising on detection image.")
        vis_img = detect_img

    # 2. Tile
    print(f"\n[2/6] Tiling (tile={TILE_SIZE}px, overlap={OVERLAP}px)...")
    tiles = tile_image(detect_img, tile_size=TILE_SIZE, overlap=OVERLAP)
    print(f"      {len(tiles)} tiles  |  detecting: {', '.join(motifs)}")

    # 3. VLM detection
    print(f"\n[3/6] VLM detection...")
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE)
    print(f"      Endpoint: {DASHSCOPE_BASE}  |  Model: {VLM_MODEL}")

    all_detections = []
    for idx, (tile_img, x_off, y_off) in enumerate(tiles):
        print(f"  Tile {idx+1:02d}/{len(tiles)}  offset=({x_off},{y_off})  size={tile_img.size}")
        try:
            dets = call_vlm_detect(client, tile_img, motifs)
            dets = [d for d in dets if d.get("confidence", 0) >= CONFIDENCE_MIN]
            dets = remap_to_global(dets, x_off, y_off)
            all_detections.extend(dets)
            if dets:
                for d in dets:
                    print(f"    + {d['label']} (conf={d['confidence']:.2f})")
            else:
                print("    (no detections)")
        except Exception as e:
            print(f"    [ERROR] Tile {idx+1} failed: {e}")

    print(f"\n      Raw detections: {len(all_detections)}")
    with open(output_dir / "raw_detections.json", "w") as f:
        json.dump(all_detections, f, indent=2)

    if not all_detections:
        print("\n  No detections found. Try lowering CONFIDENCE_MIN or revising prompts.")
        return

    # 4. NMS
    print(f"\n[4/6] NMS (IoU threshold={NMS_IOU_THRESH})...")
    post_nms = nms(all_detections, iou_threshold=NMS_IOU_THRESH)
    print(f"      After NMS: {len(post_nms)} (was {len(all_detections)})")

    # 5. Verification pass
    if not skip_verify:
        print(f"\n[4b] Verification pass ({len(post_nms)} candidates)...")
        verified = []
        for det in post_nms:
            result = verify_detection(client, detect_img, det)
            status = "✓ confirmed" if result else "✗ rejected"
            print(f"    {status}: {det['label']} at {[int(v) for v in det['bbox']]}")
            if result:
                verified.append(det)
        print(f"      Confirmed: {len(verified)}/{len(post_nms)}")
        final_detections = verified
    else:
        final_detections = post_nms

    if not final_detections:
        print("\n  No detections survived verification.")
        print("  Run with --skip-verify to bypass, or lower CONFIDENCE_MIN.")
        return

    # 6. SAM2 segmentation
    print(f"\n[5/6] SAM2 segmentation on {len(final_detections)} detection(s)...")
    vis_np  = np.array(vis_img)
    results = segment_detections(vis_np, final_detections)
    # results : (det, mask, sam2_score)
    # box = np.array(det["bbox"], dtype=np.float32)  # [x1, y1, x2, y2]

    # 7. Export
    print(f"\n[6/6] Exporting to {output_dir}/")
    visualize_results(vis_img, results, str(output_dir / "detection_overlay.png"))
    export_masks(results, output_dir)
    export_json(results, str(output_dir / "detections.json"))

    high  = sum(1 for r in results if r["sam2_score"] >= SAM2_SCORE_MIN)
    print(f"\n✓ Done. {len(results)} motif(s) segmented — {high} high confidence, "
          f"{len(results)-high} flagged for review.")
    
    # MEAN AVERAGE PRECISION
    ann_path = "instances_default.json"
    
    try:
        with open(ann_path, "r") as file:
            ann_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find ground truth file at '{ann_path}'.")
        ann_data = {"annotations": []}

    # 1. Directly parse all annotations into a flat list, converting XYWH to XYXY
    annotations = []
    for ann in ann_data.get("annotations", []):
        xg, yg, wg, hg = ann["bbox"]
        xyxy_gt = [xg, yg, xg + wg, yg + hg]
        annotations.append(xyxy_gt)

    # 2. Your 'results' list contains all detections for this single image.
    # Structure remains: [{"detection": det, "mask": m, "sam2_score": s}, ...]
    # results = [...] 

    # 3. Compute AP directly for the single image (for one image, AP equals mAP)
    if len(annotations) > 0:
        mAP = compute_ap(results, annotations, iou_threshold=0.5)
        print(f"Mean Average Precision (mAP@0.5): {mAP:.4f}")
    else:
        print("\nNo annotations detected to evaluate.")

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes.
    Both boxes must be in [x1, y1, x2, y2] (XYXY) format.
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Find intersection boundaries
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Calculate intersection area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate individual box areas
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    
    # Calculate union area
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compute_ap(results, annotations, iou_threshold=0.5):
    """
    Calculates Average Precision (AP) for a single image.
    """
    if len(annotations) == 0 or len(results) == 0:
        print(f"Empty data! Annotations: {len(annotations)}, Results: {len(results)}")
        return 0.0

    results = sorted(results, key=lambda x: x["detection"].get("confidence", 0.0), reverse=True)

    tp = np.zeros(len(results))
    fp = np.zeros(len(results))
    used = [False] * len(annotations)

    print(f"\n--- Debugging {len(results)} Detections against {len(annotations)} GT Boxes ---")

    for d_idx, res in enumerate(results):
        bbox = np.array(res["detection"]["bbox"], dtype=np.float32)
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for g_idx, ann_box in enumerate(annotations):
            iou = compute_iou(bbox, ann_box)
            
            # DEBUG PRINT FOR EVERY COMPARISON
            print(f"Det {d_idx} {bbox} vs GT {g_idx} {ann_box} -> IoU: {iou:.4f}")
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1 and not used[best_gt_idx]:
            tp[d_idx] = 1.0
            used[best_gt_idx] = True  
            print(f"  => MATCHED! True Positive (IoU: {best_iou:.4f})")
        else:
            fp[d_idx] = 1.0
            print(f"  => MISSED! False Positive (Best IoU: {best_iou:.4f})")

    # 4. Compute cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(annotations)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 5. Integrate area under Precision-Recall curve
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stele Motif Detection & Segmentation v2")
    parser.add_argument("--image",       default=IMAGE_PATH,    help="Inverted B&W TIF path")
    parser.add_argument("--original",    default=ORIGINAL_PATH, help="Original coloured JPG (optional)")
    parser.add_argument("--motifs",      nargs="+", default=DEFAULT_MOTIFS,
                        help="One or more motif types, e.g. --motifs dragon lotus cloud")
    parser.add_argument("--output",      default="output",      help="Output directory")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the verification pass (faster but less accurate)")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        original_path=args.original,
        motifs=args.motifs,
        output_dir=Path(args.output),
        skip_verify=args.skip_verify,
    )
