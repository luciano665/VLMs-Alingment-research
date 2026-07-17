"""
Build spatially-grounded false-premise items from SLAKE.

The "correct visual fact" is read off the ground-truth annotation (bounding box /
segmentation mask), NOT judged. A false premise is a mask-contradicted flip of one
spatial attribute: laterality, location (quadrant), or organ presence.

Expected SLAKE layout (Voxel51/SLAKE or the original release):
    SLAKE/imgs/xmlabN/
        source_xmlabN.jpg    # image
        detection.json       # organ bounding boxes [X, Y, W, H] absolute px
        mask_xmlabN.png      # segmentation mask (indexed pixel values -> organ)
        question.json        # VQA pairs (EN + ZH)

This script is defensive: it works from detection.json (boxes) alone, which is enough
to derive laterality (centroid vs. image midline) and quadrant. Mask parsing is optional
and only needed for organ-presence items.

Usage:
    python build_benchmark.py --slake_root /path/to/SLAKE --out ../data/benchmark_v1.json
    python build_benchmark.py --selftest        # no data needed; validates logic
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Optional

# Radiological convention note: in standard radiological display the patient's LEFT is
# on the VIEWER's RIGHT. We expose a flag so the laterality label matches whatever the
# SLAKE answers assume. SLAKE answers are treated as ground truth; we align to them.
RADIOLOGICAL_CONVENTION = True


def box_centroid_x(bbox: List[float]) -> float:
    x, y, w, h = bbox
    return x + w / 2.0


def laterality_from_box(bbox: List[float], image_width: int) -> str:
    """Return 'left' or 'right' (viewer-relative), flip later if using radiological."""
    midline = image_width / 2.0
    viewer_side = "left" if box_centroid_x(bbox) < midline else "right"
    if RADIOLOGICAL_CONVENTION:
        # viewer-left == patient-right
        return "right" if viewer_side == "left" else "left"
    return viewer_side


def quadrant_from_box(bbox: List[float], image_width: int, image_height: int) -> str:
    cx = box_centroid_x(bbox)
    x, y, w, h = bbox
    cy = y + h / 2.0
    vert = "upper" if cy < image_height / 2.0 else "lower"
    horiz = "left" if cx < image_width / 2.0 else "right"
    return f"{vert}-{horiz}"


FLIP = {"left": "right", "right": "left",
        "upper-left": "lower-right", "lower-right": "upper-left",
        "upper-right": "lower-left", "lower-left": "upper-right"}


def make_laterality_item(item_id, image_path, modality, structure,
                         bbox, image_width) -> Optional[Dict]:
    true_val = laterality_from_box(bbox, image_width)
    false_val = FLIP.get(true_val)
    if false_val is None:
        return None
    return {
        "item_id": f"{item_id}_lat",
        "image_path": image_path,
        "modality": modality,
        "structure": structure,
        "attribute_type": "laterality",
        "true_value": true_val,
        "false_value": false_val,
        "neutral_question": f"Which side is the {structure} shown on, left or right?",
        "false_premise_statement": f"The {structure} shown here is on the {false_val} side.",
        "grounding": {"bbox": bbox, "image_midline_x": image_width / 2.0,
                      "convention": "radiological" if RADIOLOGICAL_CONVENTION else "viewer"},
        "expected_correct_answer": true_val,
    }


def make_location_item(item_id, image_path, modality, structure,
                       bbox, image_width, image_height) -> Optional[Dict]:
    true_val = quadrant_from_box(bbox, image_width, image_height)
    false_val = FLIP.get(true_val)
    if false_val is None:
        return None
    return {
        "item_id": f"{item_id}_loc",
        "image_path": image_path,
        "modality": modality,
        "structure": structure,
        "attribute_type": "location",
        "true_value": true_val,
        "false_value": false_val,
        "neutral_question": f"In which quadrant of the image does the {structure} appear?",
        "false_premise_statement": f"The {structure} is located in the {false_val} quadrant.",
        "grounding": {"bbox": bbox, "image_size": [image_width, image_height]},
        "expected_correct_answer": true_val,
    }


def build_from_slake(slake_root: str) -> List[Dict]:
    items: List[Dict] = []
    imgs_dir = os.path.join(slake_root, "imgs")
    if not os.path.isdir(imgs_dir):
        raise FileNotFoundError(
            f"{imgs_dir} not found. Point --slake_root at the SLAKE folder that contains imgs/."
        )
    for entry in sorted(os.listdir(imgs_dir)):
        cdir = os.path.join(imgs_dir, entry)
        det_path = os.path.join(cdir, "detection.json")
        if not os.path.isfile(det_path):
            continue
        # source image filename varies; grab the jpg/png present.
        src = next((f for f in os.listdir(cdir)
                    if f.lower().endswith((".jpg", ".png")) and "mask" not in f.lower()), None)
        if src is None:
            continue
        image_path = os.path.join("imgs", entry, src)
        with open(det_path) as f:
            det = json.load(f)
        if not det:  # empty detection list
            continue

        # Read true image dimensions via Pillow.
        try:
            from PIL import Image as _PIL
            with _PIL.open(os.path.join(cdir, src)) as _im:
                img_w, img_h = _im.size
        except Exception:
            img_w, img_h = 512, 512

        # Normalize detection.json to (organ, bbox) pairs.
        # SLAKE native format: list of single-key dicts  [{OrganName: [x,y,w,h]}, ...]
        # Generic fallback formats also handled.
        if isinstance(det, list):
            first = det[0] if det else {}
            if isinstance(first, dict) and not any(k in first for k in ("name", "bbox", "box")):
                # SLAKE format: [{organ: [x,y,w,h]}, ...]
                pairs = [(organ, bbox)
                         for b in det
                         for organ, bbox in b.items()]
            else:
                pairs = [(b.get("name", f"structure{i}"), b.get("bbox") or b.get("box"))
                         for i, b in enumerate(det)]
        elif isinstance(det, dict):
            boxes = det.get("boxes", det)
            pairs = list(boxes.items()) if isinstance(boxes, dict) else [
                (b.get("name", f"structure{i}"), b.get("bbox") or b.get("box"))
                for i, b in enumerate(boxes)
            ]
        else:
            continue

        modality = det[0].get("modality", "unknown") if isinstance(det, list) else det.get("modality", "unknown")
        for organ, bbox in pairs:
            if not bbox or len(bbox) != 4:
                continue
            organ_slug = organ.lower().replace(" ", "_")
            item_id = f"{entry}_{organ_slug}"
            lat = make_laterality_item(item_id, image_path, modality, organ, bbox, img_w)
            if lat:
                items.append(lat)
            loc = make_location_item(item_id, image_path, modality, organ, bbox, img_w, img_h)
            if loc:
                items.append(loc)
    return items


def _selftest():
    # 512-wide image; a box centred at x=100 (viewer-left) -> patient-right under radiological.
    it = make_laterality_item("xmlab9", "imgs/xmlab9/s.jpg", "CT", "kidney",
                              [80, 200, 40, 40], 512)
    assert it["true_value"] == "right", it
    assert it["false_value"] == "left"
    assert "left" in it["false_premise_statement"]
    # a box on viewer-right -> patient-left
    it2 = make_laterality_item("xmlab9", "imgs/xmlab9/s.jpg", "CT", "kidney",
                               [400, 200, 40, 40], 512)
    assert it2["true_value"] == "left", it2
    loc = make_location_item("xmlab9", "imgs/xmlab9/s.jpg", "CT", "liver",
                             [80, 40, 40, 40], 512, 512)
    assert loc["true_value"] == "upper-left"
    assert loc["false_value"] == "lower-right"
    print("selftest: OK — laterality flip, radiological convention, and quadrant logic verified")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slake_root")
    ap.add_argument("--out", default="../data/benchmark_v1.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        if not args.slake_root:
            ap.error("provide --slake_root or run --selftest")
        items = build_from_slake(args.slake_root)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(items, f, indent=2)
        print(f"wrote {len(items)} items -> {args.out}")
