#!/usr/bin/env python3
"""
Compare PyTorch vs CoreML detection results on real bird photos.
Picks sample images from each directory, runs both pipelines,
and reports per-image detection agreement.
"""

import glob
import logging
import os
import sys
import time
from copy import deepcopy

import numpy as np
import rawpy
import torch
from PIL import Image

# Apply patches before any rfdetr imports
import rfdetr_coreml  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def load_arw(path):
    """Load a Sony ARW file and return a PIL Image."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
    return Image.fromarray(rgb)


def postprocess(boxes, logits, img_w, img_h, conf_threshold=0.3):
    """Convert raw model outputs to detections list."""
    import scipy.special
    probs = scipy.special.softmax(logits[0], axis=-1)
    scores = probs[:, :-1].max(axis=-1)
    labels = probs[:, :-1].argmax(axis=-1)

    detections = []
    for i in range(len(scores)):
        if scores[i] < conf_threshold:
            continue
        cx = boxes[0, i, 0] * img_w
        cy = boxes[0, i, 1] * img_h
        w = boxes[0, i, 2] * img_w
        h = boxes[0, i, 3] * img_h
        cls = COCO_CLASSES[labels[i]] if labels[i] < len(COCO_CLASSES) else f"class_{labels[i]}"
        detections.append({
            "class": cls,
            "score": float(scores[i]),
            "cx": float(cx), "cy": float(cy),
            "w": float(w), "h": float(h),
            "query_idx": i,
        })
    detections.sort(key=lambda d: -d["score"])
    return detections


def identify_outputs(pred):
    """Identify boxes and logits from CoreML output dict."""
    boxes = logits = None
    for k in pred:
        v = pred[k]
        if hasattr(v, 'shape') and len(v.shape) == 3:
            if v.shape[-1] == 4:
                boxes = v
            elif v.shape[-1] > 4:
                logits = v
    return boxes, logits


def main():
    import coremltools as ct
    from rfdetr_coreml.export import (
        MODEL_REGISTRY, NormalizedWrapper, _import_model_class,
        IMAGENET_MEAN, IMAGENET_STD,
    )

    model_name = "base"
    resolution = MODEL_REGISTRY[model_name][1]
    mlpackage_path = f"output/rf-detr-{model_name}-fp32.mlpackage"

    # =========================================================================
    # Load models
    # =========================================================================
    logger.info("Loading PyTorch model...")
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped_pt = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model

    logger.info("Loading CoreML model...")
    ml_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)

    # =========================================================================
    # Collect bird photos (sample from each directory)
    # =========================================================================
    dirs = [
        "/Volumes/P7000Z 2T/10060201",
        "/Volumes/P7000Z 2T/10260208",
        "/Volumes/P7000Z 2T/11060121",
        "/Volumes/P7000Z 2T/11860223",
    ]

    samples_per_dir = 3
    arw_files = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.ARW")))
        if files:
            # Pick evenly spaced samples
            step = max(1, len(files) // samples_per_dir)
            selected = files[::step][:samples_per_dir]
            arw_files.extend(selected)
            logger.info(f"  {d}: {len(files)} files, selected {len(selected)}")

    logger.info(f"Total test images: {len(arw_files)}")

    # =========================================================================
    # Run comparison on each image
    # =========================================================================
    all_results = []
    conf_threshold = 0.3

    for idx, arw_path in enumerate(arw_files):
        fname = os.path.basename(arw_path)
        logger.info(f"\n[{idx+1}/{len(arw_files)}] {fname}")

        # Load image
        pil_img = load_arw(arw_path)
        img_w, img_h = pil_img.size
        logger.info(f"  Size: {img_w}x{img_h}")

        # --- PyTorch inference ---
        pt_input = torch.from_numpy(
            np.array(pil_img.resize((resolution, resolution)))
        ).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            pt_out = wrapped_pt(pt_input)

        if pt_out[0].shape[-1] == 4:
            pt_boxes, pt_logits = pt_out[0].numpy(), pt_out[1].numpy()
        else:
            pt_logits, pt_boxes = pt_out[0].numpy(), pt_out[1].numpy()

        pt_dets = postprocess(pt_boxes, pt_logits, img_w, img_h, conf_threshold)

        # --- CoreML inference ---
        coreml_img = pil_img.resize((resolution, resolution))
        cm_pred = ml_model.predict({"image": coreml_img})
        cm_boxes, cm_logits = identify_outputs(cm_pred)
        cm_dets = postprocess(cm_boxes, cm_logits, img_w, img_h, conf_threshold)

        # --- Compare ---
        # Per-query box diff (all 300 queries, in pixels)
        box_diff_px = np.abs(pt_boxes - cm_boxes) * max(img_w, img_h)
        max_box_diff = box_diff_px.max()
        p50_box_diff = np.median(np.max(box_diff_px.reshape(300, 4), axis=-1))

        # Detection count
        pt_classes = [d["class"] for d in pt_dets]
        cm_classes = [d["class"] for d in cm_dets]

        # Compare top detections
        n_compare = min(len(pt_dets), len(cm_dets), 10)
        match_count = 0
        box_diffs = []
        for i in range(n_compare):
            if pt_dets[i]["query_idx"] == cm_dets[i]["query_idx"]:
                match_count += 1
            dx = abs(pt_dets[i]["cx"] - cm_dets[i]["cx"])
            dy = abs(pt_dets[i]["cy"] - cm_dets[i]["cy"])
            box_diffs.append(max(dx, dy))

        result = {
            "file": fname,
            "pt_count": len(pt_dets),
            "cm_count": len(cm_dets),
            "max_box_diff_px": float(max_box_diff),
            "p50_box_diff_px": float(p50_box_diff),
            "top_match": match_count,
            "top_n": n_compare,
            "pt_top_class": pt_classes[0] if pt_classes else "none",
            "cm_top_class": cm_classes[0] if cm_classes else "none",
        }
        all_results.append(result)

        # Print detections
        logger.info(f"  PyTorch: {len(pt_dets)} dets (conf>{conf_threshold})")
        for d in pt_dets[:5]:
            logger.info(f"    {d['class']:<15s} conf={d['score']:.3f} ({d['cx']:.0f},{d['cy']:.0f}) {d['w']:.0f}x{d['h']:.0f}")

        logger.info(f"  CoreML:  {len(cm_dets)} dets (conf>{conf_threshold})")
        for d in cm_dets[:5]:
            logger.info(f"    {d['class']:<15s} conf={d['score']:.3f} ({d['cx']:.0f},{d['cy']:.0f}) {d['w']:.0f}x{d['h']:.0f}")

        logger.info(f"  Max box diff (all queries): {max_box_diff:.2f} px")
        logger.info(f"  P50 box diff: {p50_box_diff:.4f} px")
        logger.info(f"  Top-{n_compare} query match: {match_count}/{n_compare}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: PyTorch vs CoreML FP32 on Bird Photos")
    print(f"{'='*70}")
    print(f"{'File':<18s} {'PT#':>3s} {'CM#':>3s} {'Top cls':<10s} {'Match?':<6s} "
          f"{'MaxΔ px':>8s} {'P50Δ px':>8s} {'TopN':>5s}")
    print("-" * 70)

    for r in all_results:
        cls_match = "YES" if r["pt_top_class"] == r["cm_top_class"] else "NO"
        print(f"{r['file']:<18s} {r['pt_count']:>3d} {r['cm_count']:>3d} "
              f"{r['pt_top_class']:<10s} {cls_match:<6s} "
              f"{r['max_box_diff_px']:>8.2f} {r['p50_box_diff_px']:>8.4f} "
              f"{r['top_match']}/{r['top_n']}")

    # Aggregates
    max_diffs = [r["max_box_diff_px"] for r in all_results]
    p50_diffs = [r["p50_box_diff_px"] for r in all_results]
    det_matches = all(r["pt_top_class"] == r["cm_top_class"] for r in all_results)
    print("-" * 70)
    print(f"{'Aggregate':<18s} {'':>3s} {'':>3s} {'':10s} "
          f"{'ALL' if det_matches else 'SOME':6s} "
          f"{max(max_diffs):>8.2f} {np.median(p50_diffs):>8.4f}")
    print()
    print(f"Total images: {len(all_results)}")
    print(f"All top-1 classes match: {'YES' if det_matches else 'NO'}")
    print(f"Max box diff across all images: {max(max_diffs):.2f} px")
    print(f"Median P50 box diff: {np.median(p50_diffs):.4f} px")


if __name__ == "__main__":
    main()
