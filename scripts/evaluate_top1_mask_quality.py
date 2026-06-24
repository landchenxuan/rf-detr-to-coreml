#!/usr/bin/env python3
"""Compare final top-1 masks between two Core ML segmentation exports.

This evaluates product-facing mask behavior rather than raw tensor parity:
select a top query, resize its raw mask to the model input size, threshold it,
then compare the resulting binary masks with IoU, Dice, boundary distance,
area change, and centroid shift.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as importlib_metadata
import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt


SEGMENTATION_MODELS = [
    "seg-nano",
    "seg-small",
    "seg-medium",
    "seg-large",
    "seg-xlarge",
    "seg-2xlarge",
]

BIRD_CLASS_ID = 16


@dataclass(frozen=True)
class EvaluationGroup:
    name: str
    class_ids: list[int | None]
    skip_unconfident_pairs: bool = False


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_images(images_dir: Path) -> list[Path]:
    paths = sorted(images_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")
    return paths


def identify_outputs(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = logits = masks = None
    for value in result.values():
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[-1] == 4 and boxes is None:
            boxes = arr
        elif arr.ndim == 4 and masks is None:
            masks = arr
        elif arr.ndim >= 2 and arr.shape[-1] > 4 and logits is None:
            logits = arr

    missing = [
        name
        for name, arr in (("boxes", boxes), ("logits", logits), ("masks", masks))
        if arr is None
    ]
    if missing:
        raise RuntimeError(f"Missing Core ML output(s): {', '.join(missing)}")

    return boxes[0], logits[0], masks[0]


def input_size(mlmodel: Any) -> tuple[int, int]:
    spec = mlmodel.get_spec()
    image_type = spec.description.input[0].type.imageType
    return int(image_type.width), int(image_type.height)


def class_mapping(mlmodel: Any) -> dict[int, str]:
    raw = mlmodel.user_defined_metadata.get("class_mapping", "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {int(k): str(v) for k, v in parsed.items()}


def resolve_torch_device(device: str) -> str:
    import torch

    if device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS requested but torch.backends.mps is not available")
    return device


def build_pytorch_model(model: str, device: str) -> tuple[Any, int]:
    import rfdetr_coreml  # noqa: F401
    from rfdetr_coreml.export import MODEL_REGISTRY, NormalizedWrapper, _import_model_class

    model_cls = _import_model_class(model)
    resolution = MODEL_REGISTRY[model][1]
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped = NormalizedWrapper(pt_model, resolution).eval().to(device)
    del rfdetr_model
    return wrapped, resolution


def run_pytorch(model: Any, image: Image.Image, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    image_np = np.asarray(image).copy()
    x = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = x.to(device)
    with torch.no_grad():
        output = model(x)
    if device == "mps":
        torch.mps.synchronize()
    boxes = output[0].detach().cpu().numpy()[0]
    logits = output[1].detach().cpu().numpy()[0]
    masks = output[2].detach().cpu().numpy()[0]
    return boxes, logits, masks


def select_top1_for_class(logits: np.ndarray, class_id: int) -> dict[str, Any]:
    query = int(np.argmax(logits[:, class_id]))
    logit = float(logits[query, class_id])
    return {
        "query": query,
        "class_id": class_id,
        "logit": logit,
        "score": float(sigmoid(logit)),
        "confident": bool(logit > 0),
    }


def select_top1_any(logits: np.ndarray) -> dict[str, Any]:
    query, class_id = [int(v) for v in np.unravel_index(np.argmax(logits), logits.shape)]
    logit = float(logits[query, class_id])
    return {
        "query": query,
        "class_id": class_id,
        "logit": logit,
        "score": float(sigmoid(logit)),
        "confident": bool(logit > 0),
    }


def resolve_class_selector(selector: str, label_map: dict[int, str]) -> int:
    if selector.isdigit():
        class_id = int(selector)
        if label_map and class_id not in label_map:
            raise ValueError(f"Class id {class_id} is not present in the model label map")
        return class_id

    normalized = selector.lower()
    for class_id, class_name in label_map.items():
        if class_name.lower() == normalized:
            return class_id

    if normalized == "bird":
        return BIRD_CLASS_ID

    raise ValueError(f"Unknown class selector: {selector}")


def build_evaluation_groups(
    class_selectors: list[str],
    include_any: bool,
    label_map: dict[int, str],
) -> list[EvaluationGroup]:
    groups: list[EvaluationGroup] = []
    if class_selectors:
        for selector in class_selectors:
            class_id = resolve_class_selector(selector, label_map)
            groups.append(EvaluationGroup(label_map.get(class_id, f"class_{class_id}"), [class_id]))
    else:
        if not label_map:
            raise ValueError("Default all-class evaluation requires class_mapping metadata in the model package")
        groups.append(
            EvaluationGroup("all_classes", sorted(label_map), skip_unconfident_pairs=True)
        )

    if include_any:
        groups.append(EvaluationGroup("any", [None]))

    return groups


def resize_raw_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.float32))
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def binary_mask(raw_mask: np.ndarray, size: tuple[int, int], threshold_logit: float) -> np.ndarray:
    return resize_raw_mask(raw_mask, size) > threshold_logit


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask, eroded)


def boundary_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float | None]:
    boundary_a = boundary(mask_a)
    boundary_b = boundary(mask_b)
    if not boundary_a.any() and not boundary_b.any():
        return {"boundary_mean_px": 0.0, "boundary_p95_px": 0.0, "boundary_max_px": 0.0}
    if not boundary_a.any() or not boundary_b.any():
        return {"boundary_mean_px": None, "boundary_p95_px": None, "boundary_max_px": None}

    dist_to_b = distance_transform_edt(~boundary_b)
    dist_to_a = distance_transform_edt(~boundary_a)
    distances = np.concatenate([dist_to_b[boundary_a], dist_to_a[boundary_b]])
    return {
        "boundary_mean_px": float(np.mean(distances)),
        "boundary_p95_px": float(np.percentile(distances, 95)),
        "boundary_max_px": float(np.max(distances)),
    }


def compare_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, Any]:
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    area_sum = area_a + area_b

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union
    if area_sum == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection / area_sum

    centroid_a = centroid(mask_a)
    centroid_b = centroid(mask_b)
    if centroid_a is None or centroid_b is None:
        centroid_shift_px = None
    else:
        centroid_shift_px = math.dist(centroid_a, centroid_b)

    if area_a == 0:
        area_change_pct = None
    else:
        area_change_pct = (area_b - area_a) / area_a * 100.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "baseline_area_px": area_a,
        "candidate_area_px": area_b,
        "area_change_pct": None if area_change_pct is None else float(area_change_pct),
        "centroid_shift_px": None if centroid_shift_px is None else float(centroid_shift_px),
        **boundary_distances(mask_a, mask_b),
    }


def finite(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = finite([r["iou"] for r in rows])
    dice_scores = finite([r["dice"] for r in rows])

    confident_rows = [r for r in rows if r["baseline_confident"]]
    confident_ious = finite([r["iou"] for r in confident_rows])
    confident_dice = finite([r["dice"] for r in confident_rows])
    confident_boundary_p95 = finite([r["boundary_p95_px"] for r in confident_rows])
    confident_centroid_shifts = finite([r["centroid_shift_px"] for r in confident_rows])
    confident_area_changes = finite([r["area_change_pct"] for r in confident_rows])

    return {
        "n_cases": len(rows),
        "n_images": len({r["image"] for r in rows}),
        "baseline_confident_cases": len(confident_rows),
        "query_changed": sum(1 for r in rows if r["query_changed"]),
        "class_changed": sum(1 for r in rows if r["class_changed"]),
        "confidence_state_changed": sum(1 for r in rows if r["confidence_state_changed"]),
        "mean_iou": mean(ious) if ious else None,
        "median_iou": median(ious) if ious else None,
        "min_iou": min(ious) if ious else None,
        "mean_dice": mean(dice_scores) if dice_scores else None,
        "median_dice": median(dice_scores) if dice_scores else None,
        "min_dice": min(dice_scores) if dice_scores else None,
        "confident_mean_iou": mean(confident_ious) if confident_ious else None,
        "confident_min_iou": min(confident_ious) if confident_ious else None,
        "confident_mean_dice": mean(confident_dice) if confident_dice else None,
        "confident_min_dice": min(confident_dice) if confident_dice else None,
        "confident_mean_boundary_p95_px": mean(confident_boundary_p95) if confident_boundary_p95 else None,
        "confident_max_boundary_p95_px": max(confident_boundary_p95) if confident_boundary_p95 else None,
        "confident_mean_centroid_shift_px": (
            mean(confident_centroid_shifts) if confident_centroid_shifts else None
        ),
        "confident_max_centroid_shift_px": (
            max(confident_centroid_shifts) if confident_centroid_shifts else None
        ),
        "confident_mean_abs_area_change_pct": (
            mean([abs(v) for v in confident_area_changes]) if confident_area_changes else None
        ),
        "confident_max_abs_area_change_pct": (
            max([abs(v) for v in confident_area_changes]) if confident_area_changes else None
        ),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def model_package(directory: Path, model: str, precision: str) -> Path:
    path = directory / f"rf-detr-{model}-{precision}.mlpackage"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def evaluate_model(
    model: str,
    baseline_kind: str,
    baseline_dir: Path,
    candidate_dir: Path,
    images: list[Path],
    class_selectors: list[str],
    include_any: bool,
    threshold_logit: float,
    compute_unit: str,
    torch_device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import coremltools as ct

    compute_units = {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[compute_unit]

    candidate_path = model_package(candidate_dir, model, "fp16")
    logging.info("Loading %s", candidate_path)
    candidate = ct.models.MLModel(str(candidate_path), compute_units=compute_units)
    label_map = class_mapping(candidate)

    baseline: Any | None = None
    pytorch_model: Any | None = None
    baseline_label: str
    resolved_torch_device: str | None = None
    if baseline_kind == "coreml":
        baseline_path = model_package(baseline_dir, model, "fp32")
        logging.info("Loading %s", baseline_path)
        baseline = ct.models.MLModel(str(baseline_path), compute_units=compute_units)
        size = input_size(baseline)
        if input_size(candidate) != size:
            raise RuntimeError(f"Input size mismatch for {model}: {size} vs {input_size(candidate)}")
        label_map = class_mapping(baseline)
        baseline_label = str(baseline_path)
    elif baseline_kind == "pytorch":
        resolved_torch_device = resolve_torch_device(torch_device)
        logging.info("Loading PyTorch %s reference on %s", model, resolved_torch_device)
        pytorch_model, resolution = build_pytorch_model(model, resolved_torch_device)
        size = (resolution, resolution)
        if input_size(candidate) != size:
            raise RuntimeError(f"Input size mismatch for {model}: {size} vs {input_size(candidate)}")
        try:
            rfdetr_version = importlib_metadata.version("rfdetr")
        except importlib_metadata.PackageNotFoundError:
            rfdetr_version = "unknown"
        baseline_label = f"pytorch:rfdetr=={rfdetr_version}:{resolved_torch_device}"
    else:
        raise ValueError(f"Unknown baseline kind: {baseline_kind}")

    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    groups = build_evaluation_groups(
        class_selectors=class_selectors,
        include_any=include_any,
        label_map=label_map,
    )

    for image_path in images:
        image = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        if baseline_kind == "coreml":
            baseline_boxes, baseline_logits, baseline_masks = identify_outputs(baseline.predict({"image": image}))
        else:
            baseline_boxes, baseline_logits, baseline_masks = run_pytorch(
                pytorch_model, image, resolved_torch_device
            )
        candidate_boxes, candidate_logits, candidate_masks = identify_outputs(candidate.predict({"image": image}))

        if baseline_masks.shape != candidate_masks.shape:
            raise RuntimeError(
                f"Mask shape mismatch for {model} on {image_path.name}: "
                f"{baseline_masks.shape} vs {candidate_masks.shape}"
            )
        if baseline_logits.shape != candidate_logits.shape:
            raise RuntimeError(
                f"Logit shape mismatch for {model} on {image_path.name}: "
                f"{baseline_logits.shape} vs {candidate_logits.shape}"
            )
        if baseline_boxes.shape != candidate_boxes.shape:
            raise RuntimeError(
                f"Box shape mismatch for {model} on {image_path.name}: "
                f"{baseline_boxes.shape} vs {candidate_boxes.shape}"
            )

        for group in groups:
            for target_class_id in group.class_ids:
                if target_class_id is None:
                    base_top = select_top1_any(baseline_logits)
                    cand_top = select_top1_any(candidate_logits)
                    target_name = group.name
                else:
                    base_top = select_top1_for_class(baseline_logits, target_class_id)
                    cand_top = select_top1_for_class(candidate_logits, target_class_id)
                    target_name = label_map.get(target_class_id, f"class_{target_class_id}")
                    if group.skip_unconfident_pairs and not base_top["confident"] and not cand_top["confident"]:
                        continue

                base_mask = binary_mask(baseline_masks[base_top["query"]], size, threshold_logit)
                cand_mask = binary_mask(candidate_masks[cand_top["query"]], size, threshold_logit)
                metrics = compare_masks(base_mask, cand_mask)

                row = {
                    "model": model,
                    "scope": group.name,
                    "target_class_id": target_class_id,
                    "target_class_name": target_name,
                    "image": image_path.name,
                    "input_size": list(size),
                    "mask_shape": list(baseline_masks.shape),
                    "baseline_query": base_top["query"],
                    "candidate_query": cand_top["query"],
                    "baseline_class_id": base_top["class_id"],
                    "candidate_class_id": cand_top["class_id"],
                    "baseline_class_name": label_map.get(base_top["class_id"], f"class_{base_top['class_id']}"),
                    "candidate_class_name": label_map.get(cand_top["class_id"], f"class_{cand_top['class_id']}"),
                    "baseline_logit": base_top["logit"],
                    "candidate_logit": cand_top["logit"],
                    "baseline_score": base_top["score"],
                    "candidate_score": cand_top["score"],
                    "baseline_confident": base_top["confident"],
                    "candidate_confident": cand_top["confident"],
                    "query_changed": base_top["query"] != cand_top["query"],
                    "class_changed": base_top["class_id"] != cand_top["class_id"],
                    "confidence_state_changed": base_top["confident"] != cand_top["confident"],
                    **metrics,
                }
                details.append(row)

    for group in groups:
        target_rows = [r for r in details if r["scope"] == group.name]
        summary = {
            "model": model,
            "scope": group.name,
            "baseline_kind": baseline_kind,
            "baseline": baseline_label,
            "candidate_package": str(candidate_path),
            "threshold_logit": threshold_logit,
            "compute_unit": compute_unit,
            "torch_device": resolved_torch_device,
            "input_size": list(size),
            **summarize_rows(target_rows),
        }
        summaries.append(summary)

    del baseline, pytorch_model, candidate
    gc.collect()
    return summaries, details


def write_markdown(path: Path, summaries: list[dict[str, Any]], details: list[dict[str, Any]]) -> None:
    lines = [
        "# Top-1 Mask Quality Report",
        "",
        "Baseline and candidate are recorded per summary row.",
        "Candidate defaults to the FP16 Core ML package.",
        "Masks are resized to model input size and thresholded at raw logit > 0.",
        "",
        "## Summary",
        "",
        "| Model | Scope | Accepted/Cases | Conf flip | Conf mean IoU | Conf min IoU | Conf mean Dice | Conf mean boundary p95 px | Conf max boundary p95 px | Conf mean centroid px | Conf max centroid px | Conf mean abs area % | Conf max abs area % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["scope"],
                    f"{row['baseline_confident_cases']}/{row['n_cases']}",
                    str(row["confidence_state_changed"]),
                    fmt(row["confident_mean_iou"]),
                    fmt(row["confident_min_iou"]),
                    fmt(row["confident_mean_dice"]),
                    fmt(row["confident_mean_boundary_p95_px"], 2),
                    fmt(row["confident_max_boundary_p95_px"], 2),
                    fmt(row["confident_mean_centroid_shift_px"], 2),
                    fmt(row["confident_max_centroid_shift_px"], 2),
                    fmt(row["confident_mean_abs_area_change_pct"], 2),
                    fmt(row["confident_max_abs_area_change_pct"], 2),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Worst Images By IoU", ""])
    for summary in summaries:
        target_rows = [
            r for r in details if r["model"] == summary["model"] and r["scope"] == summary["scope"]
        ]
        worst_pool = [r for r in target_rows if r["baseline_confident"]]
        worst = sorted(worst_pool or target_rows, key=lambda r: r["iou"])[:5]
        lines.extend([f"### {summary['model']} / {summary['scope']}", ""])
        lines.append(
            "| Image | Class | IoU | Dice | Baseline | Candidate | Score | Boundary p95 px | Area change % | Centroid px |"
        )
        lines.append("| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |")
        for row in worst:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["image"],
                        row["target_class_name"],
                        fmt(row["iou"]),
                        fmt(row["dice"]),
                        f"{row['baseline_class_name']} q{row['baseline_query']}",
                        f"{row['candidate_class_name']} q{row['candidate_query']}",
                        f"{row['baseline_score']:.3f}->{row['candidate_score']:.3f}",
                        fmt(row["boundary_p95_px"], 2),
                        fmt(row["area_change_pct"], 2),
                        fmt(row["centroid_shift_px"], 2),
                    ]
                )
                + " |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("output/rfdetr-1-8-1-fp32-acceptance"))
    parser.add_argument("--baseline-kind", choices=["coreml", "pytorch"], default="coreml")
    parser.add_argument("--candidate-dir", type=Path, default=Path("output/rfdetr-1-8-1-fp16-rerun"))
    parser.add_argument("--images-dir", type=Path, default=Path("scripts/test_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/top1-mask-quality-rfdetr-1-8-1"))
    parser.add_argument("--model", action="append", choices=SEGMENTATION_MODELS)
    parser.add_argument(
        "--class",
        dest="target_classes",
        action="append",
        metavar="NAME_OR_ID",
        help="Evaluate one concrete class by COCO name or class id. Omit to run the all-class sweep.",
    )
    parser.add_argument(
        "--top1-any",
        action="store_true",
        help="Also evaluate the single highest-scoring query/class pair across the full logit tensor.",
    )
    parser.add_argument("--threshold-logit", type=float, default=0.0)
    parser.add_argument("--torch-device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--compute-unit",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        default="all",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    models = args.model or SEGMENTATION_MODELS
    target_classes = args.target_classes or []
    images = load_images(args.images_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    for model in models:
        summaries, details = evaluate_model(
            model=model,
            baseline_kind=args.baseline_kind,
            baseline_dir=args.baseline_dir,
            candidate_dir=args.candidate_dir,
            images=images,
            class_selectors=target_classes,
            include_any=args.top1_any,
            threshold_logit=args.threshold_logit,
            compute_unit=args.compute_unit,
            torch_device=args.torch_device,
        )
        all_summaries.extend(summaries)
        all_details.extend(details)

    summary_path = args.output_dir / "summary.json"
    details_path = args.output_dir / "details.jsonl"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as f:
        for row in all_details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_markdown(report_path, all_summaries, all_details)

    print(f"Wrote {summary_path}")
    print(f"Wrote {details_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
