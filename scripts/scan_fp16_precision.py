#!/usr/bin/env python3
"""Scan RF-DETR Core ML mixed-precision candidates.

This is an experimental harness for finding which MIL op groups can be run
through coremltools' FP16ComputePrecision transform without changing RF-DETR
outputs beyond configured thresholds.
"""

import argparse
import gc
import glob
import json
import logging
import os
import time
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

import rfdetr_coreml  # noqa: F401  # apply conversion patches before rfdetr imports
from rfdetr_coreml.export import MODEL_REGISTRY, NormalizedWrapper, _import_model_class

logger = logging.getLogger(__name__)

TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    selector: Callable | None


def input_names(op) -> list[str]:
    names = []
    for value in op.inputs.values():
        values = value if isinstance(value, (list, tuple)) else [value]
        for var in values:
            name = getattr(var, "name", None)
            if name:
                names.append(name)
    return names


def has_input_name(op, needle: str) -> bool:
    return any(needle in name for name in input_names(op))


def has_any_input_name(op, needles: tuple[str, ...]) -> bool:
    names = input_names(op)
    return any(any(needle in name for needle in needles) for name in names)


def is_backbone_encoder_weight(op) -> bool:
    return has_input_name(op, "model.backbone.0.encoder.encoder.encoder.layer.")


def is_projector_weight(op) -> bool:
    return has_input_name(op, "model.backbone.0.projector.")


def strategies() -> list[Strategy]:
    return [
        Strategy("fp32", "FP32 Core ML control", None),
        Strategy("full_fp16", "All valid ops through FP16ComputePrecision", lambda op: True),
        Strategy("conv_all", "All convolution ops", lambda op: op.op_type == "conv"),
        Strategy(
            "conv_projector",
            "YOLO-style projector convolutions only",
            lambda op: op.op_type == "conv" and is_projector_weight(op),
        ),
        Strategy(
            "conv_patch_embed",
            "DinoV2 patch-embedding convolution only",
            lambda op: op.op_type == "conv" and has_input_name(op, "patch_embeddings.projection."),
        ),
        Strategy("linear_all", "All linear ops", lambda op: op.op_type == "linear"),
        Strategy(
            "linear_backbone_mlp",
            "DinoV2 backbone MLP linear ops",
            lambda op: op.op_type == "linear"
            and is_backbone_encoder_weight(op)
            and has_input_name(op, ".mlp."),
        ),
        Strategy(
            "linear_backbone_attn_qkv",
            "DinoV2 backbone attention query/key/value linear ops",
            lambda op: op.op_type == "linear"
            and is_backbone_encoder_weight(op)
            and has_any_input_name(op, (".attention.attention.query.", ".attention.attention.key.", ".attention.attention.value.")),
        ),
        Strategy(
            "linear_backbone_attn_out",
            "DinoV2 backbone attention output linear ops",
            lambda op: op.op_type == "linear"
            and is_backbone_encoder_weight(op)
            and has_input_name(op, ".attention.output.dense."),
        ),
        Strategy(
            "linear_transformer_encoder_heads",
            "Transformer encoder proposal/class/bbox linear ops",
            lambda op: op.op_type == "linear"
            and has_any_input_name(op, ("model.transformer.enc_output.", "model.transformer.enc_out_")),
        ),
        Strategy(
            "linear_decoder_cross_value",
            "Decoder deformable cross-attention value_proj linear ops",
            lambda op: op.op_type == "linear" and has_input_name(op, ".cross_attn.value_proj."),
        ),
        Strategy(
            "linear_decoder_cross_offsets",
            "Decoder deformable cross-attention sampling_offsets linear ops",
            lambda op: op.op_type == "linear" and has_input_name(op, ".cross_attn.sampling_offsets."),
        ),
        Strategy(
            "linear_decoder_cross_attention_weights",
            "Decoder deformable cross-attention attention_weights linear ops",
            lambda op: op.op_type == "linear" and has_input_name(op, ".cross_attn.attention_weights."),
        ),
        Strategy(
            "linear_decoder_cross_output",
            "Decoder deformable cross-attention output_proj linear ops",
            lambda op: op.op_type == "linear" and has_input_name(op, ".cross_attn.output_proj."),
        ),
        Strategy(
            "linear_decoder_ffn",
            "Decoder feed-forward linear1/linear2 ops",
            lambda op: op.op_type == "linear"
            and has_any_input_name(op, (".decoder.layers.0.linear", ".decoder.layers.1.linear")),
        ),
        Strategy(
            "linear_bbox_head",
            "Final bbox head linear ops",
            lambda op: op.op_type == "linear" and has_input_name(op, "model.bbox_embed."),
        ),
        Strategy(
            "linear_class_head",
            "Final class head linear op",
            lambda op: op.op_type == "linear" and has_input_name(op, "model.class_embed."),
        ),
        Strategy("matmul_all", "All matmul ops", lambda op: op.op_type == "matmul"),
        Strategy("softmax_all", "All softmax ops", lambda op: op.op_type == "softmax"),
        Strategy("gelu_all", "All GELU ops", lambda op: op.op_type == "gelu"),
        Strategy("relu_all", "All ReLU ops", lambda op: op.op_type == "relu"),
        Strategy("silu_all", "All SiLU ops", lambda op: op.op_type == "silu"),
        Strategy("layer_norm_all", "All layer_norm ops", lambda op: op.op_type == "layer_norm"),
        Strategy(
            "layer_norm_projector",
            "Projector batchnorm-as-layer_norm ops",
            lambda op: op.op_type == "layer_norm" and is_projector_weight(op),
        ),
        Strategy(
            "layer_norm_decoder",
            "Decoder layer_norm ops",
            lambda op: op.op_type == "layer_norm" and has_input_name(op, "model.transformer.decoder."),
        ),
        Strategy("resample_all", "All deformable-attention resample ops", lambda op: op.op_type == "resample"),
        Strategy(
            "deformable_cross_value_resample",
            "Cross-attention value projection plus resample",
            lambda op: (
                (op.op_type == "linear" and has_input_name(op, ".cross_attn.value_proj."))
                or op.op_type == "resample"
            ),
        ),
        Strategy(
            "heads_bbox_class",
            "Final bbox and class heads together",
            lambda op: op.op_type == "linear"
            and has_any_input_name(op, ("model.bbox_embed.", "model.class_embed.")),
        ),
        Strategy(
            "conservative_first_guess",
            "Conservative non-coordinate candidate set",
            lambda op: (
                (op.op_type == "conv" and is_projector_weight(op))
                or (
                    op.op_type == "linear"
                    and (
                        (is_backbone_encoder_weight(op) and has_input_name(op, ".mlp."))
                        or has_input_name(op, ".cross_attn.value_proj.")
                        or has_input_name(op, ".cross_attn.output_proj.")
                    )
                )
                or op.op_type in {"gelu", "relu", "silu"}
            ),
        ),
    ]


def load_test_images(limit: int | None = None) -> list[str]:
    paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No test images found in {TEST_IMAGES_DIR}")
    return paths


def build_models(model_name: str):
    resolution = MODEL_REGISTRY[model_name][1]
    model_cls = _import_model_class(model_name)
    rfdetr_model = model_cls()
    pt_model = deepcopy(rfdetr_model.model.model).cpu().eval()
    pt_model.export()
    wrapped = NormalizedWrapper(pt_model, resolution).eval()
    del rfdetr_model

    dummy = torch.rand(1, 3, resolution, resolution)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy)
    return resolution, wrapped, traced


def build_mil_ops(traced, resolution):
    import coremltools as ct

    prog = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, resolution, resolution), scale=1.0 / 255.0)],
        convert_to="milinternal",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )
    ops = []
    for function in prog.functions.values():
        ops.extend(function.operations)
    return ops


def compute_references(wrapped, image_paths, resolution):
    from PIL import Image

    refs = []
    for image_path in image_paths:
        pil_img = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        img_np = np.array(pil_img)
        pt_input = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            pt_out = wrapped(pt_input)
        refs.append({
            "path": image_path,
            "image": pil_img,
            "boxes": pt_out[0].detach().cpu().numpy()[0],
            "logits": pt_out[1].detach().cpu().numpy()[0],
        })
    return refs


def identify_coreml_outputs(result):
    boxes = logits = None
    for value in result.values():
        arr = np.array(value)
        if arr.ndim >= 2 and arr.shape[-1] == 4 and boxes is None:
            boxes = arr
        elif arr.ndim >= 2 and arr.shape[-1] > 4 and logits is None:
            logits = arr
    if boxes is None or logits is None:
        raise RuntimeError("Could not identify Core ML boxes/logits outputs")
    return boxes[0], logits[0]


def export_strategy(traced, resolution, strategy: Strategy, output_path: str, reuse: bool):
    import coremltools as ct

    if reuse and os.path.exists(output_path):
        return output_path, 0.0, True

    if strategy.selector is None:
        compute_precision = ct.precision.FLOAT32
    elif strategy.name == "full_fp16":
        compute_precision = ct.precision.FLOAT16
    else:
        compute_precision = ct.transform.FP16ComputePrecision(op_selector=strategy.selector)

    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, resolution, resolution), scale=1.0 / 255.0)],
        convert_to="mlprogram",
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(output_path)
    return output_path, time.perf_counter() - t0, False


def evaluate_model(mlpackage_path, refs, resolution, latency_runs: int):
    import coremltools as ct

    ml_model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
    max_box_diff_px = 0.0
    max_logit_diff = 0.0
    max_score_diff = 0.0
    total_confident = 0
    confidence_state_changes = 0
    class_argmax_changes = 0
    per_image = []
    latencies = []

    for ref in refs:
        for _ in range(2):
            ml_model.predict({"image": ref["image"]})

        t0 = time.perf_counter()
        result = None
        for _ in range(latency_runs):
            result = ml_model.predict({"image": ref["image"]})
        latencies.append((time.perf_counter() - t0) / latency_runs * 1000)

        cm_boxes, cm_logits = identify_coreml_outputs(result)
        ref_scores = ref["logits"].max(axis=1)
        cm_scores = cm_logits.max(axis=1)
        confident = ref_scores > 0
        cm_confident = cm_scores > 0
        confidence_changes = int(np.count_nonzero(confident != cm_confident))
        n_conf = int(confident.sum())
        total_confident += n_conf
        confidence_state_changes += confidence_changes

        if n_conf:
            box_diff_px = float(np.abs(ref["boxes"][confident] - cm_boxes[confident]).max()) * resolution
            logit_diff = float(np.abs(ref["logits"][confident] - cm_logits[confident]).max())
            score_diff = float(np.abs(sigmoid(ref_scores[confident]) - sigmoid(cm_scores[confident])).max())
            class_changes = int(np.count_nonzero(
                np.argmax(ref["logits"][confident], axis=1) != np.argmax(cm_logits[confident], axis=1)
            ))
        else:
            box_diff_px = 0.0
            logit_diff = 0.0
            score_diff = 0.0
            class_changes = 0

        max_box_diff_px = max(max_box_diff_px, box_diff_px)
        max_logit_diff = max(max_logit_diff, logit_diff)
        max_score_diff = max(max_score_diff, score_diff)
        class_argmax_changes += class_changes
        per_image.append({
            "image": os.path.basename(ref["path"]),
            "confident": n_conf,
            "confidence_state_changes": confidence_changes,
            "class_argmax_changes": class_changes,
            "box_diff_px": box_diff_px,
            "logit_diff": logit_diff,
            "score_diff": score_diff,
        })

    del ml_model
    gc.collect()
    return {
        "max_box_diff_px": max_box_diff_px,
        "max_logit_diff": max_logit_diff,
        "max_score_diff": max_score_diff,
        "median_latency_ms": float(np.median(latencies)),
        "total_confident": total_confident,
        "confidence_state_changes": confidence_state_changes,
        "class_argmax_changes": class_argmax_changes,
        "per_image": per_image,
    }


def selected_op_counts(strategy: Strategy, ops) -> dict[str, int]:
    if strategy.selector is None:
        return {}
    counts = Counter()
    for op in ops:
        try:
            if strategy.selector(op):
                counts[op.op_type] += 1
        except Exception:
            pass
    return dict(sorted(counts.items()))


def write_summary(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Strategy | Threshold result | Box px | Score diff | Class changes | Conf changes | Latency ms | Selected ops | Description |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in rows:
            counts = ",".join(f"{k}:{v}" for k, v in row["selected_ops"].items()) or "-"
            f.write(
                f"| {row['strategy']} | {row['threshold_result']} | "
                f"{row.get('max_box_diff_px', float('nan')):.4f} | "
                f"{row.get('max_score_diff', float('nan')):.6f} | "
                f"{row.get('class_argmax_changes', 0)} | "
                f"{row.get('confidence_state_changes', 0)} | "
                f"{row.get('median_latency_ms', float('nan')):.2f} | "
                f"{counts} | {row['description']} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Scan Core ML FP16 precision candidates")
    parser.add_argument("--model", default="nano", choices=list(MODEL_REGISTRY))
    parser.add_argument("--output-dir", default="output/fp16-scan")
    parser.add_argument("--images", type=int, help="Limit number of test images")
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--max-box-diff-px", type=float, default=2.0)
    parser.add_argument("--strategy", action="append", help="Run only this strategy name; may be repeated")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing mlpackages")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore")

    os.makedirs(args.output_dir, exist_ok=True)
    all_strategies = strategies()
    if args.strategy:
        wanted = set(args.strategy)
        all_strategies = [s for s in all_strategies if s.name in wanted]
        missing = wanted - {s.name for s in all_strategies}
        if missing:
            raise SystemExit(f"Unknown strategy name(s): {sorted(missing)}")

    resolution, wrapped, traced = build_models(args.model)
    ops = build_mil_ops(traced, resolution)
    refs = compute_references(wrapped, load_test_images(args.images), resolution)
    del wrapped
    gc.collect()

    jsonl_path = os.path.join(args.output_dir, f"{args.model}-results.jsonl")
    summary_path = os.path.join(args.output_dir, f"{args.model}-summary.md")
    rows = []

    with open(jsonl_path, "a", encoding="utf-8") as jsonl:
        for strategy in all_strategies:
            mlpackage_path = os.path.join(args.output_dir, f"rf-detr-{args.model}-{strategy.name}.mlpackage")
            selected_counts = selected_op_counts(strategy, ops)
            row = {
                "model": args.model,
                "strategy": strategy.name,
                "description": strategy.description,
                "resolution": resolution,
                "thresholds": {
                    "max_box_diff_px": args.max_box_diff_px,
                    "class_argmax_changes": 0,
                    "confidence_state_changes": 0,
                },
                "selected_ops": selected_counts,
            }
            try:
                _, export_seconds, reused = export_strategy(traced, resolution, strategy, mlpackage_path, args.reuse)
                metrics = evaluate_model(mlpackage_path, refs, resolution, args.latency_runs)
                row.update(metrics)
                row["export_seconds"] = export_seconds
                row["reused"] = reused
                row["threshold_result"] = (
                    "within_thresholds"
                    if metrics["max_box_diff_px"] <= args.max_box_diff_px
                    and metrics["class_argmax_changes"] == 0
                    and metrics["confidence_state_changes"] == 0
                    else "exceeds_thresholds"
                )
            except Exception as exc:
                row["threshold_result"] = "error"
                row["error"] = repr(exc)
            rows.append(row)
            jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl.flush()
            print(
                f"{row['strategy']:<38s} {row['threshold_result']:<18s} "
                f"box={row.get('max_box_diff_px', float('nan')):8.3f}px "
                f"score={row.get('max_score_diff', float('nan')):9.6f} "
                f"class={row.get('class_argmax_changes', 0):3d} "
                f"conf={row.get('confidence_state_changes', 0):3d} "
                f"lat={row.get('median_latency_ms', float('nan')):7.2f}ms "
                f"ops={row['selected_ops']}"
            )

    write_summary(summary_path, rows)
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
