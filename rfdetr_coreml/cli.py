#!/usr/bin/env python3
"""
CLI entry point for rfdetr-coreml.

Usage:
    rfdetr-coreml --model nano
    rfdetr-coreml --model nano --weights path/to/finetuned.pth
    rfdetr-coreml --model nano --batch-size 2
    rfdetr-coreml --model all --output-dir output
"""

import argparse
import logging
import sys
import time

# Apply patches before any rfdetr imports
import rfdetr_coreml  # noqa: F401

from rfdetr_coreml.export import MODEL_REGISTRY, export_to_coreml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for rfdetr-coreml command."""
    parser = argparse.ArgumentParser(description="Export RF-DETR to CoreML")
    parser.add_argument(
        "--model",
        type=str,
        default="nano",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model variant to export (default: nano)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp16", "fp32"],
        help="Compute precision (default: fp32). WARNING: fp16 has known "
             "precision issues with deformable attention.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom .pth weights (fine-tuned model). "
             "If not specified, uses pre-trained COCO weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the exported model (default: 1). "
             "batch=1 uses ImageType input; batch>1 uses TensorType (float32 NCHW [0,1]).",
    )
    args = parser.parse_args()

    if args.weights and args.model == "all":
        parser.error("--weights cannot be used with --model all")

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    results = {}
    for model_name in models:
        logger.info(f"{'='*60}")
        batch_desc = f", batch={args.batch_size}" if args.batch_size > 1 else ""
        logger.info(f"Exporting: {model_name} ({args.precision}{batch_desc})")
        logger.info(f"{'='*60}")
        t0 = time.time()
        try:
            path = export_to_coreml(
                model_name, args.output_dir, args.precision,
                weights_path=args.weights,
                batch_size=args.batch_size,
            )
            elapsed = time.time() - t0
            results[model_name] = (path, elapsed)
            logger.info(f"OK: {model_name} exported in {elapsed:.1f}s → {path}")
        except Exception as e:
            elapsed = time.time() - t0
            results[model_name] = (None, elapsed)
            logger.error(f"FAILED: {model_name} after {elapsed:.1f}s — {e}", exc_info=True)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Export Summary")
    logger.info(f"{'='*60}")
    all_ok = True
    for model_name, (path, elapsed) in results.items():
        status = f"OK → {path}" if path else "FAILED"
        if not path:
            all_ok = False
        logger.info(f"  {model_name:8s}: {status} ({elapsed:.1f}s)")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
