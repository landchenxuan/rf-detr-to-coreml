# Changelog

All notable changes to this project are recorded here.

## Unreleased

### Changed

- Target released PyPI packages for the compatibility baseline:
  `coremltools==9.0`, `rfdetr==1.8.3`, `torch==2.7.0`, and
  `torchvision==0.22.0`.
- Clarified that the package supports Python 3.10 while the exact benchmark
  dependency pins require Python 3.11 or newer.
- Simplified the README around install, export, verification, production notes,
  and the patch overlay.
- Updated export documentation for RF-DETR's official ONNX/TFLite paths and
  switched the ONNX benchmark helper to RF-DETR's official ONNX exporter.
- Removed deprecated upstream variants from the supported model registry.
- Refreshed the README latency, diff, FP16, and ONNX benchmark snapshots for
  RF-DETR 1.8.1 on Apple M5 Pro 18-core using real test images.
- Added pinned ONNX benchmark reproduction dependencies to `requirements.txt`.
- Added a README ONNX detection benchmark table and aligned ONNX benchmark box
  diffs with the confident-query accuracy checks used by `test_export.py`.
- Updated FP16 guidance from raw-query diffing to detection-oriented equivalence:
  compare confident reference detections, class argmax changes, and
  confidence-state changes.
- Documented FP16 detection deltas for nano, small, medium, and large on
  Apple M5 Pro without treating the limited snapshot as a support matrix.
- Added FP16 Core ML compute-unit latency notes for `ALL`, `CPU_AND_GPU`, and
  `CPU_AND_NE` so the Neural Engine path is not inferred from `ALL` latency.

### Fixed

- Updated the detection `large` registry entry to `RFDETRLarge`, matching
  released `rfdetr` 1.8.1.
- Recorded installed `rfdetr` and `coremltools` versions in Core ML metadata
  instead of hardcoding the stale RF-DETR 1.5.1 version.
- Supported current RF-DETR segmentation export by replacing the export-time
  depthwise-conv custom autograd path with a Core ML-convertible equivalent.
- Kept the released `coremltools` 9.0 meshgrid workaround in place.
- Corrected the FP16 validation narrative that treated low-confidence unmatched
  queries as stable detections.

### Added

- Added a fast no-download CI smoke test for dependency imports, CLI wiring,
  and model registry resolution.
- Added explicit Core ML vs PyTorch reference verification commands.
- Added PR hygiene through a repository pull request template.
- Added Core ML metadata for class names, class IDs, and class ID-to-name
  mapping.
- Added a YOLO-compatible `names` Core ML metadata alias for tools that expect
  dense output-index class labels.
- Added `scripts/scan_fp16_precision.py` for mixed-precision scan experiments
  and use `scripts/test_export.py --precision fp16` as the FP16 validation
  entrypoint.
- Extended `scripts/benchmark_latency.py` with `--precision` and GPU/NE compute
  unit timing, and added `scripts/test_export.py --compute-unit` for target
  validation under a specific Core ML compute unit.
- Updated `scripts/validate_coreml.swift` to inspect `ALL`, `CPU_AND_GPU`,
  `CPU_AND_NE`, and `CPU_ONLY`.

## 0.1.0

### Added

- Initial RF-DETR to Core ML export package.
- Detection and segmentation model export through `rfdetr-coreml`.
- Runtime patch overlay for RF-DETR/Core ML conversion gaps.
- Benchmark, precision, batch, and native Core ML validation scripts.
