# Changelog

All notable changes to this project are recorded here.

## Unreleased

### Changed

- Target released PyPI packages for the compatibility baseline:
  `coremltools==9.0`, `rfdetr==1.7.1`, `torch==2.7.0`, and
  `torchvision==0.22.0`.
- Simplified the README around install, export, verification, production notes,
  and the patch overlay.
- Updated export documentation for RF-DETR 1.7's official ONNX/TFLite paths and
  clarified that this repository's ONNX scripts are legacy comparison tools.

### Fixed

- Updated the detection `large` registry entry to `RFDETRLarge`, matching
  released `rfdetr` 1.7.1.
- Recorded installed `rfdetr` and `coremltools` versions in Core ML metadata
  instead of hardcoding the stale RF-DETR 1.5.1 version.
- Supported RF-DETR 1.7.x segmentation export by replacing the export-time
  depthwise-conv custom autograd path with a Core ML-convertible equivalent.
- Kept the released `coremltools` 9.0 meshgrid workaround in place.

### Added

- Added a fast no-download CI smoke test for dependency imports, CLI wiring,
  and model registry resolution.
- Added explicit Core ML vs PyTorch reference verification commands.
- Added PR hygiene through a repository pull request template.

## 0.1.0

### Added

- Initial RF-DETR to Core ML export package.
- Detection and segmentation model export through `rfdetr-coreml`.
- Runtime patch overlay for RF-DETR/Core ML conversion gaps.
- Benchmark, precision, batch, and native Core ML validation scripts.
