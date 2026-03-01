# RF-DETR to CoreML

## Project Structure
- `rfdetr_coreml/` — Python package (monkey-patch overlay + export logic)
- `export_coreml.py` — CLI wrapper (delegates to `rfdetr_coreml.cli`)
- `test_*.py`, `benchmark_*.py` — Test and benchmark scripts

## Key Constraints
- Never fork upstream rfdetr — use runtime monkey-patches only
- FP32 only in production (FP16 breaks deformable attention)
- Tested with: Python 3.12, torch 2.7.0, coremltools 8.1, rfdetr 1.5.1
