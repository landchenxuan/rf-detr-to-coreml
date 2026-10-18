#!/usr/bin/env python3
"""Real Core ML checks of the image/tensor adapter, without RF-DETR weights."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import coremltools as ct
import numpy as np
from PIL import Image
import torch
from torch import nn
from rfdetr.export.prepare import ExportGraph

from rfdetr_coreml.export import NormalizedWrapper, export_to_coreml, _class_names_to_metadata


class AdapterTests(unittest.TestCase):
    def check_export(self, batch_size, precision):
        # Identity exposes every normalized pixel, so channel swaps, scaling
        # mistakes and applying normalization twice cannot hide behind a model.
        pixels = np.array([
            [[255, 0, 128], [12, 64, 240]],
            [[0, 255, 32], [200, 100, 0]],
        ], dtype=np.uint8)
        raw = pixels.transpose(2, 0, 1)[None].astype(np.float32) / 255
        raw = np.ascontiguousarray(np.concatenate([raw, 1 - raw], axis=0)[:batch_size])
        mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
        expected = (raw - mean) / std
        graph = ExportGraph(
            model=NormalizedWrapper(nn.Identity(), 2).eval(),
            input_tensors=torch.from_numpy(raw), input_names=("input",),
            output_names=("normalized_pixels",), dynamic_axes=None,
            shape=(2, 2), backbone_only=False,
        )
        metadata = {
            "rfdetr_version": "1.11.2",
            "coremltools_version": "9.0",
            "model_variant": "small",
            "precision": precision,
            **_class_names_to_metadata(["cat", "bird"], [0, 1], "checkpoint"),
        }
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.best.pth"
            torch.save({"model": {}, "class_names": ["cat", "bird"]}, checkpoint)
            fake_model = SimpleNamespace(model=SimpleNamespace(
                model=nn.Identity(), args=SimpleNamespace(num_classes=2),
            ))
            # Replace only the RF-DETR weights/preparation. The public adapter,
            # official exporter, naming, package save/load and prediction are real.
            with patch("rfdetr_coreml.export._import_model_class", return_value=Mock(return_value=fake_model)), \
                 patch("rfdetr_coreml.export._prepare_image_graph", return_value=graph):
                path = export_to_coreml(
                    "small", directory, precision, weights_path=str(checkpoint), batch_size=batch_size,
                )
            batch_tag = f"-batch{batch_size}" if batch_size > 1 else ""
            self.assertEqual(Path(path).name, f"rf-detr-small-checkpoint.best-{precision}{batch_tag}.mlpackage")
            model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
            spec = model.get_spec()
            inp = spec.description.input[0]
            self.assertEqual(inp.name, "image")
            self.assertEqual(inp.type.WhichOneof("Type"), "imageType" if batch_size == 1 else "multiArrayType")
            self.assertEqual({k: model.user_defined_metadata[k] for k in metadata}, metadata)
            self.assertEqual(model.author, "rfdetr_coreml")
            self.assertIn("RF-DETR small", model.short_description)
            if batch_size == 1:
                self.assertEqual((inp.type.imageType.width, inp.type.imageType.height), (2, 2))
                data = Image.fromarray(pixels)
            else:
                self.assertEqual(list(inp.type.multiArrayType.shape), [2, 3, 2, 2])
                data = raw
            result = model.predict({"image": data})
            actual = result[spec.description.output[0].name]
            tolerance = 0.005 if precision == "fp16" else 1e-5
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)

    def test_image_fp32(self):
        self.check_export(1, "fp32")

    def test_image_fp16(self):
        self.check_export(1, "fp16")

    def test_batch_tensor(self):
        self.check_export(2, "fp32")


if __name__ == "__main__":
    unittest.main()
