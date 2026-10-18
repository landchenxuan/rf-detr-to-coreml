"""Input adapter for RF-DETR 1.11's official Core ML exporter.

The minor-version dependency bound covers this narrow exporter integration.
Upstream retains graph capture, decompositions, op checks and conversion passes.
"""

from dataclasses import replace
from functools import partial

import coremltools as ct
import numpy as np
from rfdetr.export._coreml.exporter import CoreMLExporter


class ImageCoreMLExporter(CoreMLExporter):
    """Keep the package's image/[0,1] tensor interface on the official pipeline."""

    def __init__(self, config, *, metadata, description):
        super().__init__(config)
        self.metadata = metadata
        self.description = description

    def _build_mlmodel(self, graph, compute_precision, api):
        shape = tuple(graph.input_tensors.shape)
        if shape[0] == 1:
            image_input = ct.ImageType(name="image", shape=shape, scale=1.0 / 255.0)
        else:
            image_input = ct.TensorType(name="image", shape=shape, dtype=np.float32)
        return super()._build_mlmodel(
            graph,
            compute_precision,
            replace(api, convert=partial(api.convert, inputs=[image_input])),
        )

    def _save_mlmodel(self, mlmodel, output_file):
        mlmodel.author = "rfdetr_coreml"
        mlmodel.short_description = self.description
        mlmodel.version = self.metadata["rfdetr_version"]
        mlmodel.user_defined_metadata.update(self.metadata)
        return super()._save_mlmodel(mlmodel, output_file)
