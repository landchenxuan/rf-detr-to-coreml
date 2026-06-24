#!/usr/bin/env python3
"""Fast smoke checks for CI.

This intentionally avoids model instantiation, checkpoint downloads, and Core ML
conversion. It verifies that the installed dependency stack can import the
package, that CLI wiring exists, and that every registry entry resolves against
the installed RF-DETR release.
"""

from contextlib import contextmanager, redirect_stdout
import ast
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import socket
import sys


EXPECTED_MODELS = {
    "nano",
    "small",
    "medium",
    "large",
    "seg-nano",
    "seg-small",
    "seg-medium",
    "seg-large",
    "seg-xlarge",
    "seg-2xlarge",
}

REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def network_disabled():
    """Fail fast if this smoke test accidentally reaches for model downloads."""
    real_create_connection = socket.create_connection
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def blocked_connect(*_args, **_kwargs):
        raise RuntimeError("network access is disabled for no-download smoke tests")

    socket.create_connection = blocked_connect
    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect
    try:
        yield
    finally:
        socket.create_connection = real_create_connection
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex


def check_versions() -> None:
    import coremltools as ct
    import torch
    import torchvision

    versions = {
        "torch": torch.__version__.split("+", 1)[0],
        "torchvision": torchvision.__version__.split("+", 1)[0],
        "coremltools": ct.__version__,
        "rfdetr": metadata.version("rfdetr"),
    }
    print("dependency versions:", versions)
    assert versions["coremltools"] == "9.0", versions
    assert versions["rfdetr"] == "1.8.1", versions
    assert versions["torch"] == "2.7.0", versions
    assert versions["torchvision"] == "0.22.0", versions


def check_documentation_provenance() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    docs = {
        "README.md": readme,
        "CHANGELOG.md": changelog,
    }

    required_snippets = {
        "README.md": [
            "The current tested compatibility baseline is RF-DETR 1.8.1, "
            "Core ML Tools 9.0, and torch 2.7.0.",
            "The package targets released `coremltools` 9.0 and `rfdetr` 1.8.1.",
            "Numbers below were measured on Apple M5 Pro 18-core, RF-DETR 1.7.1, "
            "coremltools 9.0, and real test images.",
            "The ONNX benchmark script uses RF-DETR 1.7.1's official ONNX exporter",
        ],
        "CHANGELOG.md": [
            "`coremltools==9.0`, `rfdetr==1.8.1`, `torch==2.7.0`, and",
            "Kept the existing latency, diff, FP16, and ONNX benchmark measurements "
            "labeled as RF-DETR 1.7.1 provenance.",
        ],
    }

    for filename, snippets in required_snippets.items():
        text = " ".join(docs[filename].split())
        missing = [snippet for snippet in snippets if snippet not in text]
        assert not missing, f"{filename} missing documentation provenance snippets: {missing}"

    forbidden_phrases = [
        "keypoint support",
        "keypoint export",
        "keypoint model",
        "keypoint outputs",
    ]
    for filename, text in docs.items():
        lowered = text.lower()
        matches = [phrase for phrase in forbidden_phrases if phrase in lowered]
        assert not matches, f"{filename} contains unsupported keypoint wording: {matches}"


def check_registry(model_registry, import_model_class) -> None:
    model_names = set(model_registry)
    missing = EXPECTED_MODELS.difference(model_names)
    extra = model_names.difference(EXPECTED_MODELS)
    assert not missing, f"missing model registry entries: {sorted(missing)}"
    assert not extra, f"unexpected model registry entries: {sorted(extra)}"

    failures = []
    for model_name in model_registry:
        try:
            model_cls = import_model_class(model_name)
        except Exception as exc:  # noqa: BLE001 - report all registry failures
            failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
            continue
        print(f"registry ok: {model_name} -> {model_cls.__module__}.{model_cls.__name__}")

    assert not failures, "registry resolution failed:\n" + "\n".join(failures)


def check_cli_help() -> None:
    import rfdetr_coreml.cli as cli

    original_argv = sys.argv[:]
    stdout = io.StringIO()
    try:
        sys.argv = ["rfdetr-coreml", "--help"]
        with redirect_stdout(stdout):
            try:
                cli.main()
            except SystemExit as exc:
                assert exc.code == 0, exc
    finally:
        sys.argv = original_argv

    help_text = stdout.getvalue()
    assert "--model" in help_text
    assert "seg-nano" in help_text


def check_class_metadata(export_module) -> None:
    metadata_dict = export_module._class_names_to_metadata(
        ["person", "dog"],
        [1, 18],
        "test",
    )
    assert ast.literal_eval(metadata_dict["names"]) == {0: "person", 1: "dog"}
    assert json.loads(metadata_dict["class_names"]) == ["person", "dog"]
    assert json.loads(metadata_dict["class_ids"]) == [1, 18]
    assert json.loads(metadata_dict["class_mapping"]) == {"1": "person", "18": "dog"}
    assert metadata_dict["class_count"] == "2"

    assert export_module._extract_checkpoint_class_labels({
        "args": {"class_names": ("cat", "bird")},
    }) == (["cat", "bird"], None)

    assert export_module._extract_checkpoint_class_labels({
        "args": {"class_names": "bird"},
    }) == (["bird"], None)

    assert export_module._extract_checkpoint_class_labels({
        "args": {"class_names": {1: "person", 18: "dog"}},
    }) == (["person", "dog"], [1, 18])

    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

    class CocoModel:
        class_names = list(COCO_CLASS_NAMES)

    coco_metadata = export_module._class_names_metadata(
        CocoModel(),
        weights_path=None,
        checkpoint_class_labels=None,
        num_classes=90,
    )
    assert coco_metadata["class_names_source"] == "coco"
    assert ast.literal_eval(coco_metadata["names"])[0] == "person"
    assert json.loads(coco_metadata["class_ids"])[0] == 1
    assert json.loads(coco_metadata["class_mapping"])["18"] == "dog"

    class FakeModel:
        class_names = ["cat", "bird"]

    custom_missing_names = export_module._class_names_metadata(
        FakeModel(),
        weights_path="custom.pth",
        checkpoint_class_labels=None,
        num_classes=2,
    )
    assert custom_missing_names == {
        "class_names_source": "unavailable",
        "class_count": "2",
    }

    custom_90_missing_names = export_module._class_names_metadata(
        CocoModel(),
        weights_path="custom-90-class.pth",
        checkpoint_class_labels=None,
        num_classes=90,
    )
    assert custom_90_missing_names == {
        "class_names_source": "unavailable",
        "class_count": "90",
    }

    class Rfdetr15Model:
        class_names = {1: "person", 18: "dog"}

    rfdetr15_metadata = export_module._class_names_metadata(
        Rfdetr15Model(),
        weights_path=None,
        checkpoint_class_labels=None,
        num_classes=90,
    )
    assert json.loads(rfdetr15_metadata["class_names"]) == ["person", "dog"]
    assert ast.literal_eval(rfdetr15_metadata["names"]) == {0: "person", 1: "dog"}
    assert json.loads(rfdetr15_metadata["class_ids"]) == [1, 18]
    assert json.loads(rfdetr15_metadata["class_mapping"]) == {"1": "person", "18": "dog"}


def check_required_rfdetr_patches() -> None:
    import rfdetr_coreml.patches as patches
    from rfdetr.models.heads import segmentation as seg_mod

    original_target = seg_mod._DepthwiseConvWithoutCuDNN
    assert original_target.__module__ == "rfdetr_coreml.patches", original_target

    del seg_mod._DepthwiseConvWithoutCuDNN
    patches._applied = False
    try:
        try:
            patches.apply_rfdetr_patches()
        except RuntimeError as exc:
            assert "segmentation" in str(exc).lower(), exc
            assert "_DepthwiseConvWithoutCuDNN" in str(exc), exc
            assert patches._applied is False
        else:
            raise AssertionError("missing segmentation depthwise target did not fail early")
    finally:
        seg_mod._DepthwiseConvWithoutCuDNN = original_target
        patches._applied = False
        patches.apply_rfdetr_patches()


def main() -> None:
    check_documentation_provenance()
    with network_disabled():
        import rfdetr_coreml
        import rfdetr_coreml.export as export_module
        from rfdetr_coreml.export import MODEL_REGISTRY, _import_model_class

        assert hasattr(rfdetr_coreml, "export_to_coreml")
        check_versions()
        check_registry(MODEL_REGISTRY, _import_model_class)
        check_cli_help()
        check_class_metadata(export_module)
        check_required_rfdetr_patches()
    print("smoke: ok")


if __name__ == "__main__":
    main()
