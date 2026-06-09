#!/usr/bin/env python3
"""Fast smoke checks for CI.

This intentionally avoids model instantiation, checkpoint downloads, and Core ML
conversion. It verifies that the installed dependency stack can import the
package, that CLI wiring exists, and that every registry entry resolves against
the installed RF-DETR release.
"""

from contextlib import contextmanager, redirect_stdout
import importlib.metadata as metadata
import io
import socket
import sys


EXPECTED_MIN_MODELS = {
    "nano",
    "small",
    "medium",
    "base",
    "large",
    "seg-preview",
    "seg-nano",
    "seg-small",
    "seg-medium",
    "seg-large",
    "seg-xlarge",
    "seg-2xlarge",
}


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

    versions = {
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "rfdetr": metadata.version("rfdetr"),
    }
    print("dependency versions:", versions)
    assert versions["coremltools"] == "9.0", versions
    assert versions["rfdetr"] == "1.7.1", versions


def check_registry(model_registry, import_model_class) -> None:
    missing = EXPECTED_MIN_MODELS.difference(model_registry)
    assert not missing, f"missing model registry entries: {sorted(missing)}"

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


def main() -> None:
    with network_disabled():
        import rfdetr_coreml
        from rfdetr_coreml.export import MODEL_REGISTRY, _import_model_class

        assert hasattr(rfdetr_coreml, "export_to_coreml")
        check_versions()
        check_registry(MODEL_REGISTRY, _import_model_class)
        check_cli_help()
    print("smoke: ok")


if __name__ == "__main__":
    main()
