"""Local verification never opts into ONNX Runtime network telemetry."""

from __future__ import annotations

import importlib.util
import os

import pytest


# Set this before test-module collection can import ONNX Runtime. Its public
# event-disable API is also called before any parity-check inference session.
# https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
os.environ["ORT_DISABLE_TELEMETRY"] = "1"


@pytest.fixture(scope="session", autouse=True)
def disable_onnx_runtime_telemetry():
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort

        ort.disable_telemetry_events()
