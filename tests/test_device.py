"""Tests for device auto-detection."""

from unittest.mock import patch

from revos.device import auto_detect_device


def test_detect_cpu_when_no_cuda():
    with patch(
        "onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    ):
        assert auto_detect_device() == "cpu"


def test_detect_cuda_when_available():
    with patch(
        "onnxruntime.get_available_providers",
        return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        assert auto_detect_device() == "cuda"


def test_detect_cpu_on_import_error():
    with patch("builtins.__import__", side_effect=ImportError("no onnxruntime")):
        assert auto_detect_device() == "cpu"
