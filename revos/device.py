"""Auto-detect available compute device (CPU or CUDA GPU)."""

import logging

logger = logging.getLogger(__name__)


def auto_detect_device() -> str:
    """Detect whether CUDA GPU is available via onnxruntime providers.

    Returns:
        "cuda" if CUDAExecutionProvider is available, "cpu" otherwise.
    """
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            logger.info("CUDA GPU detected via onnxruntime")
            return "cuda"
    except ImportError:
        logger.warning("onnxruntime not installed, defaulting to CPU")

    logger.info("Using CPU for inference")
    return "cpu"
