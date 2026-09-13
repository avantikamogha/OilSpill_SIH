"""Reusable oil-spill segmentation and export pipelines."""

__all__ = ["DetectionConfig", "detect_image", "evaluate_model", "infer_directory", "train_model"]


def __getattr__(name: str):
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")