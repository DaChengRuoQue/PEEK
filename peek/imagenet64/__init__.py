"""Self-contained ImageNet-64 EDM evaluation support."""

from .protocol import EvaluationProtocol, model_call_count, paper_protocol

__all__ = ["EvaluationProtocol", "model_call_count", "paper_protocol"]
