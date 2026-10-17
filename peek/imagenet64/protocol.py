"""Immutable protocol for the paper-derived ImageNet-64 comparison."""

from __future__ import annotations

from dataclasses import dataclass


PAPER_NUM_IMAGES = 50_000
PAPER_BASELINE_NFE = 9
PAPER_PEEK_BASE_NFE = 6
PAPER_PEEK_GATES = (1, 1, 1, 0, 0)


def model_call_count(base_nfe: int, gates: tuple[int, ...]) -> int:
    if base_nfe < 1:
        raise ValueError("base_nfe must be positive")
    if any(gate not in (0, 1) for gate in gates):
        raise ValueError("PEEK gates must be binary")
    return int(base_nfe) + sum(gates)


@dataclass(frozen=True)
class EvaluationProtocol:
    num_images: int = PAPER_NUM_IMAGES
    baseline_nfe: int = PAPER_BASELINE_NFE
    peek_base_nfe: int = PAPER_PEEK_BASE_NFE
    peek_gates: tuple[int, ...] = PAPER_PEEK_GATES
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    def __post_init__(self) -> None:
        if self.num_images != PAPER_NUM_IMAGES:
            raise ValueError("the evaluation protocol requires exactly 50,000 images")
        if self.baseline_nfe != PAPER_BASELINE_NFE:
            raise ValueError("the compute-matched baseline requires 9 model evaluations")
        if self.peek_base_nfe != PAPER_PEEK_BASE_NFE:
            raise ValueError("the PEEK path requires 6 base model evaluations")
        if self.peek_gates != PAPER_PEEK_GATES:
            raise ValueError("the evaluation requires the fixed PEEK gate schedule (1, 1, 1, 0, 0)")
        if self.peek_total_nfe != self.baseline_nfe:
            raise ValueError("baseline and PEEK must have matching model-call counts")

    @property
    def peek_total_nfe(self) -> int:
        return model_call_count(self.peek_base_nfe, self.peek_gates)


def paper_protocol() -> EvaluationProtocol:
    return EvaluationProtocol()
