"""Greedy interpolation-schedule search for PEEK.

Each nonzero candidate must strictly improve both the current schedule and an
ordinary sampler with matched denoiser-evaluation cost. Baseline metrics and
optional selected-prefix states are reused across gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


_NO_PREFIX = object()


@dataclass(frozen=True)
class SearchEvaluation:
    """Metric from one schedule evaluation and an optional next prefix."""

    metric: float
    next_prefix: Any = None


@dataclass(frozen=True)
class BaselineEvaluation:
    """Metric from one ordinary baseline and its reusable state cache."""

    metric: float
    cache_state: Any = None


@dataclass(frozen=True)
class CandidateRecord:
    """One nonzero alpha considered at a schedule gate."""

    alpha: float
    metric: float
    alpha_schedule: tuple[float, ...]


@dataclass(frozen=True)
class SearchRecord:
    """Decision and comparisons for one peekable gate."""

    step_index: int
    selected_alpha: float
    selected_metric: float
    current_metric_before_step: float
    matched_baseline_metric: float
    compute_matched_nfe: int
    accepted: bool
    short_circuited: bool
    candidates: tuple[CandidateRecord, ...]


@dataclass(frozen=True)
class SearchResult:
    """Selected schedule and complete greedy-search trace."""

    alpha_schedule: tuple[float, ...]
    metric: float
    selected_count: int
    actual_nfe: int
    records: tuple[SearchRecord, ...]
    baseline_metrics: tuple[tuple[int, float], ...]


ScheduleEvaluator = Callable[..., float | SearchEvaluation]
BaselineEvaluator = Callable[[int], float | BaselineEvaluation]
BaselineAdvancer = Callable[[Any, int, int], float | BaselineEvaluation]
PrefixAdvancer = Callable[[Any, int, float], Any]


def _validate(base_nfe: int, candidate_alphas: Sequence[float]) -> tuple[float, ...]:
    if int(base_nfe) < 2:
        raise ValueError("base_nfe must be at least 2")
    candidates = tuple(float(alpha) for alpha in candidate_alphas)
    if not candidates:
        raise ValueError("candidate_alphas must contain at least one nonzero value")
    if any(not 0.0 < alpha <= 1.0 for alpha in candidates):
        raise ValueError("candidate alphas must be in (0, 1]")
    if any(right <= left for left, right in zip(candidates, candidates[1:])):
        raise ValueError("candidate alphas must be strictly increasing")
    return candidates


def _evaluate(
    evaluator: ScheduleEvaluator,
    schedule: Sequence[float],
    *,
    start_index: int,
    prefix_state: Any,
    capture_next_prefix: bool,
) -> SearchEvaluation:
    result = evaluator(
        tuple(float(alpha) for alpha in schedule),
        start_index=int(start_index),
        prefix_state=prefix_state,
        capture_next_prefix=bool(capture_next_prefix),
    )
    if isinstance(result, SearchEvaluation):
        return result
    return SearchEvaluation(metric=float(result))


def _evaluate_baseline(result: float | BaselineEvaluation) -> BaselineEvaluation:
    if isinstance(result, BaselineEvaluation):
        return result
    return BaselineEvaluation(metric=float(result))


def greedy_schedule_search(
    *,
    base_nfe: int,
    candidate_alphas: Sequence[float],
    evaluate_schedule: ScheduleEvaluator,
    evaluate_baseline: BaselineEvaluator,
    advance_baseline: BaselineAdvancer | None = None,
    initial_prefix: Any = _NO_PREFIX,
    advance_prefix: PrefixAdvancer | None = None,
    short_circuit: bool = True,
    tolerance: float = 1e-8,
) -> SearchResult:
    """Search PEEK gates in order with compute-matched guardrails.

    ``evaluate_schedule`` receives ``start_index``, ``prefix_state``, and
    ``capture_next_prefix`` keyword arguments. Without prefix caching it should
    evaluate the full schedule; with caching it may evaluate only the suffix.
    """

    candidates = _validate(base_nfe, candidate_alphas)
    prefix_enabled = initial_prefix is not _NO_PREFIX
    if prefix_enabled and advance_prefix is None:
        raise ValueError("advance_prefix is required when initial_prefix is provided")
    num_peekable = int(base_nfe) - 1
    alpha_schedule = [0.0] * num_peekable
    initial = _evaluate(
        evaluate_schedule,
        alpha_schedule,
        start_index=0,
        prefix_state=None,
        capture_next_prefix=False,
    )
    current_metric = float(initial.metric)
    selected_count = 0
    baseline_cache: dict[int, BaselineEvaluation] = {}
    records: list[SearchRecord] = []
    prefix_state = initial_prefix

    for step_index in range(num_peekable):
        compute_matched_nfe = int(base_nfe) + selected_count + 1
        if compute_matched_nfe not in baseline_cache:
            source_nfe = max(
                (
                    nfe
                    for nfe, evaluation in baseline_cache.items()
                    if nfe < compute_matched_nfe and evaluation.cache_state is not None
                ),
                default=None,
            )
            if advance_baseline is not None and source_nfe is not None:
                source = baseline_cache[source_nfe]
                baseline_cache[compute_matched_nfe] = _evaluate_baseline(
                    advance_baseline(source.cache_state, source_nfe, compute_matched_nfe)
                )
            else:
                baseline_cache[compute_matched_nfe] = _evaluate_baseline(
                    evaluate_baseline(compute_matched_nfe)
                )
        baseline_metric = float(baseline_cache[compute_matched_nfe].metric)
        metric_before = current_metric
        best_alpha = 0.0
        best_metric = current_metric
        best_schedule = list(alpha_schedule)
        best_next_prefix = None
        previous_candidate_metric: float | None = None
        candidate_records: list[CandidateRecord] = []
        was_short_circuited = False

        for alpha in candidates:
            trial = list(alpha_schedule)
            trial[step_index] = alpha
            evaluation = _evaluate(
                evaluate_schedule,
                trial,
                start_index=step_index if prefix_enabled else 0,
                prefix_state=prefix_state if prefix_enabled else None,
                capture_next_prefix=bool(prefix_enabled and step_index < num_peekable - 1),
            )
            metric = float(evaluation.metric)
            candidate_records.append(
                CandidateRecord(alpha=alpha, metric=metric, alpha_schedule=tuple(trial))
            )
            if (
                short_circuit
                and previous_candidate_metric is not None
                and metric > previous_candidate_metric + float(tolerance)
            ):
                was_short_circuited = True
                break
            previous_candidate_metric = metric
            if metric < best_metric - float(tolerance) and metric < baseline_metric - float(tolerance):
                best_alpha = alpha
                best_metric = metric
                best_schedule = trial
                best_next_prefix = evaluation.next_prefix

        accepted = best_alpha != 0.0
        if accepted:
            alpha_schedule = best_schedule
            current_metric = best_metric
            selected_count += 1
        records.append(
            SearchRecord(
                step_index=step_index,
                selected_alpha=best_alpha,
                selected_metric=best_metric,
                current_metric_before_step=metric_before,
                matched_baseline_metric=baseline_metric,
                compute_matched_nfe=compute_matched_nfe,
                accepted=accepted,
                short_circuited=was_short_circuited,
                candidates=tuple(candidate_records),
            )
        )

        if prefix_enabled and step_index < num_peekable - 1:
            if accepted and best_next_prefix is not None:
                prefix_state = best_next_prefix
            else:
                assert advance_prefix is not None
                prefix_state = advance_prefix(prefix_state, step_index, best_alpha)

    return SearchResult(
        alpha_schedule=tuple(alpha_schedule),
        metric=current_metric,
        selected_count=selected_count,
        actual_nfe=int(base_nfe) + selected_count,
        records=tuple(records),
        baseline_metrics=tuple(
            (nfe, float(evaluation.metric))
            for nfe, evaluation in sorted(baseline_cache.items())
        ),
    )
