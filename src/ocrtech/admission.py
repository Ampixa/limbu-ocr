"""Recognizer admission decision artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ValidationError


@dataclass(slots=True)
class FailedAdmissionGate:
    baseline: str
    metric: str
    slices: list[str] = field(default_factory=list)
    pairs: int | None = None
    candidate_mean: float | None = None
    baseline_mean: float | None = None
    absolute_improvement: float | None = None
    relative_improvement: float | None = None
    win_count: int | None = None
    loss_count: int | None = None
    tie_count: int | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "metric": self.metric,
            "slices": self.slices,
            "pairs": self.pairs,
            "candidate_mean": self.candidate_mean,
            "baseline_mean": self.baseline_mean,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "tie_count": self.tie_count,
            "reason": self.reason,
        }


@dataclass(slots=True)
class RecognizerAdmissionDecision:
    validation_report: str
    decision: str
    claim_status: str
    passed: bool
    sample_count: int | None
    missing_requirements: list[str] = field(default_factory=list)
    failed_gates: list[FailedAdmissionGate] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    decision_path: str | None = None
    markdown_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_report": self.validation_report,
            "decision": self.decision,
            "claim_status": self.claim_status,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "missing_requirements": self.missing_requirements,
            "failed_gates": [gate.to_dict() for gate in self.failed_gates],
            "next_actions": self.next_actions,
            "decision_path": self.decision_path,
            "markdown_path": self.markdown_path,
        }


@dataclass(slots=True)
class RecognizerFailureRun:
    source_path: str
    validation_report: str | None
    model_card: str | None
    model_id: str
    backend: str
    base_model: str
    decision: str
    claim_status: str
    passed: bool
    sample_count: int | None
    failed_gate_count: int
    total_pairs: int
    total_wins: int
    total_losses: int
    total_ties: int
    failed_slices: list[str]
    failed_metrics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "validation_report": self.validation_report,
            "model_card": self.model_card,
            "model_id": self.model_id,
            "backend": self.backend,
            "base_model": self.base_model,
            "decision": self.decision,
            "claim_status": self.claim_status,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "failed_gate_count": self.failed_gate_count,
            "total_pairs": self.total_pairs,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_ties": self.total_ties,
            "failed_slices": self.failed_slices,
            "failed_metrics": self.failed_metrics,
        }


@dataclass(slots=True)
class RecognizerFailureFamily:
    family_key: str
    backend: str
    base_model: str
    run_count: int
    rejected_count: int
    admitted_count: int
    total_failed_gates: int
    total_pairs: int
    total_wins: int
    total_losses: int
    total_ties: int
    failed_slices: list[str]
    failed_metrics: list[str]
    recommendation: str
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_key": self.family_key,
            "backend": self.backend,
            "base_model": self.base_model,
            "run_count": self.run_count,
            "rejected_count": self.rejected_count,
            "admitted_count": self.admitted_count,
            "total_failed_gates": self.total_failed_gates,
            "total_pairs": self.total_pairs,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_ties": self.total_ties,
            "failed_slices": self.failed_slices,
            "failed_metrics": self.failed_metrics,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class RecognizerFailureReview:
    admission_decisions: list[str]
    validation_reports: list[str]
    model_cards: list[str]
    min_rejections_to_stop: int
    runs: list[RecognizerFailureRun]
    families: list[RecognizerFailureFamily]
    next_actions: list[str]
    review_path: str | None = None
    markdown_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "admission_decisions": self.admission_decisions,
            "validation_reports": self.validation_reports,
            "model_cards": self.model_cards,
            "min_rejections_to_stop": self.min_rejections_to_stop,
            "runs": [run.to_dict() for run in self.runs],
            "families": [family.to_dict() for family in self.families],
            "next_actions": self.next_actions,
            "review_path": self.review_path,
            "markdown_path": self.markdown_path,
        }


def decide_recognizer_admission(validation_report: str | Path, output_dir: str | Path) -> RecognizerAdmissionDecision:
    report_path = Path(validation_report)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"admission validation report does not exist: {report_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid admission validation JSON {report_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"admission validation report must be a JSON object: {report_path}")

    claim_status = str(payload.get("claim_status") or "")
    passed = bool(payload.get("passed"))
    decision = "admit" if claim_status == "validated" and passed else "reject"
    failed_gates = _failed_gates(payload)
    missing_requirements = _missing_requirements(payload)
    next_actions = _next_actions(decision, failed_gates, missing_requirements)

    sample_count_value = payload.get("sample_count")
    sample_count = int(sample_count_value) if isinstance(sample_count_value, int | float) and not isinstance(sample_count_value, bool) else None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    decision_path = out / "recognizer-admission-decision.json"
    markdown_path = out / "recognizer-admission-decision.md"
    decision_report = RecognizerAdmissionDecision(
        validation_report=str(report_path),
        decision=decision,
        claim_status=claim_status or "not_run",
        passed=passed,
        sample_count=sample_count,
        missing_requirements=missing_requirements,
        failed_gates=failed_gates,
        next_actions=next_actions,
        decision_path=str(decision_path),
        markdown_path=str(markdown_path),
    )
    decision_path.write_text(json.dumps(decision_report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, decision_report)
    return decision_report


def review_recognizer_failures(
    admission_decisions: list[str | Path],
    validation_reports: list[str | Path],
    output_dir: str | Path,
    *,
    model_cards: list[str | Path] | None = None,
    min_rejections_to_stop: int = 3,
) -> RecognizerFailureReview:
    if min_rejections_to_stop < 1:
        raise ValidationError("min_rejections_to_stop must be >= 1")
    if not admission_decisions and not validation_reports:
        raise ValidationError("recognizer failure review requires at least one admission decision or validation report")

    cards = [Path(path) for path in (model_cards or [])]
    card_payloads = [_read_json_object(path, "model card") for path in cards]
    runs: list[RecognizerFailureRun] = []
    for index, path_value in enumerate(admission_decisions):
        path = Path(path_value)
        decision_payload = _read_json_object(path, "recognizer admission decision")
        validation_payload = _read_optional_validation_payload(decision_payload, path)
        model_identity = _model_identity_for_run(index, card_payloads, cards, validation_payload, path)
        runs.append(_failure_run_from_decision(path, decision_payload, model_identity))
    offset = len(runs)
    for index, path_value in enumerate(validation_reports):
        path = Path(path_value)
        validation_payload = _read_json_object(path, "admission validation report")
        decision_payload = _decision_payload_from_validation(path, validation_payload)
        model_identity = _model_identity_for_run(offset + index, card_payloads, cards, validation_payload, path)
        runs.append(_failure_run_from_decision(path, decision_payload, model_identity))

    families = _failure_families(runs, min_rejections_to_stop=min_rejections_to_stop)
    next_actions = _failure_review_next_actions(families)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    review_path = out / "recognizer-failure-review.json"
    markdown_path = out / "recognizer-failure-review.md"
    review = RecognizerFailureReview(
        admission_decisions=[str(Path(path)) for path in admission_decisions],
        validation_reports=[str(Path(path)) for path in validation_reports],
        model_cards=[str(path) for path in cards],
        min_rejections_to_stop=min_rejections_to_stop,
        runs=runs,
        families=families,
        next_actions=next_actions,
        review_path=str(review_path),
        markdown_path=str(markdown_path),
    )
    review_path.write_text(json.dumps(review.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_failure_review_markdown(markdown_path, review)
    return review


def stopped_recognizer_family_reasons(failure_review: str | Path, *, backend: str, base_model: str) -> list[str]:
    review_path = Path(failure_review)
    payload = _read_json_object(review_path, "recognizer failure review")
    families = payload.get("families")
    if not isinstance(families, list):
        raise ValidationError(f"recognizer failure review missing families list: {review_path}")
    canonical_base = _canonical_recognizer_base_model(base_model)
    reasons: list[str] = []
    for item in families:
        if not isinstance(item, dict):
            continue
        family_backend = str(item.get("backend") or "")
        family_base = _canonical_recognizer_base_model(str(item.get("base_model") or ""))
        if family_backend != backend or family_base != canonical_base:
            continue
        if item.get("recommendation") != "stop_family":
            continue
        reasons.append(
            "recognizer family is stopped by {review}: family={family} rejected_runs={rejected} "
            "wins/losses/ties={wins}/{losses}/{ties}".format(
                review=review_path,
                family=item.get("family_key") or f"{family_backend}::{family_base}",
                rejected=item.get("rejected_count"),
                wins=item.get("total_wins"),
                losses=item.get("total_losses"),
                ties=item.get("total_ties"),
            )
        )
    return reasons


def assert_recognizer_family_not_stopped(failure_review: str | Path, *, backend: str, base_model: str) -> None:
    reasons = stopped_recognizer_family_reasons(failure_review, backend=backend, base_model=base_model)
    if reasons:
        raise ValidationError("; ".join(reasons))


def _missing_requirements(payload: dict[str, Any]) -> list[str]:
    value = payload.get("missing_requirements")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _failed_gates(payload: dict[str, Any]) -> list[FailedAdmissionGate]:
    gates = payload.get("gates")
    if not isinstance(gates, list):
        return []
    failed: list[FailedAdmissionGate] = []
    for item in gates:
        if not isinstance(item, dict) or item.get("passed") is True:
            continue
        gate = item.get("gate")
        gate_payload = gate if isinstance(gate, dict) else {}
        failed.append(
            FailedAdmissionGate(
                baseline=str(gate_payload.get("baseline") or ""),
                metric=str(gate_payload.get("metric") or ""),
                slices=_gate_slices(gate_payload),
                pairs=_int_or_none(item.get("pairs")),
                candidate_mean=_float_or_none(item.get("candidate_mean")),
                baseline_mean=_float_or_none(item.get("baseline_mean")),
                absolute_improvement=_float_or_none(item.get("absolute_improvement")),
                relative_improvement=_float_or_none(item.get("relative_improvement")),
                win_count=_int_or_none(item.get("win_count")),
                loss_count=_int_or_none(item.get("loss_count")),
                tie_count=_int_or_none(item.get("tie_count")),
                reason=str(item.get("reason")) if item.get("reason") else None,
            )
        )
    return failed


def _gate_slices(gate: dict[str, Any]) -> list[str]:
    value = gate.get("slices")
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    single = gate.get("slice")
    return [str(single)] if single else []


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    as_float = float(value)
    if not math.isfinite(as_float):
        return None
    return int(as_float)


def _next_actions(decision: str, failed_gates: list[FailedAdmissionGate], missing_requirements: list[str]) -> list[str]:
    if decision == "admit":
        return [
            "Attach this admission report to the model card and run claim preflight with --require-model-admission.",
            "Promote the recognizer only through the full document SOTA portfolio; admission is not a final SOTA claim.",
        ]
    actions: list[str] = []
    failed_slices = {slice_name.lower() for gate in failed_gates for slice_name in gate.slices}
    failed_metrics = {gate.metric for gate in failed_gates}
    losses = sum((gate.loss_count or 0) for gate in failed_gates)
    wins = sum((gate.win_count or 0) for gate in failed_gates)
    if "english" in failed_slices or "en" in failed_slices or "line_crop" in failed_slices:
        actions.append("Do not train another Devanagari-only recognizer branch until English preservation is explicitly designed and gated.")
    if "nepali" in failed_slices:
        actions.append("Investigate the Nepali recognizer path separately; the current branch does not clear the Devanagari admission slice.")
    if "cer" in failed_metrics or "wer" in failed_metrics:
        actions.append("Prioritize recognizer correctness before document-structure integration; OCR gates are failing upstream.")
    if losses > 0 and wins == 0:
        actions.append("Stop this exact model family for engine promotion; it has no paired wins in the admission failure gates.")
    if not actions and missing_requirements:
        actions.append("Fix the missing validation requirements before spending more GPU time.")
    if not actions:
        actions.append("Treat this branch as rejected until a new validation report passes the admission gate.")
    return actions


def _write_markdown(path: Path, decision: RecognizerAdmissionDecision) -> None:
    lines = [
        "# Recognizer Admission Decision",
        "",
        f"Validation report: `{decision.validation_report}`",
        f"Decision: `{decision.decision}`",
        f"Claim status: `{decision.claim_status}`",
        f"Passed: `{'yes' if decision.passed else 'no'}`",
        f"Samples: `{decision.sample_count if decision.sample_count is not None else 'unknown'}`",
        "",
        "## Missing Requirements",
        "",
    ]
    lines.extend(f"- {item}" for item in decision.missing_requirements) if decision.missing_requirements else lines.append("- none")
    lines.extend(
        [
            "",
            "## Failed Gates",
            "",
            "| baseline | metric | slices | pairs | candidate | baseline mean | abs improvement | wins | losses | ties | reason |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if decision.failed_gates:
        for gate in decision.failed_gates:
            lines.append(
                "| {baseline} | {metric} | {slices} | {pairs} | {candidate} | {baseline_mean} | {absolute} | {wins} | {losses} | {ties} | {reason} |".format(
                    baseline=gate.baseline,
                    metric=gate.metric,
                    slices=",".join(gate.slices),
                    pairs="" if gate.pairs is None else gate.pairs,
                    candidate=_format_optional_float(gate.candidate_mean),
                    baseline_mean=_format_optional_float(gate.baseline_mean),
                    absolute=_format_optional_float(gate.absolute_improvement),
                    wins="" if gate.win_count is None else gate.win_count,
                    losses="" if gate.loss_count is None else gate.loss_count,
                    ties="" if gate.tie_count is None else gate.tie_count,
                    reason=(gate.reason or "").replace("|", "\\|"),
                )
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in decision.next_actions) if decision.next_actions else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _read_optional_validation_payload(decision_payload: dict[str, Any], decision_path: Path) -> dict[str, Any] | None:
    value = decision_payload.get("validation_report")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute() and not path.exists():
        path = decision_path.parent / path
    if not path.exists():
        return None
    return _read_json_object(path, "admission validation report")


def _decision_payload_from_validation(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    claim_status = str(payload.get("claim_status") or "")
    passed = bool(payload.get("passed"))
    return {
        "validation_report": str(path),
        "decision": "admit" if claim_status == "validated" and passed else "reject",
        "claim_status": claim_status or "not_run",
        "passed": passed,
        "sample_count": payload.get("sample_count"),
        "failed_gates": [gate.to_dict() for gate in _failed_gates(payload)],
    }


def _model_identity_for_run(
    index: int,
    card_payloads: list[dict[str, Any]],
    cards: list[Path],
    validation_payload: dict[str, Any] | None,
    source_path: Path,
) -> tuple[str, str, str, str | None]:
    if index < len(card_payloads):
        payload = card_payloads[index]
        return (
            str(payload.get("model_id") or "unknown"),
            str(payload.get("backend") or "unknown"),
            str(payload.get("base_model") or "unknown"),
            str(cards[index]),
        )
    if validation_payload:
        candidate_path = _candidate_model_config_path(validation_payload, source_path)
        if candidate_path and candidate_path.exists():
            payload = _read_json_object(candidate_path, "candidate model card")
            return (
                str(payload.get("model_id") or "unknown"),
                str(payload.get("backend") or "unknown"),
                str(payload.get("base_model") or "unknown"),
                str(candidate_path),
            )
    return ("unknown", "unknown", "unknown", None)


def _candidate_model_config_path(validation_payload: dict[str, Any], source_path: Path) -> Path | None:
    provenance = validation_payload.get("provenance")
    if not isinstance(provenance, dict):
        return None
    candidate = provenance.get("candidate_model_config")
    if not isinstance(candidate, dict):
        return None
    path_value = candidate.get("path")
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    relative = source_path.parent / path
    return relative if relative.exists() else path


def _failure_run_from_decision(path: Path, payload: dict[str, Any], model_identity: tuple[str, str, str, str | None]) -> RecognizerFailureRun:
    gates = payload.get("failed_gates")
    failed_gates = [_failed_gate_from_payload(item) for item in gates if isinstance(item, dict)] if isinstance(gates, list) else []
    model_id, backend, base_model, model_card = model_identity
    return RecognizerFailureRun(
        source_path=str(path),
        validation_report=str(payload.get("validation_report")) if payload.get("validation_report") else None,
        model_card=model_card,
        model_id=model_id,
        backend=backend,
        base_model=base_model,
        decision=str(payload.get("decision") or "unknown"),
        claim_status=str(payload.get("claim_status") or "not_run"),
        passed=bool(payload.get("passed")),
        sample_count=_int_or_none(payload.get("sample_count")),
        failed_gate_count=len(failed_gates),
        total_pairs=sum(gate.pairs or 0 for gate in failed_gates),
        total_wins=sum(gate.win_count or 0 for gate in failed_gates),
        total_losses=sum(gate.loss_count or 0 for gate in failed_gates),
        total_ties=sum(gate.tie_count or 0 for gate in failed_gates),
        failed_slices=sorted({slice_name for gate in failed_gates for slice_name in gate.slices}),
        failed_metrics=sorted({gate.metric for gate in failed_gates if gate.metric}),
    )


def _failed_gate_from_payload(payload: dict[str, Any]) -> FailedAdmissionGate:
    return FailedAdmissionGate(
        baseline=str(payload.get("baseline") or ""),
        metric=str(payload.get("metric") or ""),
        slices=[str(item) for item in payload.get("slices", []) if str(item)] if isinstance(payload.get("slices"), list) else [],
        pairs=_int_or_none(payload.get("pairs")),
        candidate_mean=_float_or_none(payload.get("candidate_mean")),
        baseline_mean=_float_or_none(payload.get("baseline_mean")),
        absolute_improvement=_float_or_none(payload.get("absolute_improvement")),
        relative_improvement=_float_or_none(payload.get("relative_improvement")),
        win_count=_int_or_none(payload.get("win_count")),
        loss_count=_int_or_none(payload.get("loss_count")),
        tie_count=_int_or_none(payload.get("tie_count")),
        reason=str(payload.get("reason")) if payload.get("reason") else None,
    )


def _failure_families(runs: list[RecognizerFailureRun], *, min_rejections_to_stop: int) -> list[RecognizerFailureFamily]:
    grouped: dict[tuple[str, str], list[RecognizerFailureRun]] = {}
    for run in runs:
        grouped.setdefault((run.backend, _canonical_recognizer_base_model(run.base_model)), []).append(run)
    families: list[RecognizerFailureFamily] = []
    for (backend, base_model), family_runs in sorted(grouped.items()):
        rejected = [run for run in family_runs if run.decision == "reject"]
        admitted = [run for run in family_runs if run.decision == "admit"]
        total_wins = sum(run.total_wins for run in family_runs)
        total_losses = sum(run.total_losses for run in family_runs)
        recommendation = "continue_with_new_gate"
        rationale: list[str] = []
        if len(rejected) >= min_rejections_to_stop and total_wins == 0 and total_losses > 0:
            recommendation = "stop_family"
            rationale.append(f"{len(rejected)} rejected runs meet the stop threshold of {min_rejections_to_stop}")
            rationale.append("failed admission gates have zero paired wins")
        elif rejected:
            recommendation = "do_not_promote"
            rationale.append(f"{len(rejected)} rejected runs remain unresolved")
        else:
            recommendation = "no_failure_guardrail"
            rationale.append("no rejected runs were provided for this family")
        families.append(
            RecognizerFailureFamily(
                family_key=f"{backend}::{base_model}",
                backend=backend,
                base_model=base_model,
                run_count=len(family_runs),
                rejected_count=len(rejected),
                admitted_count=len(admitted),
                total_failed_gates=sum(run.failed_gate_count for run in family_runs),
                total_pairs=sum(run.total_pairs for run in family_runs),
                total_wins=total_wins,
                total_losses=total_losses,
                total_ties=sum(run.total_ties for run in family_runs),
                failed_slices=sorted({slice_name for run in family_runs for slice_name in run.failed_slices}),
                failed_metrics=sorted({metric for run in family_runs for metric in run.failed_metrics}),
                recommendation=recommendation,
                rationale=rationale,
            )
        )
    return families


def _canonical_recognizer_base_model(base_model: str) -> str:
    value = base_model.strip()
    lowered = value.lower()
    if "devanagari_pp-ocrv3_mobile_rec" in lowered:
        return "PaddlePaddle/devanagari_PP-OCRv3_mobile_rec"
    return value or "unknown"


def _failure_review_next_actions(families: list[RecognizerFailureFamily]) -> list[str]:
    if any(family.recommendation == "stop_family" for family in families):
        return [
            "Do not spend more GPU time on stopped recognizer families without a different base architecture or training objective.",
            "Move engine work toward measurable document parsing with admitted recognizer components or a document-native model.",
            "Keep these negative runs in the paper as ablation evidence, not as promoted model results.",
        ]
    if any(family.recommendation == "do_not_promote" for family in families):
        return [
            "Do not promote rejected recognizers into the engine until a new admission report passes.",
            "Add more runs or model-card bindings before declaring a family stopped.",
        ]
    return ["No rejected recognizer family was strong enough to trigger a stop recommendation."]


def _write_failure_review_markdown(path: Path, review: RecognizerFailureReview) -> None:
    lines = [
        "# Recognizer Failure Review",
        "",
        f"Admission decisions: `{len(review.admission_decisions)}`",
        f"Validation reports: `{len(review.validation_reports)}`",
        f"Model cards: `{len(review.model_cards)}`",
        f"Stop threshold: `{review.min_rejections_to_stop}` rejected runs with zero paired wins",
        "",
        "## Families",
        "",
        "| family | runs | rejected | admitted | failed gates | pairs | wins | losses | slices | recommendation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for family in review.families:
        lines.append(
            "| {family} | {runs} | {rejected} | {admitted} | {failed_gates} | {pairs} | {wins} | {losses} | {slices} | {recommendation} |".format(
                family=family.family_key.replace("|", "\\|"),
                runs=family.run_count,
                rejected=family.rejected_count,
                admitted=family.admitted_count,
                failed_gates=family.total_failed_gates,
                pairs=family.total_pairs,
                wins=family.total_wins,
                losses=family.total_losses,
                slices=",".join(family.failed_slices),
                recommendation=family.recommendation,
            )
        )
    lines.extend(["", "## Runs", ""])
    for run in review.runs:
        family = f"{run.backend}::{_canonical_recognizer_base_model(run.base_model)}"
        lines.append(
            f"- `{run.source_path}`: decision=`{run.decision}` model=`{run.model_id}` family=`{family}` wins/losses/ties=`{run.total_wins}/{run.total_losses}/{run.total_ties}`"
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in review.next_actions)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"
