from __future__ import annotations

"""Shared MCP tool metadata, discovery, and intent-driven planning."""

from collections import Counter
from dataclasses import dataclass
import inspect
import math
import re
from typing import Any


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_INFRASTRUCTURE_PARAMETERS = frozenset({"device", "output_dir", "export_dir"})
_PLAN_STAGES = {
    "prepare": 0,
    "inspect": 1,
    "workflow": 2,
    "simulate": 3,
    "analyze": 4,
    "fit": 5,
    "compare": 6,
    "export": 7,
    "other": 8,
}


def _tokenize(text: str) -> tuple[str, ...]:
    normalized = text.replace("_", " ").lower()
    return tuple(_TOKEN_PATTERN.findall(normalized))


def _cosine_similarity(
    left: dict[str, float],
    right: dict[str, float],
    *,
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(weight * right.get(token, 0.0) for token, weight in left.items())
    if dot <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _join_phrases(phrases: tuple[str, ...]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} or {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, or {phrases[-1]}"


def _humanized_tool_phrase(name: str) -> str:
    words = name.replace("_", " ")
    if name.endswith("_workflow"):
        return f"the user wants an end-to-end {words[:-9].strip()} workflow"
    if name.endswith("_pipeline"):
        return f"the user wants an integrated {words[:-9].strip()} pipeline"
    if name.endswith("_report"):
        return f"the user wants a {words} result"
    if name.endswith("_analysis"):
        return f"the user wants {words}"
    for prefix in ("inspect", "describe", "validate", "list", "read", "write", "create", "upload", "delete"):
        if name.startswith(f"{prefix}_"):
            return f"the user wants to {words}"
    for prefix in (
        "simulate",
        "project",
        "analyze",
        "compute",
        "check",
        "compare",
        "solve",
        "fit",
        "infer",
        "design",
        "optimize",
        "train",
        "tune",
        "query",
        "export",
        "materialize",
        "pivot",
        "unpivot",
        "interpolate",
        "window",
    ):
        if name.startswith(f"{prefix}_"):
            return f"the user wants to {words}"
    return f"the user needs the {words} capability"


def _infer_plan_stage(name: str) -> str:
    if name in {"guide_workflow", "plan_experiment"}:
        return "workflow"
    if name.endswith(("_workflow", "_pipeline", "_experiment")):
        return "workflow"
    if name.startswith(("inspect_", "describe_", "supported_", "validate_", "list_", "server_info", "read_")):
        return "inspect"
    if name.startswith(("write_", "create_", "upload_", "bootstrap_", "delete_")):
        return "prepare"
    if name.startswith(("simulate_", "project_", "solve_", "split_operator_")):
        return "simulate"
    if name.startswith(("fit_", "infer_", "design_", "optimize_", "train_", "tune_", "execute_")):
        return "fit"
    if name.startswith(("compare_",)):
        return "compare"
    if name.startswith(("export_", "materialize_", "compress_", "report_")):
        return "export"
    if name.startswith(
        (
            "analyze_",
            "compute_",
            "check_",
            "momentum_",
            "fourier_",
            "pade_",
            "hilbert_",
            "correlation_",
            "richardson_",
            "kramers_",
            "spectral_",
            "scattering_",
            "berry_",
            "quantum_",
            "perturbation_",
            "wkb_",
            "operator_",
            "symplectic_",
            "decompose_",
            "detect_",
            "estimate_",
            "probe_",
            "self_test",
        )
    ):
        return "analyze"
    return "other"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    base_description: str
    intent_phrases: tuple[str, ...]
    bounded: bool
    stage: str
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...]

    @property
    def document(self) -> str:
        intent = " ".join(self.intent_phrases)
        parts = [self.name.replace("_", " "), self.base_description]
        if intent:
            parts.append(intent)
        parts.append(self.stage)
        return " ".join(parts)

    @property
    def intent_description(self) -> str:
        intent = self.intent_phrases or (_humanized_tool_phrase(self.name),)
        return f"Use when {_join_phrases(intent)}. {self.base_description}"


@dataclass(frozen=True, slots=True)
class ToolRank:
    tool: str
    score: float
    coverage: float
    matched_terms: tuple[str, ...]
    stage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "score": round(self.score, 6),
            "coverage": round(self.coverage, 6),
            "matched_terms": list(self.matched_terms),
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class ToolPlanStep:
    order: int
    tool: str
    stage: str
    purpose: str
    matched_terms: tuple[str, ...]
    required_parameters: tuple[str, ...]
    optional_parameters: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "tool": self.tool,
            "stage": self.stage,
            "purpose": self.purpose,
            "matched_terms": list(self.matched_terms),
            "required_parameters": list(self.required_parameters),
            "optional_parameters": list(self.optional_parameters),
            "parameter_template": {name: None for name in self.required_parameters},
        }


@dataclass(frozen=True, slots=True)
class ToolPlanningResult:
    intent: str
    anchor_tool: str | None
    plan_steps: tuple[ToolPlanStep, ...]
    considered_tools: tuple[ToolRank, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "anchor_tool": self.anchor_tool,
            "plan_steps": [step.to_dict() for step in self.plan_steps],
            "considered_tools": [candidate.to_dict() for candidate in self.considered_tools],
            "notes": list(self.notes),
        }


class ToolCatalog:
    """Inspectable registry for MCP tool metadata and retrieval-based planning."""

    def __init__(self) -> None:
        self._descriptors: dict[str, ToolDescriptor] = {}
        self._analysis_cache: tuple[
            dict[str, dict[str, float]],
            dict[str, float],
            dict[str, dict[str, float]],
        ] | None = None

    def register(
        self,
        name: str,
        base_description: str,
        function: Any,
        *,
        bounded: bool = False,
        intent_phrases: tuple[str, ...] = (),
    ) -> ToolDescriptor:
        signature = inspect.signature(function)
        required_parameters: list[str] = []
        optional_parameters: list[str] = []
        for parameter in signature.parameters.values():
            if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if parameter.name in _INFRASTRUCTURE_PARAMETERS:
                continue
            if parameter.default is inspect.Signature.empty:
                required_parameters.append(parameter.name)
            else:
                optional_parameters.append(parameter.name)
        descriptor = ToolDescriptor(
            name=name,
            base_description=base_description.strip(),
            intent_phrases=tuple(phrase.strip() for phrase in intent_phrases if phrase.strip()),
            bounded=bounded,
            stage=_infer_plan_stage(name),
            required_parameters=tuple(required_parameters),
            optional_parameters=tuple(optional_parameters),
        )
        self._descriptors[name] = descriptor
        self._analysis_cache = None
        return descriptor

    def describe(self, name: str, fallback: str) -> str:
        descriptor = self._descriptors.get(name)
        if descriptor is None:
            return fallback
        return descriptor.intent_description

    def related_tools(self, name: str, *, limit: int = 5) -> tuple[str, ...]:
        descriptor = self._descriptors.get(name)
        if descriptor is None:
            return ()
        weights, norms, _ = self._analysis()
        source_weights = weights.get(name, {})
        source_norm = norms.get(name, 0.0)
        ranked: list[tuple[str, float]] = []
        for other_name, other_descriptor in self._descriptors.items():
            if other_name == name or other_descriptor.stage == "inspect":
                continue
            score = _cosine_similarity(
                source_weights,
                weights.get(other_name, {}),
                left_norm=source_norm,
                right_norm=norms.get(other_name, 0.0),
            )
            if score > 0.0:
                ranked.append((other_name, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return tuple(other_name for other_name, _ in ranked[:limit])

    def rank_tools(self, intent: str, *, limit: int = 8) -> tuple[ToolRank, ...]:
        weights, norms, idf = self._analysis()
        query_tokens = _tokenize(intent)
        if not query_tokens:
            return ()
        query_counter = Counter(query_tokens)
        query_weights = {
            token: float(count) * idf.get(token, 0.0)
            for token, count in query_counter.items()
            if idf.get(token, 0.0) > 0.0
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0.0:
            return ()
        ranked: list[ToolRank] = []
        for name, descriptor in self._descriptors.items():
            score = _cosine_similarity(
                query_weights,
                weights.get(name, {}),
                left_norm=query_norm,
                right_norm=norms.get(name, 0.0),
            )
            if score <= 0.0:
                continue
            overlap = set(query_weights).intersection(weights.get(name, {}))
            coverage = sum(query_weights[token] for token in overlap)
            matched_terms = tuple(
                token
                for token, _ in sorted(
                    ((token, query_weights[token] * weights[name][token]) for token in overlap),
                    key=lambda item: (-item[1], item[0]),
                )[:4]
            )
            ranked.append(
                ToolRank(
                    tool=name,
                    score=score,
                    coverage=coverage,
                    matched_terms=matched_terms,
                    stage=descriptor.stage,
                )
            )
        ranked.sort(
            key=lambda item: (
                -item.coverage,
                -item.score,
                _PLAN_STAGES.get(item.stage, 99),
                item.tool,
            )
        )
        return tuple(ranked[:limit])

    def plan(self, intent: str, *, max_steps: int = 4) -> ToolPlanningResult:
        ranked = self.rank_tools(intent, limit=12)
        if not ranked:
            return ToolPlanningResult(
                intent=intent,
                anchor_tool=None,
                plan_steps=(),
                considered_tools=(),
                notes=(
                    "No tool matched the supplied intent strongly enough to build a plan.",
                    "Try naming the physical task, data source, or desired diagnostic more explicitly.",
                ),
            )
        anchor = self._select_anchor(ranked)
        steps: list[ToolPlanStep] = []
        used_tools: set[str] = set()

        preparatory = self._select_preparatory_step(anchor, ranked)
        if preparatory is not None:
            steps.append(self._build_step(preparatory, order=len(steps) + 1, anchor=anchor, role="prepare"))
            used_tools.add(preparatory.tool)

        steps.append(self._build_step(anchor, order=len(steps) + 1, anchor=anchor, role="anchor"))
        used_tools.add(anchor.tool)

        follow_ups = self._select_follow_up_steps(anchor, ranked, used_tools, limit=max_steps - len(steps))
        for follow_up in follow_ups:
            steps.append(self._build_step(follow_up, order=len(steps) + 1, anchor=anchor, role="follow_up"))
            used_tools.add(follow_up.tool)

        return ToolPlanningResult(
            intent=intent,
            anchor_tool=anchor.tool,
            plan_steps=tuple(steps),
            considered_tools=ranked[:5],
            notes=(
                "The planner ranks tools from their capability documents and intent phrases, not from hand-written tool chains.",
                "It proposes tool order only. Parameter values remain blank for the caller or model to choose.",
            ),
        )

    def _analysis(
        self,
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, dict[str, float]]]:
        if self._analysis_cache is not None:
            return self._analysis_cache
        documents = {name: _tokenize(descriptor.document) for name, descriptor in self._descriptors.items()}
        document_frequency: Counter[str] = Counter()
        for tokens in documents.values():
            document_frequency.update(set(tokens))
        doc_count = max(len(documents), 1)
        idf = {
            token: math.log((1.0 + doc_count) / (1.0 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        weights: dict[str, dict[str, float]] = {}
        norms: dict[str, float] = {}
        for name, tokens in documents.items():
            counts = Counter(tokens)
            vector = {token: float(count) * idf[token] for token, count in counts.items()}
            weights[name] = vector
            norms[name] = math.sqrt(sum(weight * weight for weight in vector.values()))
        self._analysis_cache = (weights, norms, idf)
        return self._analysis_cache

    def _build_step(self, rank: ToolRank, *, order: int, anchor: ToolRank, role: str) -> ToolPlanStep:
        descriptor = self._descriptors[rank.tool]
        if role == "prepare":
            purpose = (
                f"Prepare or validate the request before `{anchor.tool}`. "
                f"Evidence overlap: {', '.join(rank.matched_terms) or 'generic setup'}."
            )
        elif role == "anchor":
            purpose = (
                f"Primary step for this intent. "
                f"Best overlap: {', '.join(rank.matched_terms) or 'broad capability match'}."
            )
        else:
            purpose = (
                f"Follow-up diagnostic that complements `{anchor.tool}`. "
                f"Evidence overlap: {', '.join(rank.matched_terms) or 'shared capability terms'}."
            )
        return ToolPlanStep(
            order=order,
            tool=rank.tool,
            stage=descriptor.stage,
            purpose=purpose,
            matched_terms=rank.matched_terms,
            required_parameters=descriptor.required_parameters,
            optional_parameters=descriptor.optional_parameters[:6],
        )

    def _select_anchor(self, ranked: tuple[ToolRank, ...]) -> ToolRank:
        substantive = [
            candidate
            for candidate in ranked
            if candidate.tool != "plan_experiment" and candidate.stage not in {"inspect", "prepare"}
        ]
        if substantive:
            return max(
                substantive,
                key=lambda candidate: (
                    candidate.coverage,
                    1 if candidate.stage == "workflow" else 0,
                    candidate.score,
                    -_PLAN_STAGES.get(candidate.stage, 99),
                ),
            )
        return ranked[0]

    def _select_preparatory_step(
        self,
        anchor: ToolRank,
        ranked: tuple[ToolRank, ...],
    ) -> ToolRank | None:
        if anchor.stage in {"prepare", "inspect"}:
            return None
        for candidate in ranked:
            if candidate.tool == anchor.tool:
                continue
            if candidate.stage not in {"prepare", "inspect"}:
                continue
            if candidate.score >= anchor.score * 0.6:
                return candidate
        return None

    def _select_follow_up_steps(
        self,
        anchor: ToolRank,
        ranked: tuple[ToolRank, ...],
        used_tools: set[str],
        *,
        limit: int,
    ) -> tuple[ToolRank, ...]:
        if limit <= 0:
            return ()
        related = self.related_tools(anchor.tool, limit=max(6, limit + 3))
        related_set = set(related)
        ranked_map = {candidate.tool: candidate for candidate in ranked}
        weights, norms, _ = self._analysis()
        anchor_weights = weights.get(anchor.tool, {})
        anchor_norm = norms.get(anchor.tool, 0.0)
        candidates: list[tuple[float, ToolRank]] = []
        for tool_name in related:
            if tool_name in used_tools:
                continue
            candidate = ranked_map.get(tool_name)
            if candidate is None:
                continue
            if candidate.stage in {"prepare", "inspect"}:
                continue
            if anchor.stage == "workflow" and candidate.stage == "workflow":
                continue
            related_score = _cosine_similarity(
                anchor_weights,
                weights.get(tool_name, {}),
                left_norm=anchor_norm,
                right_norm=norms.get(tool_name, 0.0),
            )
            combined_score = (0.65 * candidate.score) + (0.35 * related_score)
            candidates.append((combined_score, candidate))
        for candidate in ranked:
            if len(candidates) >= max(limit * 2, limit + 2):
                break
            if candidate.tool in used_tools or candidate.tool in related_set:
                continue
            if candidate.stage in {"prepare", "inspect"}:
                continue
            if anchor.stage == "workflow" and candidate.stage == "workflow":
                continue
            candidates.append((candidate.score * 0.5, candidate))
        candidates.sort(
            key=lambda item: (
                -item[0],
                _PLAN_STAGES.get(item[1].stage, 99),
                item[1].tool,
            )
        )
        selected: list[ToolRank] = []
        seen: set[str] = set()
        for _, candidate in candidates:
            if candidate.tool in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.tool)
            if len(selected) >= limit:
                break
        return tuple(selected)


__all__ = [
    "ToolCatalog",
    "ToolDescriptor",
    "ToolPlanStep",
    "ToolPlanningResult",
    "ToolRank",
]
