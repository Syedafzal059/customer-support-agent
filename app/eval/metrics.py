"""Offline evaluation metrics (Phase 1.3)."""

from __future__ import annotations

from dataclasses import dataclass

from app.eval.schemas import EvalCase
from app.orchestrator.agent import ChatTurnOutcome


@dataclass(frozen=True)
class RoutingMetrics:
    route_matches_expected: bool
    ticket_id_matches_expected: bool | None


def score_routing(case: EvalCase, outcome: ChatTurnOutcome) -> RoutingMetrics:
    route_ok = outcome.source == case.route_expected

    if case.route_expected != "ticket":
        tid_ok: bool | None = None
    else:
        # Compare when ChatTurnOutcome carries ticket_id; until then leave unscored.
        tid_ok = None

    return RoutingMetrics(
        route_matches_expected=route_ok,
        ticket_id_matches_expected=tid_ok,
    )
