"""Minimal user-defined policy example: prioritize longer units."""

from __future__ import annotations

from claimreplay.models import PolicyView
from claimreplay.policies import RankedUnit, VerificationPolicy


class LongestUnitFirstPolicy(VerificationPolicy):
    name = "example:longest-unit-first"

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        rows = (RankedUnit(unit.unit_id, float(len(unit.text))) for unit in response.units)
        return tuple(sorted(rows, key=lambda row: (-float(row.score), row.unit_id)))

