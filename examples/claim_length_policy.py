"""Third-party example policy; this file is outside the ClaimReplay package."""

from claimreplay.models import PolicyView
from claimreplay.policies import RankedUnit, VerificationPolicy


class ClaimLength(VerificationPolicy):
    """Rank by Unicode code-point length, then opaque unit ID."""

    name = "external:claim-length-v1"

    def rank(self, response: PolicyView, *, seed: int) -> tuple[RankedUnit, ...]:
        del seed
        rows = (RankedUnit(unit.unit_id, float(len(unit.text))) for unit in response.units)
        return tuple(sorted(rows, key=lambda row: (-float(row.score), row.unit_id)))
