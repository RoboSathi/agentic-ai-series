"""The run budget: hard caps on every resource a run can spend."""

from pydantic import BaseModel, ConfigDict, Field

# Counter names the BudgetGuard is built with (seconds is the wall clock).
SEARCHES = "searches"
FETCHES = "fetches"
TOKENS = "tokens"
ROUNDS = "rounds"
GAP_FILL_ROUNDS = "gap_fill_rounds"


class Budget(BaseModel):
    """Caps for one run. Agent config supplies the defaults and the ceiling."""

    model_config = ConfigDict(frozen=True)

    max_searches: int = Field(ge=0)
    max_fetches: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    max_seconds: float = Field(gt=0)
    max_rounds: int = Field(ge=1)
    max_dag_nodes: int = Field(ge=1)
    max_gap_fill_rounds: int = Field(ge=0)
    concurrency: int = Field(ge=1)

    def capped_by(self, ceiling: "Budget") -> "Budget":
        """This budget, lowered where it asks for more than the ceiling allows.

        A Brief may propose *lower* limits than the agent's defaults, never
        higher ones: the user can make a run cheaper, not more expensive
        than the deployment agreed to pay for.
        """
        mine, cap = self.model_dump(), ceiling.model_dump()
        return Budget.model_validate({key: min(mine[key], cap[key]) for key in mine})

    def counters(self) -> dict[str, float]:
        """The BudgetGuard caps this budget implies."""
        return {
            SEARCHES: self.max_searches,
            FETCHES: self.max_fetches,
            TOKENS: self.max_tokens,
            ROUNDS: self.max_rounds,
            GAP_FILL_ROUNDS: self.max_gap_fill_rounds,
        }
