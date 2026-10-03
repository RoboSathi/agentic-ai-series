"""E11 — Evaluation Integrity: poisoned labels, judges, Goodhart.

Three ways an eval lies to you.

Cohen's kappa (1960): raw agreement flatters any judge on a skewed label
distribution. Always answering "supported" scores 90% when 90% of claims
are supported, and kappa is what removes the agreement chance explains.

Poisoned ground truth: a golden set labelled from what agents *did*
rather than what policy *says* certifies existing practice and punishes
correct behaviour. The provenance ledger makes that findable.

Exit artifact: the provenance ledger and a judge calibration report.
"""

from agent_platform.evals import GoldenDatasetProvenanceLedger, JudgeCalibrator, LabelProvenance
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool
from episodes._shared import EpisodeOutput, main


async def run() -> EpisodeOutput:
    """Show kappa catching a lazy judge, and the provenance of a poisoned set."""
    output = EpisodeOutput(
        episode="E11",
        title="Evaluation Integrity",
        principle="Raw agreement flatters. Kappa removes what chance explains.",
    )
    calibrator = JudgeCalibrator()

    human = ["yes"] * 9 + ["no"]
    lazy = ["yes"] * 10
    lazy_report = calibrator.agreement(lazy, human)
    output.say("[bold]A judge that always says 'yes'[/bold], on a 90/10 split")
    output.say(f"  observed agreement : {lazy_report.observed_agreement:.0%}  (looks excellent)")
    output.say(f"  expected by chance : {lazy_report.expected_agreement:.0%}")
    output.say(
        f"  Cohen's kappa      : {lazy_report.cohens_kappa:.2f}  acceptable={lazy_report.acceptable}"
    )

    real = ["yes"] * 8 + ["no", "no"]
    real_report = calibrator.agreement(real, human)
    output.say("")
    output.say("[bold]A judge that actually discriminates[/bold]")
    output.say(f"  observed agreement : {real_report.observed_agreement:.0%}")
    output.say(
        f"  Cohen's kappa      : {real_report.cohens_kappa:.2f}  acceptable={real_report.acceptable}"
    )

    pool = SqliteConnectionPool(IN_MEMORY)
    ledger = GoldenDatasetProvenanceLedger(pool)
    ledger.record(
        LabelProvenance(
            case_id="c1", label="deny", basis="historical_practice", labeller="legacy-export"
        )
    )
    ledger.record(
        LabelProvenance(case_id="c2", label="approve", basis="written_policy", labeller="reviewer")
    )
    corrected = ledger.relabel("c1", "approve", "reviewer", "written policy says approve")
    report = ledger.bias_report()

    output.say("")
    output.say("[bold]Poisoned ground truth[/bold]")
    output.say(f"  labels from historical practice : {report.practice_ratio:.0%}")
    output.say(f"  corrected, superseding {corrected.superseded!r} → {corrected.label!r}")
    output.say("  The correction is recorded, not overwritten: a corrected eval")
    output.say("  set whose corrections are invisible cannot be reviewed.")

    return output.model_copy(
        update={
            "artifact": {
                "lazy judge kappa": lazy_report.cohens_kappa,
                "real judge kappa": real_report.cohens_kappa,
                "kappa bar": "0.60",
                "labels relabelled": report.relabelled,
                "practice-based ratio": report.practice_ratio,
            }
        }
    )


if __name__ == "__main__":
    main(run)
