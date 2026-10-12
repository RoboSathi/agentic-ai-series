"""Where a framework's report goes: `<reports>/<framework>-<run_id>.md`."""

from pathlib import Path
from string import Template
from typing import Any

from agent_platform.config import get_settings


def report_path(framework: str, run_id: str) -> Path:
    """The report file for one framework's view of one run."""
    reporting = get_settings().evals.reporting
    return reporting.resolved_output_dir / reporting.files[framework].format(run_id=run_id)


def write_report(framework: str, run_id: str, body: str) -> Path:
    """Write the report, replacing an earlier render of the same run."""
    path = report_path(framework, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    return path


def fill(template: Path, **values: Any) -> str:
    """A report template with its `${placeholders}` filled."""
    text = template.read_text(encoding="utf-8")
    return Template(text).substitute({key: str(value) for key, value in values.items()})
