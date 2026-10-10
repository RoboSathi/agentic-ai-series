"""Reading and checking AGENT.md and SKILL.md files (ADR 0010).

A skill is YAML frontmatter (what it is for, when it applies) over a
markdown body (what to do). Both files are checked for the sections the
configuration requires, and anything malformed fails the build: an agent
quietly running without the procedure its author wrote is worse than an
agent that will not start.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_platform.config.settings import SkillSettings
from agent_platform.skills.types import AgentProfile, SkillDocument


class SkillDocumentError(ValueError):
    """A profile or skill file that cannot be used as written."""

    def __init__(self, origin: str, reason: str) -> None:
        super().__init__(f"{origin}: {reason}")
        self.origin = origin
        self.reason = reason


class _Frontmatter(BaseModel):
    """The keys a skill may declare — and only these."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    triggers: list[str] = []
    keywords: list[str] = []
    prefetch: list[str] = []
    tools: list[str] = []


def load_profile(settings: SkillSettings) -> AgentProfile:
    """The agent's AGENT.md, checked for its sections and digested."""
    path = settings.resolved_profile_path
    if not path.is_file():
        raise SkillDocumentError(str(path), "no profile file (skills.profile_path)")
    text = path.read_text(encoding="utf-8")
    _require_sections(text, settings.profile_sections, settings, str(path))
    _refuse_internal_tool_names(text, settings, str(path))
    return AgentProfile(text=text, digest=digest(text, settings))


def load_skill_documents(settings: SkillSettings) -> list[SkillDocument]:
    """Every authored skill under the agent's skills directory, by name."""
    root = settings.resolved_skills_dir
    if not root.is_dir():
        return []
    folders = sorted(path.parent for path in root.glob(f"*/{settings.skill_file}"))
    return [_load_folder(folder, settings) for folder in folders]


def parse_stored(text: str, key: str, settings: SkillSettings) -> SkillDocument:
    """A procedure read back from memory: an authored skill, or a lesson.

    A consolidated lesson is one sentence with no frontmatter. It has no
    triggers, so only a query that names its subject selects it.
    Sections are not checked again: the text was judged when it was
    authored, and a stored version must stay readable after the required
    sections change.
    """
    if text.startswith(settings.frontmatter_delimiter):
        return _parse(text, settings, key)
    return SkillDocument(name=key, description=text, body=text, raw=text, prompt=text)


def parse_skill(raw: str, settings: SkillSettings, origin: str) -> SkillDocument:
    """One authored skill's text, frontmatter validated and sections checked."""
    document = _parse(raw, settings, origin)
    _require_sections(document.body, settings.skill_sections, settings, origin)
    _refuse_internal_tool_names(raw, settings, origin)
    return document


def digest(text: str, settings: SkillSettings) -> str:
    """A short content hash: the version of something that is not stored."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: settings.digest_chars]


def _parse(raw: str, settings: SkillSettings, origin: str) -> SkillDocument:
    meta, body = _split(raw, settings.frontmatter_delimiter, origin)
    front = _frontmatter(meta, origin)
    _check_identity(front, settings, origin)
    prompt = _sent_sections(body.strip(), settings)
    return SkillDocument(
        name=front.name,
        description=front.description.strip(),
        triggers=tuple(front.triggers),
        keywords=tuple(front.keywords),
        prefetch=tuple(front.prefetch),
        tools=tuple(front.tools),
        body=body.strip(),
        prompt=prompt,
        steps=len(re.findall(settings.step_pattern, prompt, flags=re.MULTILINE)),
        raw=raw,
    )


def _sent_sections(body: str, settings: SkillSettings) -> str:
    """Only the sections the model acts on, each with its heading, in file order."""
    heading = re.compile(settings.section_pattern, flags=re.MULTILINE)
    starts = list(heading.finditer(body))
    if not starts:
        return body
    ends = [match.start() for match in starts[1:]] + [len(body)]
    kept = [
        body[match.start() : end].strip()
        for match, end in zip(starts, ends, strict=True)
        if match.group(1) in settings.render.sections_sent
    ]
    return settings.render.separator.join(kept)


def _refuse_internal_tool_names(text: str, settings: SkillSettings, origin: str) -> None:
    """The model is offered model names; a `namespace.name` in its text names nothing."""
    found = re.findall(settings.internal_tool_reference, text)
    if found:
        reason = f"names tools by internal name ({', '.join(found)}); use model names"
        raise SkillDocumentError(origin, reason)


def _load_folder(folder: Path, settings: SkillSettings) -> SkillDocument:
    path = folder / settings.skill_file
    document = parse_skill(_with_references(folder, settings), settings, str(path))
    if document.name != folder.name:
        raise SkillDocumentError(str(path), f"name '{document.name}' is not its folder")
    return document


def _with_references(folder: Path, settings: SkillSettings) -> str:
    """The skill file with its references appended, so one digest covers both."""
    parts = [(folder / settings.skill_file).read_text(encoding="utf-8").rstrip()]
    for reference in sorted((folder / settings.references_dir).glob(settings.references_glob)):
        heading = settings.render.reference_heading.format(name=reference.stem)
        parts += [heading, reference.read_text(encoding="utf-8").strip()]
    return settings.render.separator.join(parts) + "\n"


def _split(raw: str, delimiter: str, origin: str) -> tuple[Any, str]:
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != delimiter:
        raise SkillDocumentError(origin, f"no frontmatter (open it with '{delimiter}')")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == delimiter:
            return yaml.safe_load("".join(lines[1:index])), "".join(lines[index + 1 :])
    raise SkillDocumentError(origin, f"frontmatter is never closed with '{delimiter}'")


def _frontmatter(meta: Any, origin: str) -> _Frontmatter:
    if not isinstance(meta, dict):
        raise SkillDocumentError(origin, "frontmatter is not a mapping of keys")
    try:
        return _Frontmatter.model_validate(meta)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        raise SkillDocumentError(origin, problems) from error


def _check_identity(front: _Frontmatter, settings: SkillSettings, origin: str) -> None:
    if len(front.name) > settings.max_name_chars or not re.fullmatch(
        settings.name_pattern, front.name
    ):
        raise SkillDocumentError(origin, f"name '{front.name}' must match {settings.name_pattern}")
    if not front.description.strip() or len(front.description) > settings.max_description_chars:
        raise SkillDocumentError(
            origin, f"description must be 1–{settings.max_description_chars} characters"
        )


def _require_sections(text: str, required: list[str], settings: SkillSettings, origin: str) -> None:
    present = set(re.findall(settings.section_pattern, text, flags=re.MULTILINE))
    missing = [section for section in required if section not in present]
    if missing:
        raise SkillDocumentError(origin, f"missing section(s): {', '.join(missing)}")
