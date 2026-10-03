"""Semantic chunking — split where the meaning changes, not every N tokens.

Two passes, cheapest signal first:

1. **Structure.** Markdown headings are boundaries an author already
   drew. A passage never straddles two sections, and it keeps its heading
   path, because "14 days" means nothing without "Returns > Window".
2. **Meaning.** Inside a section, adjacent sentences are embedded and a
   boundary falls wherever the cosine distance across a gap is above the
   configured percentile of that section's gaps — the topic shifted.

Token bounds then keep passages useful: a fragment below `min_tokens` is
folded into its neighbour, and a run above `max_tokens` is split at
sentence boundaries, so no passage is too thin to answer anything or too
long to rank sharply.
"""

import hashlib
import math
import re

from agent_platform.config.settings import RagChunkingSettings
from agent_platform.llm.embeddings import EmbeddingClient, cosine_similarity
from agent_platform.memory.semantic.rag.types import HEADING_SEPARATOR, Document, Passage

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# End of sentence: terminal punctuation, then space, then something that
# starts a sentence. Keeps "e.g. this" and "$12.50" intact.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*-])")

Section = tuple[tuple[str, ...], list[str]]


class SemanticChunker:
    """Heading-aware, embedding-breakpoint chunker (see module docstring)."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        settings: RagChunkingSettings,
        chars_per_token: int,
        batch_size: int,
    ) -> None:
        self._embedder = embedder
        self._settings = settings
        self._chars_per_token = chars_per_token
        self._batch_size = batch_size

    @property
    def signature(self) -> str:
        """Changes whenever a setting that moves a boundary changes."""
        payload = (
            f"{type(self).__name__}:{self._settings.model_dump_json()}:{self._chars_per_token}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    async def chunk(self, document: Document) -> list[Passage]:
        """The document's passages, in reading order."""
        sections = [s for s in _sections(document.content) if s[1]]
        sentences = [sentence for _, section in sections for sentence in section]
        vectors = await self._embed(sentences)
        groups: list[tuple[tuple[str, ...], str]] = []
        offset = 0
        for headings, section in sections:
            section_vectors = vectors[offset : offset + len(section)]
            offset += len(section)
            groups.extend(
                (headings, text) for text in self._split_section(section, section_vectors)
            )
        return [self._passage(document, index, *group) for index, group in enumerate(groups)]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embedder.embed(batch, "passage"))
        return vectors

    def _split_section(self, sentences: list[str], vectors: list[list[float]]) -> list[str]:
        """One section's sentences, grouped at semantic breakpoints and bounded."""
        gaps = [1.0 - cosine_similarity(a, b) for a, b in zip(vectors, vectors[1:])]
        threshold = _percentile(gaps, self._settings.breakpoint_percentile)
        groups: list[list[str]] = [[sentences[0]]]
        for sentence, gap in zip(sentences[1:], gaps):
            if gap > threshold:
                groups.append([])
            groups[-1].append(sentence)
        return self._merge_small(self._split_large(groups))

    def _split_large(self, groups: list[list[str]]) -> list[list[str]]:
        """Break any group over `max_tokens` at sentence boundaries."""
        bounded: list[list[str]] = []
        for group in groups:
            current: list[str] = []
            for sentence in group:
                if (
                    current
                    and self._tokens(" ".join([*current, sentence])) > self._settings.max_tokens
                ):
                    bounded.append(current)
                    current = []
                current.append(sentence)
            bounded.append(current)
        return bounded

    def _merge_small(self, groups: list[list[str]]) -> list[str]:
        """Fold any group under `min_tokens` into the one before it."""
        merged: list[str] = []
        for group in groups:
            text = " ".join(group)
            if merged and self._tokens(text) < self._settings.min_tokens:
                merged[-1] = f"{merged[-1]} {text}"
            else:
                merged.append(text)
        if len(merged) > 1 and self._tokens(merged[0]) < self._settings.min_tokens:
            merged[0:2] = [f"{merged[0]} {merged[1]}"]
        return merged

    def _passage(
        self, document: Document, ordinal: int, headings: tuple[str, ...], text: str
    ) -> Passage:
        # The heading path is part of what is embedded and matched: a
        # passage reading "within 14 days" is only findable as a returns
        # answer if it says it is one.
        content = f"{HEADING_SEPARATOR.join(headings)}\n{text}" if headings else text
        return Passage(
            passage_id=f"{document.path}#{ordinal}",
            document_path=document.path,
            ordinal=ordinal,
            heading_path=headings,
            content=content,
            token_estimate=self._tokens(content),
        )

    def _tokens(self, text: str) -> int:
        return math.ceil(len(text) / self._chars_per_token)


def markdown_sections(markdown: str) -> list[str]:
    """Each non-empty section as "Heading > Sub" plus its body — structure only.

    The first pass of the chunker, without the embedding pass: for a caller
    that needs an author's sections as they stand (grounding evidence, say)
    and must not spend an embedding call per sentence to get them.
    """
    return [
        f"{HEADING_SEPARATOR.join(path)}\n{' '.join(sentences)}" if path else " ".join(sentences)
        for path, sentences in _sections(markdown)
        if sentences
    ]


def _sections(markdown: str) -> list[Section]:
    """Split markdown into (heading path, sentences) per section."""
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    lines: list[str] = []
    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match is None:
            lines.append(line)
            continue
        sections.append((tuple(title for _, title in path), _sentences(lines)))
        level = len(match.group(1))
        path = [entry for entry in path if entry[0] < level] + [(level, match.group(2))]
        lines = []
    sections.append((tuple(title for _, title in path), _sentences(lines)))
    return sections


def _sentences(lines: list[str]) -> list[str]:
    """Sentences from a section body; each list item or line is at least one."""
    sentences: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            sentences.extend(part.strip() for part in _SENTENCE_END.split(stripped) if part.strip())
    return sentences


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile; infinite when there is nothing to split."""
    if not values:
        return math.inf
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
