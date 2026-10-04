"""The research kind's labelled examples (`kind_examples.yaml`), and how to score a classifier on them.

The examples are what the intent classifier votes with (nearest labelled examples over the local
embedder); these helpers load them, split them by label, and print a confusion matrix — for
measuring a classifier, not for training one.
"""

import random
from collections import defaultdict
from pathlib import Path

import yaml

from research_agent.scoping.kind_encoder import LABELS


def load_examples(path: Path) -> list[tuple[str, str]]:
    """(text, label) pairs; a label outside the research kinds is an error, not a skipped row."""
    raw = yaml.safe_load(path.read_text())
    unknown = sorted(set(raw) - set(LABELS))
    if unknown:
        raise ValueError(f"unknown label(s) in {path.name}: {unknown}; expected {list(LABELS)}")
    bad = [text for texts in raw.values() for text in texts if not isinstance(text, str)]
    if bad:  # usually an unquoted "label: text" line that YAML read as a mapping
        raise ValueError(
            f"non-text example(s) in {path.name}: {bad[:3]} — quote lines containing ': '"
        )
    return [(text.strip(), label) for label, texts in raw.items() for text in texts]


def stratified_split(
    examples: list[tuple[str, str]], held_out: float = 0.2, seed: int = 7
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Train and test sets with every label in both (at least one test example per label)."""
    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for example in examples:
        by_label[example[1]].append(example)
    rng = random.Random(seed)
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        cut = max(1, round(len(group) * held_out))
        test += group[:cut]
        train += group[cut:]
    return train, test


def confusion(true: list[str], predicted: list[str]) -> str:
    cells: dict[tuple[str, str], int] = defaultdict(int)
    for t, p in zip(true, predicted, strict=True):
        cells[(t, p)] += 1
    width = max(len(label) for label in LABELS)
    rows = [" " * width + "  " + " ".join(f"{label[:5]:>5}" for label in LABELS)]
    for t in LABELS:
        rows.append(f"{t:<{width}}  " + " ".join(f"{cells[(t, p)]:>5}" for p in LABELS))
    return "\n".join(rows)
