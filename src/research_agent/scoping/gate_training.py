"""Fine-tune the scope gate's encoder on `gate_training.yaml` (`make train-gate ra`).

    python -m research_agent.scoping.gate_training --base distilbert-base-uncased

Needs the platform's `torch` and `transformers`. Holds out a stratified share of the examples,
prints accuracy and a confusion matrix on it, and saves the model to
`research_agent.gate_encoder.local_model_dir`. Publish it and set `gate_encoder.model` (and turn
`features.gate_encoder` on) to use it; until then the LLM gate answers every message.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml

from research_agent.scoping.encoder_gate import LABELS


def load_examples(path: Path) -> list[tuple[str, str]]:
    """(text, label) pairs; a label outside the gate's set is an error, not a skipped row."""
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


def main() -> None:  # pragma: no cover - needs torch and a model download
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from research_agent.config import get_agent_settings

    cfg = get_agent_settings().gate_encoder
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    train, test = stratified_split(load_examples(Path(cfg.training_data_path)))
    label2id: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label2id
    )

    class Rows(torch.utils.data.Dataset):  # type: ignore[type-arg]
        def __init__(self, rows: list[tuple[str, str]]) -> None:
            self.enc = tokenizer([t for t, _ in rows], truncation=True, padding=True, max_length=64)
            self.y = [label2id[label] for _, label in rows]

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, i: int) -> dict[str, object]:
            return {**{k: torch.tensor(v[i]) for k, v in self.enc.items()}, "labels": self.y[i]}

    out = Path(cfg.local_model_dir)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "runs"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            learning_rate=5e-5,
            report_to=[],
            save_strategy="no",
        ),  # fmt: skip
        train_dataset=Rows(train),
    )
    trainer.train()
    logits = trainer.predict(Rows(test)).predictions
    predicted = [LABELS[i] for i in np.argmax(logits, axis=1)]
    true = [label for _, label in test]
    accuracy = sum(p == t for p, t in zip(predicted, true, strict=True)) / len(true)
    print(
        f"held-out accuracy: {accuracy:.1%} on {len(test)} examples\n{confusion(true, predicted)}"
    )
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
