"""Dialogue summaries from a repository-local seq2seq model.

Used by memory dreams to say, in one sentence, what a customer wanted in
a session. Local for the same reason the NLI and intent encoders are:
a dream reads every session, and a remote call per session is the cost
this exists to avoid. It degrades rather than raises — a model that
cannot load latches off once, and the dream falls back to its exact
template summary instead of failing.
"""

import asyncio

from agent_platform.config import get_settings
from agent_platform.config.settings import DialogueSummarizerSettings
from agent_platform.llm.local_models import LocalSequenceClassifier
from agent_platform.logging import get_logger

_SEQ2SEQ = "AutoModelForSeq2SeqLM"


class LocalDialogueSummarizer:
    """Summarises a transcript with a local encoder-decoder (BART-SAMSum by default)."""

    def __init__(self, settings: DialogueSummarizerSettings | None = None) -> None:
        self._settings = settings or get_settings().llm.dialogue_summarizer
        self._artifact = LocalSequenceClassifier(
            model_name=self._settings.model,
            local_dir=self._settings.resolved_local_model_dir,
            download_if_missing=self._settings.download_if_missing,
            purpose="dialogue summarizer",
            auto_class=_SEQ2SEQ,
        )
        self._unavailable = not self._settings.enabled
        self._logger = get_logger("llm")

    @property
    def model_name(self) -> str:
        return self._artifact.model_name

    def ensure_loaded_sync(self) -> None:
        """Load now, in this thread (a probe or a console warming up before its event loop)."""
        self._artifact.load_sync()

    async def summarize(self, dialogue: str) -> str | None:
        """One summary of the dialogue, or None when the model is unavailable."""
        if self._unavailable or not dialogue.strip():
            return None
        try:
            tokenizer, model, torch = await self._artifact.load()
        except Exception as exc:
            self._unavailable = True
            self._logger.warning("Dialogue summarizer unavailable; template only", error=str(exc))
            return None
        return await asyncio.to_thread(self._generate, tokenizer, model, torch, dialogue)

    def _generate(self, tokenizer: object, model: object, torch: object, dialogue: str) -> str:
        inputs = tokenizer(  # type: ignore[operator]
            dialogue,
            return_tensors="pt",
            truncation=True,
            max_length=self._settings.max_input_tokens,
        )
        with torch.no_grad():  # type: ignore[attr-defined]
            output = model.generate(  # type: ignore[attr-defined]
                **inputs,
                max_new_tokens=self._settings.max_summary_tokens,
                num_beams=self._settings.num_beams,
            )
        return str(tokenizer.decode(output[0], skip_special_tokens=True)).strip()  # type: ignore[attr-defined]
