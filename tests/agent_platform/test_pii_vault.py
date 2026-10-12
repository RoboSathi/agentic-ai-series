"""Presidio detection, keyed redaction tokens, and the audit vault behind them.

Tokens replace values everywhere a message goes (model, logs, memory, traces); the vault maps a
token back to its value, encrypted, and only an audited reveal reads it. Presidio's own tests
load the local spaCy pipeline and are skipped when it has not been downloaded (`make models`).
"""

from pathlib import Path

import pytest

from agent_platform.config import get_settings
from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.security.pii import PatternPIIDetector, PIIFilterImpl, SqlitePIIVault
from agent_platform.security.pii.vault import VaultKeyError, vault_key

KEY = b"0123456789abcdef0123456789abcdef"
OTHER_KEY = b"fedcba9876543210fedcba9876543210"
EMAIL = "ada@example.com"


def _vault(tmp_path: Path, key: bytes = KEY) -> SqlitePIIVault:
    return SqlitePIIVault(key, SqliteConnectionPool(str(tmp_path / "vault.db")), agent_id="test")


def _filter(tmp_path: Path, key: bytes = KEY) -> PIIFilterImpl:
    return PIIFilterImpl([PatternPIIDetector()], vault=_vault(tmp_path, key))


def test_a_value_becomes_a_category_token(tmp_path: Path) -> None:
    redacted = _filter(tmp_path).redact(f"mail {EMAIL}", "tokenize")
    assert EMAIL not in redacted and redacted.startswith("mail [EMAIL_ADDRESS:")


def test_the_same_value_gets_the_same_token_only_under_the_same_key(tmp_path: Path) -> None:
    first = _filter(tmp_path).redact(EMAIL, "tokenize")
    assert first == _filter(tmp_path).redact(EMAIL, "tokenize")
    assert first != _filter(tmp_path / "other", OTHER_KEY).redact(EMAIL, "tokenize")


def test_the_vault_reveals_what_a_token_stood_for(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    pii = PIIFilterImpl([PatternPIIDetector()], vault=vault)
    pii.redact(f"mail {EMAIL}", "tokenize")
    token = vault.token("EMAIL_ADDRESS", EMAIL)
    revealed = vault.reveal(token)
    assert revealed is not None and revealed.value == EMAIL
    assert revealed.category == "EMAIL_ADDRESS"


def test_the_stored_value_is_encrypted(tmp_path: Path) -> None:
    _filter(tmp_path).redact(EMAIL, "tokenize")
    raw = (tmp_path / "vault.db").read_bytes()
    assert EMAIL.encode() not in raw


def test_another_key_cannot_read_the_vault(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    token = vault.token("EMAIL_ADDRESS", EMAIL)
    vault.keep("EMAIL_ADDRESS", token, EMAIL)
    with pytest.raises(ValueError):
        _vault(tmp_path, OTHER_KEY).reveal(token)


def test_without_a_vault_tokenizing_masks(tmp_path: Path) -> None:
    redacted = PIIFilterImpl([PatternPIIDetector()], vault=None).redact(EMAIL, "tokenize")
    assert redacted == get_settings().security.pii.mask_token


def test_the_card_lines_carry_tokens_never_values(tmp_path: Path) -> None:
    spans = _filter(tmp_path).tokens(f"mail {EMAIL}")
    assert [span.category for span in spans] == ["EMAIL_ADDRESS"]
    assert (
        EMAIL not in str(spans) and len(spans[0].token) == get_settings().security.pii.token_chars
    )


def test_a_required_vault_without_a_key_stops_start_up(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings().security.pii.vault.model_copy(update={"required": True})
    monkeypatch.delenv(settings.key_env, raising=False)
    with pytest.raises(VaultKeyError):
        vault_key(settings)


def test_a_key_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(VaultKeyError):
        SqlitePIIVault(b"short", SqliteConnectionPool(":memory:"), agent_id="test")


# -- Presidio, on the real local pipeline ----------------------------------------------------

_PIPELINE = get_settings().security.pii.presidio.resolved_local_model_dir
needs_pipeline = pytest.mark.skipif(
    not (_PIPELINE / "config.cfg").exists(), reason="spaCy pipeline not downloaded (make models)"
)


@needs_pipeline
def test_presidio_finds_what_no_pattern_can() -> None:
    from agent_platform.config.settings import PresidioSettings
    from agent_platform.security.pii.presidio import PresidioPIIDetector

    settings: PresidioSettings = get_settings().security.pii.presidio.model_copy(
        update={"entities": ["PERSON", "LOCATION", "PHONE_NUMBER"]}
    )
    found = PresidioPIIDetector(PatternPIIDetector(), settings).detect(
        "I'm Priya Sharma from Pune, call 415-555-0100 about order 10155"
    )
    assert {(span.category, span.text) for span in found} == {
        ("PERSON", "Priya Sharma"),
        ("LOCATION", "Pune"),
        ("PHONE_NUMBER", "415-555-0100"),
    }


@needs_pipeline
def test_names_and_places_are_left_alone_unless_the_agent_lists_them() -> None:
    """A research subject ("Marie Curie") is not a customer's name: the platform's default list
    omits names and places; the support agent's own config adds them."""
    import yaml

    from agent_platform.config import REPO_ROOT
    from agent_platform.security.pii.presidio import PresidioPIIDetector

    raw = yaml.safe_load((REPO_ROOT / "src/agent_platform/config/config.yaml").read_text())
    default = raw["security"]["pii"]["presidio"]["entities"]
    settings = get_settings().security.pii.presidio.model_copy(update={"entities": default})
    text = "Marie Curie's work in Paris"
    assert PresidioPIIDetector(PatternPIIDetector(), settings).detect(text) == []
    assert {span.category for span in PresidioPIIDetector(PatternPIIDetector()).detect(text)} == {
        "PERSON",
        "LOCATION",
    }
