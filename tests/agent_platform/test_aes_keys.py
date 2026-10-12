"""AES keys from the environment: raw characters, or their hex encoding."""

from agent_platform.core.primitives.keys import aes_key


def test_a_raw_key_of_a_valid_length_is_read_raw() -> None:
    """Existing deployments (`openssl rand -hex 16` = 32 characters) keep their key."""
    key = "0123456789abcdef0123456789abcdef"
    assert aes_key(key) == key.encode()


def test_a_64_character_hex_key_is_a_32_byte_key() -> None:
    key = "00" * 32
    assert aes_key(key) == bytes(32)


def test_anything_else_is_returned_as_is_for_the_caller_to_refuse() -> None:
    assert len(aes_key("short")) == 5
    assert len(aes_key("zz" * 40)) == 80
