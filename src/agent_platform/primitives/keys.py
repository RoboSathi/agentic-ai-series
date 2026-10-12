"""AES keys read from the environment: the raw characters, or their hex encoding.

A key was always read as its raw characters (`openssl rand -hex 16` gives 32 characters:
a 32-byte AES-256 key). `openssl rand -hex 32` gives 64 characters — no valid raw length,
but exactly the hex encoding of a 32-byte key — and used to leave checkpoints unencrypted.
A raw key of a valid length is still read raw, so existing encrypted data stays readable.
"""

AES_KEY_LENGTHS = (16, 24, 32)


def aes_key(value: str) -> bytes:
    """The key's bytes: raw when that is a valid AES length, else hex-decoded when that is."""
    raw = value.encode("utf-8")
    if len(raw) in AES_KEY_LENGTHS:
        return raw
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        return raw
    return decoded if len(decoded) in AES_KEY_LENGTHS else raw
