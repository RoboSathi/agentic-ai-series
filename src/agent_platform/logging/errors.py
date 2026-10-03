"""An exception as one line a person can read, the same way everywhere."""


def describe_error(exc: BaseException) -> str:
    """`TimeoutError`, or `HTTPStatusError: 503 Service Unavailable` when it says more."""
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
