"""Permission strings with segment wildcards (LLD §12.1).

A permission is `a:b:c:d`; a grant may put `*` in any segment. Matching
is per segment, not a glob over the whole string, so `memory:*:erase:*`
can never be satisfied by a permission that merely *contains* "erase".
"""

from collections.abc import Iterable

_SEPARATOR = ":"
_WILDCARD = "*"


def permission_granted(grants: Iterable[str], required: str) -> str | None:
    """The first grant that covers `required`, or None."""
    wanted = required.split(_SEPARATOR)
    for grant in grants:
        if _covers(grant.split(_SEPARATOR), wanted):
            return grant
    return None


def _covers(grant: list[str], wanted: list[str]) -> bool:
    if len(grant) != len(wanted):
        return False
    return all(have in (_WILDCARD, need) for have, need in zip(grant, wanted))
