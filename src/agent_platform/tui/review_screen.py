"""The console's Review screen: the one human-review queue, decided by pointing (ADR 0012).

The same items, scope and service `/queue` uses — this screen is a second
way in, never a second source of truth. Items are grouped by kind; the
selected one is shown in full; Approve and Deny stay disabled until the
comment box holds a reason, because every decision needs one.
"""

from collections.abc import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static

from agent_platform.commands.review import QueueCommand
from agent_platform.config import get_settings
from agent_platform.review import ReviewItem
from agent_platform.review.service import ReviewError, ReviewService

APPROVE_ID = "review-approve"
DENY_ID = "review-deny"
COMMENT_ID = "review-comment"
LIST_ID = "review-list"
DETAIL_ID = "review-detail"
_ITEM_PREFIX = "review-item-"


class ReviewScreen(Screen[None]):
    """Pending review items, one selected, approved or denied with a comment."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, command: QueueCommand, note: Callable[[str], None]) -> None:
        super().__init__()
        self._command = command
        self._note = note
        self._words = get_settings().review.words
        self._service: ReviewService | None = None
        self._items: dict[str, ReviewItem] = {}
        self._selected: ReviewItem | None = None
        # `ListView.clear()` removes asynchronously, so a rebuilt row must never reuse
        # the id of the one it replaces (the same rule as `ActionBar`).
        self._generation = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(id=LIST_ID)
            with Vertical():
                yield Static(self._words.nothing_selected, id=DETAIL_ID)
                yield Input(placeholder=self._words.comment_placeholder, id=COMMENT_ID)
                with Horizontal():
                    yield Button(self._words.approve_button, id=APPROVE_ID, variant="success")
                    yield Button(self._words.deny_button, id=DENY_ID, variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.title = self._words.tab_title
        self.refresh_items()

    def refresh_items(self) -> None:
        """Re-read the queue: what is pending now, in this console's scope."""
        self._service = self._command.service()
        listing = self.query_one(f"#{LIST_ID}", ListView)
        listing.clear()
        self._generation += 1
        self._items = {}
        visible = self._service.visible(self._command.subject())
        self.sub_title = self._words.title.format(n=len(visible))
        for item in visible:
            key = f"{_ITEM_PREFIX}{self._generation}-{item.id}"
            self._items[key] = item
            line = f"{item.kind} · " + " · ".join(self._service.cells(item))
            listing.append(ListItem(Label(line), id=key))
        self._select(None)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        key = event.item.id if event.item is not None else None
        self._select(self._items.get(key or ""))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == COMMENT_ID:
            self._enable()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id not in (APPROVE_ID, DENY_ID) or self._selected is None:
            return
        event.stop()
        comment = self.query_one(f"#{COMMENT_ID}", Input)
        await self.decide("approve" if event.button.id == APPROVE_ID else "deny", comment.value)

    async def decide(self, decision: str, comment: str) -> None:
        """Apply a decision to the selected item, then show what it did and re-read the queue."""
        if self._selected is None or self._service is None:
            return
        reviewer = get_settings().review.console_reviewer
        try:
            outcome = await self._service.decide(
                int(self._selected.id or 0),
                "approve" if decision == "approve" else "deny",
                reviewer,
                comment,
                self._command.subject(),
            )
        except ReviewError as error:
            self.query_one(f"#{DETAIL_ID}", Static).update(str(error))
            return
        title = self._words.decided.format(
            id=outcome.item_id, kind=outcome.kind, status=outcome.status, reviewer=reviewer,
            comment=self._service.redact(" ".join(comment.split())),
        )  # fmt: skip
        for line in (title, *outcome.lines):
            self._note(line)
        self.query_one(f"#{COMMENT_ID}", Input).value = ""
        self.refresh_items()

    def _select(self, item: ReviewItem | None) -> None:
        self._selected = item
        detail = self.query_one(f"#{DETAIL_ID}", Static)
        if item is None or self._service is None:
            detail.update(self._words.nothing_selected)
        else:
            heading = self._words.show_title.format(id=item.id, kind=item.kind, status=item.status)
            fields = zip(self._service.columns(item.kind), self._service.cells(item), strict=True)
            lines = [f"{name}: {value}" for name, value in fields]
            detail.update("\n".join([heading, *lines]))
        self._enable()

    def _enable(self) -> None:
        """Approve and Deny only with an item selected and a comment long enough to be one."""
        comment = self.query_one(f"#{COMMENT_ID}", Input).value.strip()
        ready = (
            self._selected is not None and len(comment) >= get_settings().review.min_comment_chars
        )
        for button in (APPROVE_ID, DENY_ID):
            self.query_one(f"#{button}", Button).disabled = not ready
