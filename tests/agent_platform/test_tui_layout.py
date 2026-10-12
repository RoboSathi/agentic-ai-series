"""Where the live status sits, and what it says.

The spinner, stage and elapsed time belong above the chat bar (as in
Claude), and say only about this turn: a session-wide memory statistic
("recall without embeddings 100%") told a reader nothing they could act on.
"""

from agent_platform.interface.tui.app import AgenticTuiApp
from agent_platform.interface.tui.widgets import StatusLineWidget


class TestTheStatusLineIsAboveTheInput:
    def test_it_is_composed_before_the_chat_bar(self) -> None:
        order = [type(widget).__name__ for widget in AgenticTuiApp().compose()]
        assert order.index("StatusLineWidget") < order.index("ChatPrompt")
        assert order.index("LiveStatusPanel") < order.index("StatusLineWidget")


class TestTheStatusLineSaysOnlyAboutThisTurn:
    def test_it_carries_no_session_statistic(self) -> None:
        line = StatusLineWidget()
        line.begin("Assembling context…")
        assert not hasattr(line, "set_context")
        assert "recall" not in line.render().plain
