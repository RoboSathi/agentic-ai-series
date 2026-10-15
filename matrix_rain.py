"""Matrix-style digital rain. Ctrl+C ends the rain and spells AGENT in digits."""

from __future__ import annotations

import curses
import random
from dataclasses import dataclass

FRAME_DELAY_MS = 200
MIN_TRAIL = 8
MAX_TRAIL = 30
MIN_SPEED = 1
MAX_SPEED = 3
COLUMN_STEP = 1
DROPS_PER_COLUMN = 2
BACKGROUND_DENSITY = 0.3
DIGITS = "0123456789"
WORD = "AGENT"
LETTER_GAP = 1
PIXEL_WIDTH = 2
ON_PIXEL = "#"

COLOR_HEAD = 1
COLOR_BRIGHT = 2
COLOR_MID = 3
COLOR_DIM = 4
PALETTE = {
    COLOR_HEAD: 231,
    COLOR_BRIGHT: 46,
    COLOR_MID: 34,
    COLOR_DIM: 22,
}
FALLBACK_COLORS = {
    COLOR_HEAD: curses.COLOR_WHITE,
    COLOR_BRIGHT: curses.COLOR_GREEN,
    COLOR_MID: curses.COLOR_GREEN,
    COLOR_DIM: curses.COLOR_GREEN,
}
EXTENDED_COLORS = 256

FONT: dict[str, tuple[str, ...]] = {
    "A": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "G": (" ####", "#    ", "#    ", "#  ##", "#   #", "#   #", " ### "),
    "E": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"),
    "N": ("#   #", "##  #", "# # #", "# # #", "#  ##", "#   #", "#   #"),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
}


@dataclass
class Drop:
    """One falling column: head row, rows per frame, and trail length."""

    x: int
    head: int
    speed: int
    length: int


def new_drop(x: int, rows: int, *, scattered: bool) -> Drop:
    length = random.randint(MIN_TRAIL, MAX_TRAIL)
    start = random.randint(-rows, rows) if scattered else -random.randint(0, rows)
    return Drop(x, start, random.randint(MIN_SPEED, MAX_SPEED), length)


def build_columns(rows: int, cols: int) -> list[Drop]:
    return [
        new_drop(x, rows, scattered=True)
        for x in range(0, cols, COLUMN_STEP)
        for _ in range(DROPS_PER_COLUMN)
    ]


def step(drops: list[Drop], rows: int) -> None:
    for i, drop in enumerate(drops):
        drop.head += drop.speed
        if drop.head - drop.length > rows:
            drops[i] = new_drop(drop.x, rows, scattered=False)


def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    extended = curses.COLORS >= EXTENDED_COLORS
    source = PALETTE if extended else FALLBACK_COLORS
    for pair, color in source.items():
        curses.init_pair(pair, color, -1)


def trail_attr(offset: int, length: int) -> int:
    if offset == 0:
        return curses.color_pair(COLOR_HEAD) | curses.A_BOLD
    fade = offset / length
    if fade < 0.3:
        return curses.color_pair(COLOR_BRIGHT) | curses.A_BOLD
    if fade < 0.7:
        return curses.color_pair(COLOR_MID)
    return curses.color_pair(COLOR_DIM)


def put(win: curses.window, y: int, x: int, text: str, attr: int) -> None:
    rows, cols = win.getmaxyx()
    if 0 <= y < rows and 0 <= x < cols - 1:
        win.addstr(y, x, text, attr)


def draw_frame(win: curses.window, drops: list[Drop]) -> None:
    win.erase()
    for drop in drops:
        for offset in range(drop.length):
            put(win, drop.head - offset, drop.x, random.choice(DIGITS), trail_attr(offset, drop.length))
    win.refresh()


def word_bitmap(word: str) -> list[str]:
    glyphs = [FONT[letter] for letter in word]
    gap = " " * LETTER_GAP
    return [gap.join(glyph[row] for glyph in glyphs) for row in range(len(glyphs[0]))]


def draw_background(win: curses.window) -> None:
    rows, cols = win.getmaxyx()
    attr = curses.color_pair(COLOR_DIM)
    for y in range(rows):
        for x in range(cols - 1):
            if random.random() < BACKGROUND_DENSITY:
                put(win, y, x, random.choice(DIGITS), attr)


def draw_word(win: curses.window, word: str) -> None:
    win.erase()
    draw_background(win)
    bitmap = word_bitmap(word)
    rows, cols = win.getmaxyx()
    width = len(bitmap[0]) * PIXEL_WIDTH
    top, left = (rows - len(bitmap)) // 2, (cols - width) // 2
    attr = curses.color_pair(COLOR_BRIGHT) | curses.A_BOLD
    for r, line in enumerate(bitmap):
        for c, cell in enumerate(line):
            if cell == ON_PIXEL:
                digits = "".join(random.choice(DIGITS) for _ in range(PIXEL_WIDTH))
                put(win, top + r, left + c * PIXEL_WIDTH, digits, attr)
    win.refresh()


def rain(win: curses.window) -> None:
    rows, cols = win.getmaxyx()
    drops = build_columns(rows, cols)
    while True:
        if win.getch() == curses.KEY_RESIZE:
            rows, cols = win.getmaxyx()
            drops = build_columns(rows, cols)
        step(drops, rows)
        draw_frame(win, drops)
        curses.napms(FRAME_DELAY_MS)


def main(win: curses.window) -> None:
    curses.curs_set(0)
    win.nodelay(True)
    init_colors()
    try:
        rain(win)
    except KeyboardInterrupt:
        pass
    win.nodelay(False)
    draw_word(win, WORD)
    try:
        win.getch()
    except KeyboardInterrupt:
        pass
    win.erase()
    win.refresh()


if __name__ == "__main__":
    curses.wrapper(main)
