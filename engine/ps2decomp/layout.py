"""The guest memory layout the engine works in, configured per game.

Everything here used to be a constant scattered through the engine, set for one
game: the global pointer the startup code establishes, where the interpreter puts
its stack, where tests put their scratch memory, which addresses hold the
program. They differ between games - the global pointer always does - so they
live in one place, read at the moment they are needed, and are set from a game's
configuration (see game.py) before any analysis runs.

The defaults are the engine's conventions for a PS2 executable loaded at
0x00100000 in 32 MB of RAM: test scratch and the interpreter's stack sit high in
RAM, above where an executable and its BSS usually end. A game whose image
reaches that high moves them in its configuration.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Layout:
    # $gp as the startup code leaves it. Detected from the entry code per game.
    gp: int = 0
    # The program: where code and data are, for telling a pointer from a number.
    image_start: int = 0x00100000
    image_end: int = 0x02000000
    # Scratch memory the differential tests fill with arguments' targets.
    scratch: int = 0x01800000
    scratch_size: int = 0x20000
    # The interpreter's stack, and how far below the top a frame may reach.
    stack_top: int = 0x01F00000
    stack_size: int = 64 * 1024
    # The return address planted in $ra, so a returning function is noticed.
    return_sentinel: int = 0xDEAD0000

    @property
    def stack_low(self) -> int:
        return self.stack_top - self.stack_size


CURRENT = Layout()


def configure(**values) -> Layout:
    """Set layout fields for everything that runs afterwards; returns the layout."""
    for name, value in values.items():
        if not hasattr(CURRENT, name):
            raise KeyError("no layout field %r" % name)
        setattr(CURRENT, name, value)
    return CURRENT


# How far below the stack top a reference is still considered part of the frame. A
# function's own frame is far smaller than this; the bound exists so a wild address
# is treated as real memory rather than silently absorbed into the frame model.
MAX_FRAME_SIZE = 0x4000


def callee_stack_top() -> int:
    """Where callees of a derived function run, clear of the function's own frame."""
    return CURRENT.stack_top - MAX_FRAME_SIZE
