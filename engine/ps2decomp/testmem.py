"""Memory contents for differential tests.

The tests run original functions against a scratch region of fake memory, and what
that region holds decides whether a function can be tested at all.

Uniform random bytes were the original choice and turned out to be a poor one. Any
pointer a function loads from them is a wild 32-bit value, so pointer-chasing code
faults on the first dereference, and any loop count it loads is in the billions, so
counted loops exhaust the instruction budget. Measured on the functions whose only
obstacle is a loop, 117 of 330 faulted and 32 ran out of budget - not because the
interpreter lacked anything (it hit no unsupported instruction at all) but because
the inputs made no sense.

No single alternative fixes everything, because a word one function reads as a
pointer another reads as a count. So there are several shapes, and a harness tries
them in turn, keeping the first under which the original executes:

  noise     uniform random words (the old behaviour, still occasionally the only fit)
  mixed     pointers into the region in even words, small integers in odd ones
  small     small integers only, for counts, indices and flags
  pointers  pointers into the region only, for linked structures

Trying all four took the loop-only set from 181 executable functions to 247.

Three more come last, for loops the others never end:

  zero       all zeros: a search for a key stops at once (the key read is 0 as
             well), a count read from memory is 0, a null link ends a list
  zero-self  each even word its own address, odd words 0
  self       each word its own address: an intrusive circular list whose
             links point back at its head (RenderWare's) is empty

Sampled on functions untestable for running out of instructions, these let
one in eight finish.

The choice of profile changes only what the test can *reach*, never what it
concludes: the original and the derivation always read the same memory.
"""

from __future__ import annotations

import random
import struct
import zlib

from . import layout

# The scratch region's place and size come from the game's layout (layout.py).

# Order matters: the first profile under which a function executes is the one it is
# tested with, so the most generally useful shapes come first.
# "forward" and "forward-mixed" come last: they only matter to functions no other
# shape lets finish - typically list walks, which loop forever on the cycles that
# random pointers make. Their pointers only point further on, or are null, so every
# chain followed through them ends.
PROFILES = ("mixed", "noise", "small", "pointers", "forward", "forward-mixed", "zero", "zero-self", "self")

# Small integers stay below this so a count read from memory gives a short loop.
SMALL_LIMIT = 16


# Each test case needs memory that *differs* from the previous case's, not memory
# that is unique across the whole run. Building 32K random words in Python for every
# case made the derivation run 25x slower, so buffers come from a small cached pool
# per profile instead; consecutive seeds still select different buffers.
POOL_SIZE = 16
_POOL = {}


def build(kind: str, seed: int) -> bytes:
    """One profile's contents for the scratch region, deterministic in `seed`."""
    key = (kind, seed % POOL_SIZE, layout.CURRENT.scratch, layout.CURRENT.scratch_size)
    buffer = _POOL.get(key)
    if buffer is None:
        buffer = _build(kind, key[1])
        _POOL[key] = buffer
    return buffer


def _build(kind: str, seed: int) -> bytes:
    """Fill the scratch region with one profile's contents, deterministically."""
    # zlib.crc32 rather than hash(): Python randomises string hashes per process,
    # which would give different "deterministic" memory on every run and make a
    # failing test impossible to reproduce.
    rng = random.Random((zlib.crc32(kind.encode("ascii")) << 16) ^ seed)
    SCRATCH, SCRATCH_SIZE = layout.CURRENT.scratch, layout.CURRENT.scratch_size
    count = SCRATCH_SIZE // 4

    def pointer() -> int:
        # 16-aligned and clear of the end, so a structure read through it stays in
        # range for a reasonable span of fields.
        return SCRATCH + rng.randrange(0, SCRATCH_SIZE - 0x100, 16)

    if kind == "noise":
        words = [rng.getrandbits(32) for _ in range(count)]
    elif kind == "small":
        words = [rng.randrange(SMALL_LIMIT) for _ in range(count)]
    elif kind == "pointers":
        words = [pointer() for _ in range(count)]
    elif kind == "mixed":
        words = [pointer() if i % 2 == 0 else rng.randrange(SMALL_LIMIT)
                 for i in range(count)]
    elif kind in ("forward", "forward-mixed"):
        def forward(i: int) -> int:
            address = SCRATCH + 4 * i
            if rng.random() < 0.25 or address + 0x200 >= SCRATCH + SCRATCH_SIZE - 0x100:
                return 0
            return (address + rng.randrange(16, 0x200)) & ~15
        words = [rng.randrange(SMALL_LIMIT) if kind == "forward-mixed" and i % 2
                 else forward(i) for i in range(count)]
    elif kind == "zero":
        words = [0] * count
    elif kind == "zero-self":
        words = [SCRATCH + 4 * i if i % 2 == 0 else 0 for i in range(count)]
    elif kind == "self":
        words = [SCRATCH + 4 * i for i in range(count)]
    else:
        raise ValueError("unknown memory profile %r" % kind)
    return struct.pack("<%dI" % count, *words)


def arguments(rng: random.Random) -> tuple:
    """A full register argument vector: eight integers, then eight floats.

    The integers are 16-aligned pointers into the first half of scratch. The first
    four are drawn exactly as they were when only four arguments existed, so the
    cases every earlier function was verified on are unchanged. The floats are bit
    patterns of ordinary values - a mix of small exact ones and random magnitudes -
    rather than random bits, which would be mostly NaNs and denormals.
    """
    scratch, size = layout.CURRENT.scratch, layout.CURRENT.scratch_size
    integers = [scratch + rng.randrange(0, size // 2, 16) for _ in range(8)]
    return tuple(integers) + tuple(float_bits(rng) for _ in range(8))


FLOAT_CASES = (0.0, 1.0, -1.0, 0.5, 2.0, -0.25, 100.0, 3.5)


def float_bits(rng: random.Random) -> int:
    """The bit pattern of a plausible float argument."""
    if rng.random() < 0.5:
        value = rng.choice(FLOAT_CASES)
    else:
        value = rng.uniform(-1000.0, 1000.0)
    return struct.unpack("<I", struct.pack("<f", value))[0]


def seed_global_pointers(memory, addresses) -> int:
    """Point zero-valued global pointer variables at the scratch region.

    Game state - a manager object, the current race, a list head - is reached
    through global pointers that are null in the executable and only set up once
    the game is running. Any function that dereferences one faulted in every test,
    whatever the scratch memory held: 712 otherwise derivable functions could not be
    tested for that reason alone. The recovered types say which globals are
    dereferenced as pointers; each one still holding zero is given a deterministic
    address in scratch. Like the memory profiles, this decides only what a test can
    reach - the original and the derivation read the same memory - and it is
    written once, outside any test's journal.
    """
    seeded = 0
    for address in sorted(addresses):
        try:
            if memory.read_int(address, 4, False) != 0:
                continue
        except Exception:
            continue
        size = layout.CURRENT.scratch_size
        # Every word in one 16-byte block gets the same target, so a [begin, end)
        # pair of adjacent globals describes an empty range rather than one
        # running from one random place to another - a walk over it, as in the
        # SIF RPC binder, otherwise never meets its end.
        block = address & ~15
        offset = (zlib.crc32(block.to_bytes(4, "little")) % (size // 2)) & ~15
        memory.write_int(address, layout.CURRENT.scratch + offset, 4)
        seeded += 1
    return seeded


# A dereference this close to address zero is a null pointer plus a field offset.
NULL_PAGE = 0x1000


def discover_null_pointers(machine, functions, is_global, budget: int = 20_000,
                           attempts: int = 8) -> int:
    """Seed the global pointers that running the code shows to be dereferenced.

    `seed_global_pointers` covers the globals the recovered types mark as pointers,
    but not pointer fields of global *objects* - a singleton whose address is
    passed to its methods, which then follow a pointer inside it that is null until
    the game runs. Recovering those statically drowns in merged types, so they are
    found by running instead: each function is run once, and when it faults just
    above address zero, the last word it read as zero is taken to be the null
    pointer, seeded like any other global, and the function run again.

    `is_global(address)` limits seeding to the program's own data - never code,
    the stack or the scratch region. The functions are visited in the order given
    and every seed is written outside any journal, so the result is deterministic
    and the same for every test that follows. Returns the addresses seeded.
    """
    from .emu import MemoryFault

    memory = machine.memory
    scratch = layout.CURRENT.scratch
    seeded = []
    memory.watch_zero = True
    try:
        for func in functions:
            args = arguments(random.Random(func["addr"]))
            for _attempt in range(attempts):
                memory.end_journal()
                memory.write(scratch, build("pointers", func["addr"]))
                memory.begin_journal()
                memory.last_zero_read = None
                fault = None
                try:
                    machine.call(func["addr"], args, max_instructions=budget)
                except MemoryFault as exc:
                    fault = exc
                except Exception:
                    pass
                source = memory.last_zero_read
                stack_pointer = machine.get(29) & 0xFFFFFFFF
                memory.rollback()
                memory.end_journal()
                if (fault is None or fault.addr >= NULL_PAGE or source is None
                        or source >= stack_pointer or not is_global(source)):
                    break
                if seed_global_pointers(memory, {source}):
                    seeded.append(source)
    finally:
        memory.watch_zero = False
        memory.last_zero_read = None
    return seeded


def page_contents(page: int) -> bytes:
    """What a page mapped on demand holds: the "mixed" shape, seeded by its address.

    Pointers into scratch in even words and small integers in odd ones, as the
    mixed profile has, so code that faulted following a wild pointer finds the
    kind of memory the tests are built on - and its own page differs from every
    other page, deterministically.
    """
    rng = random.Random(zlib.crc32(page.to_bytes(4, "little")) ^ 0x9A6E)
    scratch, size = layout.CURRENT.scratch, layout.CURRENT.scratch_size
    words = [scratch + rng.randrange(0, size - 0x100, 16) if i % 2 == 0
             else rng.randrange(SMALL_LIMIT) for i in range(0x1000 // 4)]
    return struct.pack("<%dI" % len(words), *words)


# How much of the stack above the test stack top - the caller's frame - is filled.
CALLER_FRAME = 0x400


def fill_caller_frame(memory) -> None:
    """Fill the stack above the test stack top with the return sentinel.

    Function discovery also finds fragments: shared epilogue tails entered with a
    frame they never built. They restore $ra (and saved registers) from their
    caller's frame and return through it - to address 0, when that memory is zero,
    so every test of them failed. With the sentinel there, a fragment's return
    ends the test as a function's does, and its own work - the restores, the
    result - is compared like any other. Written once, outside any journal.
    """
    from .emu import RETURN_SENTINEL

    words = CALLER_FRAME // 4
    memory.write(layout.CURRENT.stack_top,
                 struct.pack("<%dI" % words, *([RETURN_SENTINEL] * words)))
