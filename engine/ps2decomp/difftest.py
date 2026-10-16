"""Differential testing of reconstructed functions against the original code.

This is the project's substitute for byte-matching.  Byte-matching needs the
original Metrowerks compiler, which is unavailable, so a reconstruction cannot be
proven *identical*.  It can, however, be proven to *behave* identically: run the
original machine code in the interpreter, run the reconstruction, and compare.

That distinction matters and is worth being precise about.  A passing differential
test shows the two agree on the inputs tested.  It is evidence, not proof - a
function with input-dependent behaviour the test never exercises can still differ.
So the harness records how many cases ran and insists on edge cases alongside
random ones, and `Report.summary()` always states the case count rather than just
"passed".
"""

from __future__ import annotations

import dataclasses
import random

from .emu import EmulationError, Machine

NUL = b"\x00"


@dataclasses.dataclass
class Mismatch:
    """One input where the reconstruction and the original disagreed."""

    case: object
    expected: object
    actual: object

    def __str__(self) -> str:
        return "case %r: original gave %r, reconstruction gave %r" % (
            self.case, self.expected, self.actual,
        )


@dataclasses.dataclass
class Report:
    name: str
    address: int
    cases_run: int = 0
    mismatches: list = dataclasses.field(default_factory=list)
    errors: list = dataclasses.field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches and not self.errors and self.cases_run > 0

    def summary(self) -> str:
        state = "VERIFIED" if self.passed else "FAILED"
        detail = "%s  %-34s 0x%08X  %d cases" % (
            state, self.name, self.address, self.cases_run
        )
        if self.mismatches:
            detail += "  %d mismatch(es)" % len(self.mismatches)
        if self.errors:
            detail += "  %d error(s)" % len(self.errors)
        return detail

    def detail(self, limit: int = 5) -> str:
        lines = [self.summary()]
        lines += ["  " + str(m) for m in self.mismatches[:limit]]
        lines += ["  error: " + e for e in self.errors[:limit]]
        return "\n".join(lines)


class StringFunctionTester:
    """Differential-tests a function whose only argument is a C string.

    The string is written into scratch memory and its address passed in $a0.  Each
    case gets a fresh slot so one case cannot contaminate the next through
    leftover bytes - a real hazard, since a missing terminator would otherwise
    read into the previous case's data.
    """

    SLOT_SIZE = 1024

    def __init__(self, machine: Machine, scratch_base: int = 0x01200000) -> None:
        self.machine = machine
        self.scratch_base = scratch_base

    def run(self, name: str, address: int, reference, cases) -> Report:
        report = Report(name=name, address=address)
        for index, case in enumerate(cases):
            payload = case if isinstance(case, bytes) else case.encode("ascii")
            slot = self.scratch_base + index * self.SLOT_SIZE
            if len(payload) + 1 > self.SLOT_SIZE:
                report.errors.append("case %d does not fit a scratch slot" % index)
                continue
            # Clear the slot first so a bug cannot read stale bytes and appear to
            # pass.
            self.machine.memory.write(slot, NUL * self.SLOT_SIZE)
            self.machine.memory.write(slot, payload + NUL)
            try:
                actual = self.machine.call(address, (slot,)) & 0xFFFFFFFF
            except EmulationError as exc:
                report.errors.append("case %r: %s" % (payload[:32], exc))
                continue
            expected = reference(payload) & 0xFFFFFFFF
            report.cases_run += 1
            if actual != expected:
                report.mismatches.append(Mismatch(payload, expected, actual))
        return report


class IntFunctionTester:
    """Differential-tests a function whose arguments and result are integers.

    This is the simplest and strongest case: with no memory involved, the corpus
    can cover the input space densely rather than sampling it, so a passing result
    is close to exhaustive for single-argument bit functions.
    """

    def __init__(self, machine: Machine) -> None:
        self.machine = machine

    def run(self, name: str, address: int, reference, cases) -> Report:
        report = Report(name=name, address=address)
        for case in cases:
            args = case if isinstance(case, tuple) else (case,)
            try:
                actual = self.machine.call(address, args)
            except EmulationError as exc:
                report.errors.append("args %r: %s" % (args, exc))
                continue
            expected = reference(*args)
            report.cases_run += 1
            if actual != expected:
                report.mismatches.append(Mismatch(args, expected, actual))
        return report


def bit_corpus() -> list:
    """Inputs chosen to exercise every bit position and the usual edge values."""
    cases = [0, 1, 2, 3, 0xFFFFFFFF, 0x80000000, 0x55555555, 0xAAAAAAAA]
    cases += [1 << i for i in range(32)]
    # Two bits set, so a routine that stops at the first cannot pass by accident.
    cases += [(1 << i) | (1 << ((i + 7) % 32)) for i in range(32)]
    rng = random.Random(777)
    cases += [rng.getrandbits(32) for _ in range(120)]
    return cases


class StringTransformTester:
    """Differential-tests a function that writes into a destination buffer.

    `strcpy`/`strcat`-shaped routines report their result in memory rather than in
    $v0, so comparing return values is not enough: the buffer contents are the
    output.  Both buffers are pre-filled with a sentinel and the byte past the
    expected terminator is checked afterwards, which catches a routine that writes
    too *much* as well as one that writes too little - an over-copy would
    otherwise pass unnoticed.

    Alignment is varied deliberately.  These routines branch on the low bits of
    their pointers and take a different path when the operands are co-aligned, so
    testing a single alignment would leave most of the code unexercised.
    """

    FILL = 0xAA
    SLOT_SIZE = 512

    def __init__(self, machine: Machine, scratch_base: int = 0x01500000) -> None:
        self.machine = machine
        self.scratch_base = scratch_base

    def run(self, name: str, address: int, reference, cases, prefixes=(b"",),
            alignments=(0, 1, 4, 8, 15)) -> Report:
        """`reference(prefix, src)` returns the expected destination contents."""
        report = Report(name=name, address=address)
        slot = 0
        fill = bytes([self.FILL]) * self.SLOT_SIZE
        for prefix in prefixes:
            for case in cases:
                src = case if isinstance(case, bytes) else case.encode("ascii")
                for align in alignments:
                    dest = self.scratch_base + slot * self.SLOT_SIZE + align
                    source = self.scratch_base + (slot + 1) * self.SLOT_SIZE + align
                    slot += 2
                    if len(prefix) + len(src) + 2 > self.SLOT_SIZE:
                        continue

                    self.machine.memory.write(dest - align, fill)
                    self.machine.memory.write(source - align, fill)
                    self.machine.memory.write(dest, prefix + NUL)
                    self.machine.memory.write(source, src + NUL)

                    try:
                        self.machine.call(address, (dest, source))
                    except EmulationError as exc:
                        report.errors.append(
                            "prefix=%r src=%r align=%d: %s"
                            % (prefix[:12], src[:16], align, exc)
                        )
                        continue

                    expected = reference(prefix, src) + NUL
                    actual = bytes(self.machine.memory.read(dest, len(expected)))
                    report.cases_run += 1
                    if actual != expected:
                        report.mismatches.append(
                            Mismatch((prefix, src, align), expected, actual)
                        )
                        continue
                    tail = self.machine.memory.read_int(dest + len(expected), 1, False)
                    if tail != self.FILL:
                        report.mismatches.append(
                            Mismatch(
                                (prefix, src, align),
                                "nothing written past the terminator",
                                "0x%02X written past the terminator" % tail,
                            )
                        )
        return report


def ascii_corpus(count: int = 40, seed: int = 20240501, max_len: int = 48) -> list:
    """Edge cases first, then deterministic random strings.

    The edge cases are the ones that actually catch bugs - empty input, single
    bytes, and lengths either side of the 8- and 16-byte boundaries that the
    SIMD string routines special-case.  The random tail broadens coverage without
    replacing them, and a fixed seed keeps failures reproducible.
    """
    cases = [b"", b"a", b"ab", b"abc"]
    for length in (7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65):
        cases.append(b"x" * length)
    cases += [
        b"Damping",
        b"Spring Coeff",
        b"Score/Boost/Air",
        b"../Export/ValueDB/Score.cfg",
        b"Slip Speed Multipler",
        b"Material Properties/Reflectivity",
        b"~!@#$%^&*()_+-=[]{}|;:,.<>?",
        b"MiXeD CaSe NaMe",
    ]
    rng = random.Random(seed)
    while len(cases) < count:
        length = rng.randint(1, max_len)
        cases.append(bytes(rng.randint(0x20, 0x7E) for _ in range(length)))
    return cases[:count] if count >= len(cases) else cases
