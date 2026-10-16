"""An R5900 interpreter, used to verify reconstructed functions behave correctly.

The project cannot prove byte-equality, because the original Metrowerks compiler
is unavailable.  Functional reconstruction needs a different kind of proof, and
this supplies it: run the *original* machine code in an interpreter, run the
reconstruction on the same inputs, and compare results.  That turns "I think this
C++ is right" into something a test can decide.

Scope is deliberately the integer core plus the common loads and stores - enough
to execute leaf and near-leaf functions, which is where reconstruction starts.
Anything outside that subset raises `UnsupportedInstruction` rather than being
quietly approximated, because an interpreter that silently guesses would produce
exactly the false confidence this is meant to remove.

Registers are modelled as full 64-bit values with MIPS III semantics: 32-bit
operations sign-extend their result into the destination register.
"""

from __future__ import annotations

import struct
import sys

from . import layout, vu0
from .r5900 import DISASM, GPR

MASK64 = (1 << 64) - 1
MASK32 = 0xFFFFFFFF

# Where the interpreter puts the stack. Chosen above the loaded image and its BSS
# so it cannot collide with real data.
# The return address planted in $ra: reaching it means the called function returned.
RETURN_SENTINEL = 0xDEAD0000

# Where the stack is, what $gp holds and the rest of the guest memory layout are
# per game; see layout.py. $gp matters for execution in particular: a great many
# functions reach globals as an offset from it, and leaving the register zero
# makes every one of them fault.


class EmulationError(Exception):
    """Base class for interpreter failures."""


class UnsupportedInstruction(EmulationError):
    def __init__(self, addr: int, word: int, mnemonic: str) -> None:
        super().__init__(
            "unsupported instruction at 0x%08X: %08x (%s)" % (addr, word, mnemonic)
        )
        self.addr = addr
        self.word = word
        self.mnemonic = mnemonic


# Decoded instructions by address, shared by every Machine: decoding is a pure
# function of the word and its address.
_DECODED = {}


class MemoryFault(EmulationError):
    def __init__(self, addr: int, size: int, store: bool) -> None:
        super().__init__(
            "%s of %d byte(s) at unmapped 0x%08X"
            % ("store" if store else "load", size, addr)
        )
        self.addr = addr


class ExecutionLimit(EmulationError):
    """Raised when a function runs longer than the instruction budget allows."""


def _sign32(value: int) -> int:
    value &= MASK32
    return value - (1 << 32) if value & 0x80000000 else value


def _sign64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


PAGE_SIZE = 0x1000
PAGE_MASK = PAGE_SIZE - 1
# At most this many pages are mapped on demand in one process: 64 MB. Pages a
# test keeps accumulate over a whole run - the derivation tests some 10,000
# functions in one process - and at 4,096 the later functions faulted for want
# of a page; Memory.map_page warns when the cap is reached.
MAX_PAGES = 16384

# Address windows mapped as plain memory besides main RAM. The scratchpad is real
# RAM on the console. The hardware registers (DMA, VIF, GIF, timers...) are not,
# but as plain memory they read zero - idle, done - until written, so code that
# configures a DMA transfer or polls for its end runs to completion instead of
# faulting. Every form of a function under test sees the same window, so this
# decides what can be tested, never whether two forms agree.
WINDOWS = (
    (0x10000000, 0x10010000),     # EE hardware registers
    (0x70000000, 0x70004000),     # scratchpad RAM, 16 KB
)


class Memory:
    """Main RAM from the image's base, plus the fixed windows; all else faults."""

    def __init__(self, base: int, data: bytes, total_size: int) -> None:
        if total_size < len(data):
            total_size = len(data)
        self.base = base
        self.size = total_size
        # The windows live after main RAM in the same buffer, so one journal of
        # (offset, previous bytes) covers every write.
        self.windows = []
        at = total_size
        for start, end in WINDOWS:
            self.windows.append((start, end, at))
            at += end - start
        self.bytes = bytearray(at)
        self.bytes[: len(data)] = data
        self._journal = None
        # Pages mapped on demand (see map_page), after the windows in the buffer.
        self.pages = {}
        self.page_order = []         # the same pages, in the order they sit in the buffer
        self.page_source = None      # page address -> its initial contents
        self.pages_exhausted = False
        # When watching, the address of the last word read as zero: what a null
        # pointer dereference most likely loaded its base from (testmem uses it).
        self.watch_zero = False
        self.last_zero_read = None

    def _offset(self, addr: int, size: int, store: bool) -> int:
        off = addr - self.base
        if 0 <= off and off + size <= self.size:
            return off
        for start, end, at in self.windows:
            if start <= addr and addr + size <= end:
                return at + addr - start
        at = self.pages.get(addr & ~PAGE_MASK)
        if at is not None and (addr & PAGE_MASK) + size <= PAGE_SIZE:
            return at + (addr & PAGE_MASK)
        raise MemoryFault(addr, size, store)

    def map_page(self, addr: int) -> bool:
        """Map the page holding `addr`, filled by `page_source`; False if it cannot be.

        For tests only: test data reaches addresses no real program state does, and
        a fault there says nothing about the function. Mapping the page and running
        again lets the test go on, and since every form of the function reads the
        same memory afterwards, the comparison is as sound as before.
        """
        page = addr & ~PAGE_MASK & MASK32
        if self.page_source is None or page in self.pages:
            return False
        if len(self.pages) >= MAX_PAGES:
            if not self.pages_exhausted:
                self.pages_exhausted = True
                print("warning: %d pages mapped for tests, the most allowed; later "
                      "tests may fault for want of a page" % MAX_PAGES, file=sys.stderr)
            return False
        self.pages[page] = len(self.bytes)
        self.page_order.append(page)
        self.bytes.extend(self.page_source(page))
        return True

    def page_mark(self) -> tuple:
        """Where on-demand mapping stands, for unmap_to."""
        return len(self.pages), len(self.bytes)

    def unmap_to(self, mark: tuple) -> None:
        """Unmap the pages mapped since `mark`. Only outside a journal: journal
        entries may point into the pages dropped."""
        assert self._journal is None, "unmapping inside a journal"
        count, size = mark
        for page in self.page_order[count:]:
            del self.pages[page]
        del self.page_order[count:]
        del self.bytes[size:]

    def undo_to(self, mark: int) -> None:
        """Undo the journalled writes after the first `mark` entries."""
        journal = self._journal
        while len(journal) > mark:
            offset, previous = journal.pop()
            self.bytes[offset : offset + len(previous)] = previous
            self._journal_bytes -= len(previous)

    def offset_of(self, addr: int) -> int:
        """Where an address lives in the backing buffer, as the journal records it."""
        return self._offset(addr, 1, False)

    def address_at(self, offset: int) -> int:
        """The address a position in the backing buffer holds; offset_of's inverse.

        A page mapped on demand lands wherever the buffer ends at the time, so
        the same address sits at different positions in two processes that
        mapped pages in a different order. Anything compared across processes -
        recorded effects - must name addresses, not positions.
        """
        if offset < self.size:
            return self.base + offset
        for start, end, at in self.windows:
            if at <= offset < at + (end - start):
                return start + offset - at
        page_area = self.windows[-1][2] + (self.windows[-1][1] - self.windows[-1][0]) \
            if self.windows else self.size
        index, within = divmod(offset - page_area, PAGE_SIZE)
        return self.page_order[index] + within

    # -- journalling ---------------------------------------------------
    #
    # Differential testing has to undo whatever a function wrote before the next
    # case runs. Snapshotting a chosen region is not enough: a function that writes
    # to a global outside that region leaves the change in place, which both
    # corrupts later cases and makes the derivation appear to disagree when it was
    # the harness that moved. Journalling every write makes the undo exact and
    # cheap, whatever the function touches.

    # A test that writes more than this is abandoned rather than allowed to grow
    # the journal - and the per-byte effect maps built from it - until the process
    # runs out of memory.
    JOURNAL_LIMIT = 1 << 20

    def begin_journal(self) -> None:
        self._journal = []
        self._journal_bytes = 0

    def rollback(self) -> None:
        """Undo every write since `begin_journal`, most recent first."""
        journal = getattr(self, "_journal", None)
        if journal is None:
            return
        for offset, previous in reversed(journal):
            self.bytes[offset : offset + len(previous)] = previous
        self._journal = []
        self._journal_bytes = 0

    def end_journal(self) -> None:
        self._journal = None

    def read(self, addr: int, size: int) -> bytes:
        off = self._offset(addr, size, False)
        return bytes(self.bytes[off : off + size])

    def write(self, addr: int, payload: bytes) -> None:
        off = self._offset(addr, len(payload), True)
        if self._journal is not None:
            self._journal_bytes += len(payload)
            if self._journal_bytes > self.JOURNAL_LIMIT:
                raise EmulationError("a test wrote more than %d bytes" % self.JOURNAL_LIMIT)
            self._journal.append((off, bytes(self.bytes[off : off + len(payload)])))
        self.bytes[off : off + len(payload)] = payload

    def read_int(self, addr: int, size: int, signed: bool) -> int:
        raw = self.read(addr, size)
        value = int.from_bytes(raw, "little", signed=signed)
        if self.watch_zero and value == 0 and size == 4:
            self.last_zero_read = addr
        return value

    def write_int(self, addr: int, value: int, size: int) -> None:
        self.write(addr, (value & ((1 << (size * 8)) - 1)).to_bytes(size, "little"))

    def read_cstring(self, addr: int, limit: int = 4096) -> bytes:
        out = bytearray()
        for i in range(limit):
            byte = self.read_int(addr + i, 1, False)
            if byte == 0:
                break
            out.append(byte)
        return bytes(out)


class Machine:
    """An R5900 interpreter over a loaded image."""

    def __init__(self, base: int, data: bytes, total_size: int = None) -> None:
        if total_size is None:
            total_size = max(len(data), layout.CURRENT.stack_top + layout.CURRENT.stack_size - base)
        self.memory = Memory(base, data, total_size)
        # The executable's own bytes. A call to anything outside them is stubbed
        # (see _stubbed_call) rather than executed.
        self.code_start = base
        self.code_end = base + len(data)
        self.stubbed_calls = 0
        self.regs = [0] * 32
        # The EE's registers are 128 bits wide. Full SIMD arithmetic is out of
        # scope, but `lq`/`sq` must preserve the upper half, because Metrowerks
        # uses them to spill callee-saved registers in nearly every prologue -
        # losing those bits would corrupt restores.
        self.regs_hi = [0] * 32
        self.hi = 0
        self.lo = 0
        self.hi1 = 0       # HI/LO of the second multiply pipeline
        self.lo1 = 0
        # Coprocessor 1: 32 single-precision registers held as raw 32-bit patterns,
        # plus the condition flag that `c.<cond>.s` sets and `bc1t`/`bc1f` test.
        #
        # Storing bits rather than Python floats matters. `lwc1`/`swc1`/`mtc1`/`mfc1`
        # are bit moves - the hardware copies all 32 bits verbatim - so keeping
        # floats here round-trips through a double and canonicalises NaN payloads,
        # silently altering values a copy must preserve. That showed up as a
        # float-copying function disagreeing with a derivation that was correct.
        self.fpr = [0] * 32
        self.fp_cond = False
        self.fp_acc = 0      # the FPU accumulator, as a bit pattern
        # VU0 in macro mode (vu0.py): vector registers as bit patterns, the
        # control registers (VI0-VI15, then Q, I, R and the rest), the accumulator.
        self.vf = vu0.initial_vf()
        self.vc = vu0.initial_vc()
        self.vacc = [0, 0, 0, 0]
        self.sa = 0          # the shift amount qfsrv uses, in bits (mtsab/mtsah)
        self.pc = 0
        self.instructions_executed = 0
        self.call_log = []
        # Optional per-address execution counts. Off by default: the derivation
        # pipeline runs millions of instructions and does not need them, but loop
        # analysis does, because the number of times a loop header is reached *is*
        # the trip count.
        self.count_visits = False
        self.visit_counts = {}

    # -- register access ----------------------------------------------

    def get(self, index: int) -> int:
        return 0 if index == 0 else self.regs[index]

    def set(self, index: int, value: int) -> None:
        if index:
            self.regs[index] = value & MASK64

    def get32(self, index: int) -> int:
        return self.get(index) & MASK32

    def set32(self, index: int, value: int) -> None:
        """Write a 32-bit result, sign-extended into the 64-bit register."""
        self.set(index, _sign32(value) & MASK64)

    # -- loading a classmethod convenience -----------------------------

    @classmethod
    def from_elf(cls, elf) -> "Machine":
        section = elf.exec_sections[0]
        data = elf.section_data(section)
        # Cover the BSS the program headers declare, so globals are addressable.
        memsz = max(
            (p.memsz for p in elf.segments if p.vaddr == section.addr), default=len(data)
        )
        total = max(len(data), memsz,
                    layout.CURRENT.stack_top + layout.CURRENT.stack_size - section.addr)
        return cls(section.addr, data, total)

    # -- execution -----------------------------------------------------

    # When set, a journalled call that faults on an unmapped page maps the page
    # and runs again (Memory.map_page). Off by default: a fault is real
    # information everywhere except in a differential test.
    map_faults = False
    FAULT_RETRIES = 16

    def call(self, addr: int, args=(), max_instructions: int = 2_000_000,
             stack_top: int = None) -> int:
        """Run the function at `addr`; see _call_once. Retries after mapping a page
        when `map_faults` is set and the memory is journalled."""
        for _attempt in range(self.FAULT_RETRIES):
            journal = self.memory._journal
            mark = len(journal) if journal is not None else None
            try:
                return self._call_once(addr, args, max_instructions, stack_top)
            except MemoryFault as exc:
                if not self.map_faults or mark is None or not self.memory.map_page(exc.addr):
                    raise
                self.memory.undo_to(mark)
        return self._call_once(addr, args, max_instructions, stack_top)

    def _call_once(self, addr: int, args=(), max_instructions: int = 2_000_000,
                   stack_top: int = None) -> int:
        """Run the function at `addr` and return its $v0 as a signed 32-bit value.

        `args` follows the register calling convention: entries 0..7 go to $a0..$a3
        and $t0..$t3, entries 8..15 are float bit patterns for $f12..$f19. More than
        that is rejected rather than silently mishandled, since stack arguments need
        the caller's frame layout.
        """
        if len(args) > 16:
            raise EmulationError("only register arguments are supported")
        # Every call starts from the same register state. Registers used to keep
        # whatever the previous run left, so a callee reading one the analysis did
        # not attribute to it saw different garbage depending on what had run
        # before - and the same test could pass alone and fail in the pipeline.
        self.regs = [0] * 32
        self.regs_hi = [0] * 32
        self.hi = self.lo = self.hi1 = self.lo1 = 0
        self.fpr = [0] * 32
        self.fp_cond = False
        self.fp_acc = 0
        self.vf = vu0.initial_vf()
        self.vc = vu0.initial_vc()
        self.vacc = [0, 0, 0, 0]
        self.sa = 0
        for i, value in enumerate(args):
            if i < 8:
                self.set((4, 5, 6, 7, 8, 9, 10, 11)[i], value & MASK64)
            else:
                self.fpr[12 + i - 8] = value & 0xFFFFFFFF
        self.set(28, layout.CURRENT.gp)   # $gp
        self.set(29, layout.CURRENT.stack_top if stack_top is None else stack_top)
        self.set(31, RETURN_SENTINEL)     # $ra
        self.pc = addr
        self.instructions_executed = 0
        if not self._is_code(addr):
            self._stubbed_call()
            return 0

        budget = max_instructions
        while True:
            if self.pc == RETURN_SENTINEL:
                break
            self.step()
            self.instructions_executed += 1
            if self.instructions_executed > budget:
                raise ExecutionLimit(
                    "exceeded %d instructions; likely an infinite loop" % budget
                )
        return _sign32(self.get32(2))

    def run_to_return(self, addr: int, max_instructions: int = 2_000_000,
                      return_address: int = None, entry_sp: int = None) -> None:
        """Run the function at `addr` from the registers as they are, until it returns.

        For a caller that is not this interpreter - lifted code calling a function
        it does not have - which hands over its whole register state, $ra
        included, and takes the whole state back. Nothing is reset. The function
        has returned when control is back at the return address with the stack
        pointer no lower than it was: a recursive call reaches the same address
        only from deeper in the stack, while an epilogue fragment returns from
        higher up, having popped a frame it never pushed.
        """
        if not self._is_code(addr):
            self._stubbed_call()
            return
        # A call returns to $ra; a tail transfer to wherever its caller returns.
        if return_address is None:
            return_address = self.get32(31)
        if entry_sp is None:
            entry_sp = self.get32(29)
        self.pc = addr
        executed = 0
        while not (self.pc == return_address and self.get32(29) >= entry_sp):
            self.step()
            executed += 1
            if executed > max_instructions:
                raise ExecutionLimit(
                    "exceeded %d instructions; likely an infinite loop" % max_instructions)

    def _decode(self, word: int, pc: int):
        """Decode an instruction, reusing an earlier decode of the same word here.

        Decoding dominated the interpreter's time - operand text and all, for every
        instruction executed - and verifying functions that call into the original
        code runs a great deal of it. The cache is keyed on the word as well as the
        address, so code that changed in memory is decoded afresh.
        """
        cached = _DECODED.get(pc)
        if cached is not None and cached[0] == word:
            return cached[1]
        ins = DISASM.decode(word, pc)
        _DECODED[pc] = (word, ins)
        return ins

    def step(self) -> None:
        """Execute one instruction, honouring the delay slot."""
        if self.count_visits:
            self.visit_counts[self.pc] = self.visit_counts.get(self.pc, 0) + 1
        if not self._is_code(self.pc):
            # Not a MemoryFault: no page should ever be mapped to run code from.
            # Execution got here through a bad return address or a jump into data,
            # and what follows would be test data decoded as instructions.
            raise EmulationError("execution left the program at 0x%08X" % self.pc)
        word = self.memory.read_int(self.pc, 4, False)
        ins = self._decode(word, self.pc)
        if not ins.is_valid:
            raise UnsupportedInstruction(self.pc, word, ".word")

        if ins.has_delay_slot:
            target, link, nullify = self._resolve_transfer(ins)
            slot_pc = self.pc + 4
            saved = self.pc
            # A "likely" branch that is NOT taken nullifies its delay slot: the
            # instruction is skipped entirely rather than executed. Running it
            # anyway silently corrupts state - it is what made the SIMD strlen
            # return wrong lengths, by clobbering a mask register and advancing a
            # cursor that should have stood still.
            if not nullify:
                slot_word = self.memory.read_int(slot_pc, 4, False)
                slot_ins = self._decode(slot_word, slot_pc)
                if slot_ins.has_delay_slot:
                    raise UnsupportedInstruction(
                        slot_pc, slot_word, "branch in delay slot"
                    )
                self.pc = slot_pc
                self._execute(slot_ins)
            if link is not None:
                self.set(31, saved + 8)
            self.pc = target if target is not None else saved + 8
            indirect = ins.mnemonic == "jalr" or (ins.mnemonic == "jr" and ins.rs != 31)
            if indirect and target != RETURN_SENTINEL and not self._is_code(target):
                self._stubbed_call()
                # A jalr returns past its delay slot; a jr through a pointer is a
                # tail call, so it returns to this function's caller.
                self.pc = saved + 8 if ins.mnemonic == "jalr" else self.get32(31)
            return

        self._execute(ins)
        self.pc += 4

    def _is_code(self, address: int) -> bool:
        return self.code_start <= address < self.code_end

    def _stubbed_call(self) -> None:
        """A call to an address outside the executable: return 0, do nothing.

        Test memory is random, so a function pointer read from it points nowhere
        real, and executing it ran random bytes until something trapped. Every form
        of a function under test - the original, its derivation, the compiled C++ -
        reaches an indirect callee through this interpreter, so all of them meet
        the same stub, and the test checks the function's own logic around it.
        """
        self.stubbed_calls += 1
        self.set(2, 0)

    def _resolve_transfer(self, ins):
        """Decide the branch outcome.

        Returns (target or None, link or None, nullify_delay_slot).  The third
        value is True only for a "likely" branch that was not taken, which is the
        one case where the delay-slot instruction must be skipped.
        """
        m = ins.mnemonic
        rs = _sign64(self.get(ins.rs))
        rt = _sign64(self.get(ins.rt))
        link = True if ins.is_call else None

        if m in ("j", "b"):
            return ins.target, None, False
        if m == "jal":
            return ins.target, True, False
        if m == "jr":
            return self.get32(ins.rs), None, False
        if m == "jalr":
            return self.get32(ins.rs), True, False
        if m in ("bc0t", "bc0tl", "bc0f", "bc0fl"):
            # COP0's condition is the DMA-complete signal that code polls with
            # `bc0f` until a transfer ends. Like the hardware registers it reads
            # as done, so the poll ends at once.
            taken = m.startswith("bc0t")
            if taken:
                return ins.target, None, False
            return (self.pc + 8, None, True) if m.endswith("l") else (None, None, False)
        if m in ("bc1t", "bc1tl"):
            taken = self.fp_cond
            if taken:
                return ins.target, None, False
            return (self.pc + 8, None, True) if m.endswith("l") else (None, None, False)
        if m in ("bc1f", "bc1fl"):
            taken = not self.fp_cond
            if taken:
                return ins.target, None, False
            return (self.pc + 8, None, True) if m.endswith("l") else (None, None, False)

        taken = {
            "beq": rs == rt,
            "bne": rs != rt,
            "beql": rs == rt,
            "bnel": rs != rt,
            "blez": rs <= 0,
            "bgtz": rs > 0,
            "blezl": rs <= 0,
            "bgtzl": rs > 0,
            "bltz": rs < 0,
            "bgez": rs >= 0,
            "bltzl": rs < 0,
            "bgezl": rs >= 0,
            "bltzal": rs < 0,
            "bgezal": rs >= 0,
        }.get(m)
        if taken is None:
            raise UnsupportedInstruction(self.pc, ins.raw, m)
        if taken:
            return ins.target, link, False
        # Not taken: a "likely" branch nullifies its delay slot, so the caller is
        # told to skip it and resume past the pair.
        if ins.is_likely:
            return self.pc + 8, link, True
        return None, link, False

    def _execute(self, ins) -> None:
        m = ins.mnemonic
        if m in _NO_EFFECT:
            return
        if ins.raw >> 26 == 0x12:
            _cop2(self, ins)
            return

        handler = _HANDLERS.get(m)
        if handler is None:
            raise UnsupportedInstruction(ins.addr, ins.raw, m)
        handler(self, ins)


# -- instruction handlers ------------------------------------------------
#
# Kept as a dispatch table rather than a long if-chain so the supported subset is
# explicit and easy to audit.

def _alu_rrr(fn):
    def run(m: Machine, ins) -> None:
        m.set32(ins.rd, fn(m.get32(ins.rs), m.get32(ins.rt)))
    return run


def _alu_d64(fn):
    def run(m: Machine, ins) -> None:
        m.set(ins.rd, fn(m.get(ins.rs), m.get(ins.rt)) & MASK64)
    return run


def _load(size: int, signed: bool):
    def run(m: Machine, ins) -> None:
        addr = (m.get32(ins.rs) + ins.imm) & MASK32
        value = m.memory.read_int(addr, size, signed)
        # Unsigned loads zero-extend into the whole register; lwu of 0xFFFFFFF0 is
        # 0x00000000FFFFFFF0, not the sign-extended word.
        if size == 8 or not signed:
            m.set(ins.rt, value & MASK64)
        else:
            m.set32(ins.rt, value)
    return run


def _store(size: int):
    def run(m: Machine, ins) -> None:
        addr = (m.get32(ins.rs) + ins.imm) & MASK32
        m.memory.write_int(addr, m.get(ins.rt), size)
    return run


def _shift(fn, use_reg: bool, width: int = 32):
    def run(m: Machine, ins) -> None:
        # Word shifts take the amount mod 32, doubleword shifts mod 64.
        amount = (m.get(ins.rs) & (width - 1)) if use_reg else ins.imm
        if width == 32:
            m.set32(ins.rd, fn(m.get32(ins.rt), amount))
        else:
            m.set(ins.rd, fn(m.get(ins.rt), amount) & MASK64)
    return run


def _op_addiu(m: Machine, ins) -> None:
    m.set32(ins.rt, m.get32(ins.rs) + ins.imm)


def _op_lui(m: Machine, ins) -> None:
    m.set32(ins.rt, (ins.imm & 0xFFFF) << 16)


def _op_andi(m: Machine, ins) -> None:
    m.set(ins.rt, m.get(ins.rs) & (ins.imm & 0xFFFF))


def _op_ori(m: Machine, ins) -> None:
    m.set(ins.rt, m.get(ins.rs) | (ins.imm & 0xFFFF))


def _op_xori(m: Machine, ins) -> None:
    m.set(ins.rt, m.get(ins.rs) ^ (ins.imm & 0xFFFF))


def _op_slti(m: Machine, ins) -> None:
    m.set32(ins.rt, 1 if _sign64(m.get(ins.rs)) < ins.imm else 0)


def _op_sltiu(m: Machine, ins) -> None:
    m.set32(ins.rt, 1 if m.get(ins.rs) < (ins.imm & MASK64) else 0)


def _op_slt(m: Machine, ins) -> None:
    m.set32(ins.rd, 1 if _sign64(m.get(ins.rs)) < _sign64(m.get(ins.rt)) else 0)


def _op_sltu(m: Machine, ins) -> None:
    m.set32(ins.rd, 1 if m.get(ins.rs) < m.get(ins.rt) else 0)


def _op_mult(m: Machine, ins) -> None:
    product = _sign32(m.get32(ins.rs)) * _sign32(m.get32(ins.rt))
    m.lo = _sign32(product & MASK32) & MASK64
    m.hi = _sign32((product >> 32) & MASK32) & MASK64
    if ins.rd:
        m.set32(ins.rd, product & MASK32)


def _op_multu(m: Machine, ins) -> None:
    product = m.get32(ins.rs) * m.get32(ins.rt)
    m.lo = _sign32(product & MASK32) & MASK64
    m.hi = _sign32((product >> 32) & MASK32) & MASK64
    if ins.rd:
        m.set32(ins.rd, product & MASK32)


def divide(a: int, b: int, signed: bool):
    """(LO, HI) of a 32-bit division as the EE computes it, sign-extended.

    Every input has a result. A zero divisor gives LO = 1 for a negative signed
    dividend and -1 otherwise, with the dividend in HI; INT_MIN / -1 gives INT_MIN
    and 0. These are the EE's answers as PCSX2 documents them. Treating division
    by zero as "HI/LO unchanged", as this interpreter once did, made every
    derivation of an unguarded divide depend on stale registers.
    """
    if signed:
        a, b = _sign32(a), _sign32(b)
        if b == 0:
            return (1 if a < 0 else -1) & MASK64, a & MASK64
        q = abs(a) // abs(b)
        if (a < 0) != (b < 0):
            q = -q
        return _sign32(q) & MASK64, _sign32(a - b * q) & MASK64
    a, b = a & MASK32, b & MASK32
    if b == 0:
        return MASK64, _sign32(a) & MASK64
    return _sign32(a // b) & MASK64, _sign32(a % b) & MASK64


def _op_div(m: Machine, ins) -> None:
    m.lo, m.hi = divide(m.get32(ins.rs), m.get32(ins.rt), True)


def _op_divu(m: Machine, ins) -> None:
    m.lo, m.hi = divide(m.get32(ins.rs), m.get32(ins.rt), False)


def _op_mfhi(m: Machine, ins) -> None:
    m.set(ins.rd, m.hi)


def _op_mflo(m: Machine, ins) -> None:
    m.set(ins.rd, m.lo)


def _op_mthi(m: Machine, ins) -> None:
    m.hi = m.get(ins.rs)


def _op_mtlo(m: Machine, ins) -> None:
    m.lo = m.get(ins.rs)


# The EE has a second multiply/divide pipeline: HI1/LO1 are the upper halves of the
# 128-bit HI/LO, and mult1/div1/... are the ordinary operations aimed at them.
# madd/maddu accumulate a 32x32 product into the 64-bit value in HI:LO's low
# words. Each is written in terms of the pipeline-0 version on a swapped-in HI/LO.

def _pipeline1(op):
    def run(m: Machine, ins) -> None:
        m.hi, m.hi1 = m.hi1, m.hi
        m.lo, m.lo1 = m.lo1, m.lo
        try:
            op(m, ins)
        finally:
            m.hi, m.hi1 = m.hi1, m.hi
            m.lo, m.lo1 = m.lo1, m.lo
    return run


def _madd(signed: bool):
    def run(m: Machine, ins) -> None:
        if signed:
            product = _sign32(m.get32(ins.rs)) * _sign32(m.get32(ins.rt))
        else:
            product = m.get32(ins.rs) * m.get32(ins.rt)
        total = (((m.hi & MASK32) << 32) | (m.lo & MASK32)) + product
        m.lo = _sign32(total & MASK32) & MASK64
        m.hi = _sign32((total >> 32) & MASK32) & MASK64
        if ins.rd:
            m.set(ins.rd, m.lo)
    return run


def _op_movz(m: Machine, ins) -> None:
    if m.get(ins.rt) == 0:
        m.set(ins.rd, m.get(ins.rs))


def _op_movn(m: Machine, ins) -> None:
    if m.get(ins.rt) != 0:
        m.set(ins.rd, m.get(ins.rs))



def _op_lq(m: Machine, ins) -> None:
    """128-bit load. The address is forced to quadword alignment by hardware."""
    addr = (m.get32(ins.rs) + ins.imm) & MASK32 & ~0xF
    m.set(ins.rt, m.memory.read_int(addr, 8, False))
    m.regs_hi[ins.rt] = m.memory.read_int(addr + 8, 8, False)


def _op_sq(m: Machine, ins) -> None:
    addr = (m.get32(ins.rs) + ins.imm) & MASK32 & ~0xF
    m.memory.write_int(addr, m.get(ins.rt), 8)
    m.memory.write_int(addr + 8, m.regs_hi[ins.rt], 8)


# The largest finite single-precision magnitude, bit pattern 0x7F7FFFFF.
FLOAT_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


def _f32(value: float) -> float:
    """Round a Python double to single precision, as the EE's FPU would.

    The Emotion Engine's FPU is deliberately not IEEE-754: it has no infinities and
    no NaNs, and an operation that overflows saturates to the largest representable
    magnitude rather than becoming infinite.

    Modelling that is not optional here. Python computes in double precision, so an
    overflowing `mul.s` reaches `struct.pack` with a value no float32 can hold and
    raises OverflowError - which showed up as the interpreter crashing outright on
    real game code. Saturating is both the fix and the more faithful behaviour, and
    it is a quirk any port has to reproduce or physics will diverge.
    """
    if value != value:          # NaN has no representation on this FPU
        return 0.0
    if value > FLOAT_MAX:
        return FLOAT_MAX
    if value < -FLOAT_MAX:
        return -FLOAT_MAX
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _fp_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _fp_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


# COP1 R-format packs its registers differently from the integer unit, and the
# disassembler does not populate rs/rt/rd for these forms, so the fields are taken
# straight from the encoding.
def _fd(ins) -> int:
    return (ins.raw >> 6) & 0x1F


def _fs(ins) -> int:
    return (ins.raw >> 11) & 0x1F


def _ft(ins) -> int:
    return (ins.raw >> 16) & 0x1F


def _op_lwc1(m: Machine, ins) -> None:
    # A bit move: no interpretation, so a NaN payload survives intact. Converting
    # to a Python float here and back on the matching `swc1` sets the quiet-NaN bit
    # and silently alters values that a copy must preserve.
    addr = (m.get32(ins.rs) + ins.imm) & MASK32
    m.fpr[ins.rt] = m.memory.read_int(addr, 4, False)


def _op_swc1(m: Machine, ins) -> None:
    addr = (m.get32(ins.rs) + ins.imm) & MASK32
    m.memory.write_int(addr, m.fpr[ins.rt] & MASK32, 4)


def _op_mtc1(m: Machine, ins) -> None:
    m.fpr[ins.rd] = m.get32(ins.rt)


def _op_mfc1(m: Machine, ins) -> None:
    m.set32(ins.rt, m.fpr[ins.rd] & MASK32)


# The FPU control registers. FCR0 reads as the EE FPU's implementation number.
# Of FCR31 only the condition bit (23) has an effect - the EE rounds toward zero
# and raises no exceptions whatever the other bits say - so only it is kept: a
# start-up `ctc1 $zero, $31` clears the condition, and `cfc1` reads it back.
FPU_IMPLEMENTATION = 0x2E00
FCR31_CONDITION = 1 << 23


def _op_cfc1(m: Machine, ins) -> None:
    if ins.rd == 0:
        value = FPU_IMPLEMENTATION
    elif ins.rd == 31:
        value = FCR31_CONDITION if m.fp_cond else 0
    else:
        value = 0
    m.set32(ins.rt, value)


def _op_ctc1(m: Machine, ins) -> None:
    if ins.rd == 31:
        m.fp_cond = bool(m.get32(ins.rt) & FCR31_CONDITION)


# Float registers hold raw bit patterns, so arithmetic converts in and out
# explicitly. Confining the conversion to these helpers is what keeps the bit moves
# above exact.
def _fp_arith(fn):
    def run(m: Machine, ins) -> None:
        a = _fp_from_bits(m.fpr[_fs(ins)])
        b = _fp_from_bits(m.fpr[_ft(ins)])
        m.fpr[_fd(ins)] = _fp_bits(_f32(fn(a, b)))
    return run


def _fp_unary(fn):
    def run(m: Machine, ins) -> None:
        m.fpr[_fd(ins)] = _fp_bits(_f32(fn(_fp_from_bits(m.fpr[_fs(ins)]))))
    return run


def _fp_compare(fn):
    def run(m: Machine, ins) -> None:
        m.fp_cond = fn(
            _fp_from_bits(m.fpr[_fs(ins)]), _fp_from_bits(m.fpr[_ft(ins)])
        )
    return run


def _op_sqrt_s(m: Machine, ins) -> None:
    """`sqrt.s fd, ft`: unlike the other unary operations, the operand is in the ft
    field (fs is zero). Reading fs took the root of $f0 instead, and the symbolic
    model made the same mistake, so the two agreed and verified it."""
    a = _fp_from_bits(m.fpr[_ft(ins)])
    m.fpr[_fd(ins)] = _fp_bits(_f32(a ** 0.5 if a >= 0 else 0.0))


def float_to_word(bits: int) -> int:
    """`cvt.w.s` as the EE does it: truncating toward zero below 2**31 in
    magnitude (an exponent up to 2**30's), and saturating beyond - 0x7FFFFFFF,
    or 0x80000000 for a negative - whatever the rest of the bits (the FPU has
    no infinities or NaNs). Wrapping to the low word instead turned particles
    by the camera into garbage colours (FINDINGS.md, "cvt.w.s saturates")."""
    bits &= MASK32
    if (bits & 0x7F800000) <= 0x4E800000:
        return int(_fp_from_bits(bits)) & MASK32
    return 0x80000000 if bits & 0x80000000 else 0x7FFFFFFF


def _op_cvt_w_s(m: Machine, ins) -> None:
    """Float to fixed, truncating toward zero and saturating as the EE does."""
    m.fpr[_fd(ins)] = float_to_word(m.fpr[_fs(ins)])


def _op_cvt_s_w(m: Machine, ins) -> None:
    m.fpr[_fd(ins)] = _fp_bits(_f32(float(_sign32(m.fpr[_fs(ins)]))))


def _op_div_s(m: Machine, ins) -> None:
    divisor = _fp_from_bits(m.fpr[_ft(ins)])
    if divisor == 0.0:
        # The FPU flags this rather than trapping, so yield zero instead of raising.
        m.fpr[_fd(ins)] = _fp_bits(0.0)
        return
    m.fpr[_fd(ins)] = _fp_bits(_f32(_fp_from_bits(m.fpr[_fs(ins)]) / divisor))



# -- MMI: the 128-bit integer SIMD subset -------------------------------
#
# Only the operations the real code actually leans on are implemented. The
# optimised strlen alone needs psubb, pnor, pand and the pcpy family, and it sits
# under a great many string routines.

def _q_get(m: Machine, index: int) -> int:
    """Read a register as one 128-bit value."""
    return (m.regs_hi[index] << 64) | m.get(index)


def _q_set(m: Machine, index: int, value: int) -> None:
    if index == 0:
        return
    m.regs[index] = value & MASK64
    m.regs_hi[index] = (value >> 64) & MASK64


def _q_bitwise(fn):
    def run(m: Machine, ins) -> None:
        _q_set(m, ins.rd, fn(_q_get(m, ins.rs), _q_get(m, ins.rt)) & ((1 << 128) - 1))
    return run


def _q_perbyte(fn):
    """Apply `fn` to each of the sixteen byte lanes independently."""
    def run(m: Machine, ins) -> None:
        a, b = _q_get(m, ins.rs), _q_get(m, ins.rt)
        out = 0
        for lane in range(16):
            shift = lane * 8
            out |= (fn((a >> shift) & 0xFF, (b >> shift) & 0xFF) & 0xFF) << shift
        _q_set(m, ins.rd, out)
    return run


def _op_pcpyld(m: Machine, ins) -> None:
    """Low doubleword of rt into the low half, low doubleword of rs into the high."""
    # Both operands are read before either half is written: `pcpyld $v1, $v1, $v0`
    # names $v1 as source and destination, and writing the low half first used to
    # put the new value in the high half too.
    if ins.rd:
        low, high = m.get(ins.rt), m.get(ins.rs)
        m.regs[ins.rd] = low
        m.regs_hi[ins.rd] = high


def _op_pcpyud(m: Machine, ins) -> None:
    """High doublewords of rs and rt, packed down."""
    if ins.rd:
        m.regs[ins.rd] = m.regs_hi[ins.rs]
        m.regs_hi[ins.rd] = m.regs_hi[ins.rt]


def _op_prot3w(m: Machine, ins) -> None:
    """Rotate rt's three low words left one place; the high word stays.

    rd.w0 = rt.w1, rd.w1 = rt.w2, rd.w2 = rt.w0, rd.w3 = rt.w3 (EE core manual,
    PROT3W). rt is read whole before rd is written, so rd may name rt."""
    v = _q_get(m, ins.rt)
    w = [(v >> (32 * i)) & 0xFFFFFFFF for i in range(4)]
    _q_set(m, ins.rd, w[1] | (w[2] << 32) | (w[0] << 64) | (w[3] << 96))


def _op_pcpyh(m: Machine, ins) -> None:
    """Broadcast the low halfword of each doubleword across it."""
    value = _q_get(m, ins.rt)
    lo_h = value & 0xFFFF
    hi_h = (value >> 64) & 0xFFFF
    out = 0
    for lane in range(4):
        out |= lo_h << (lane * 16)
        out |= hi_h << (64 + lane * 16)
    _q_set(m, ins.rd, out)


def _q_lanes(width: int, fn):
    """Apply `fn(a, b)` to each `width`-bit lane of rs and rt, as unsigned values."""
    mask = (1 << width) - 1

    def run(m: Machine, ins) -> None:
        a, b = _q_get(m, ins.rs), _q_get(m, ins.rt)
        out = 0
        for shift in range(0, 128, width):
            out |= (fn((a >> shift) & mask, (b >> shift) & mask) & mask) << shift
        _q_set(m, ins.rd, out)
    return run


def _s32(value: int) -> int:
    return value - (1 << 32) if value & 0x80000000 else value


def _signed(width: int):
    """A lane, read as a signed `width`-bit number."""
    top = 1 << (width - 1)
    return lambda value: value - (top << 1) if value & top else value


def _q_compare(width: int, greater: bool):
    """pcgt*/pceq*: all ones in a lane where rs > rt (signed), or rs == rt."""
    signed, ones = _signed(width), (1 << width) - 1
    if greater:
        return _q_lanes(width, lambda a, b: ones if signed(a) > signed(b) else 0)
    return _q_lanes(width, lambda a, b: ones if a == b else 0)


def _q_upper_interleave(width: int):
    """pextuh/pextuw: the upper doubleword's lanes of rt and rs, alternated."""
    mask = (1 << width) - 1

    def run(m: Machine, ins) -> None:
        a, b, out = _q_get(m, ins.rs) >> 64, _q_get(m, ins.rt) >> 64, 0
        for i in range(64 // width):
            out |= ((b >> (i * width)) & mask) << (2 * i * width)
            out |= ((a >> (i * width)) & mask) << ((2 * i + 1) * width)
        _q_set(m, ins.rd, out)
    return run


def _op_psrlh(m: Machine, ins) -> None:
    value, out, amount = _q_get(m, ins.rt), 0, ins.imm & 0xF
    for shift in range(0, 128, 16):
        out |= (((value >> shift) & 0xFFFF) >> amount) << shift
    _q_set(m, ins.rd, out)


def _op_pextlw(m: Machine, ins) -> None:
    """Interleave the low words: rt.w0, rs.w0, rt.w1, rs.w1."""
    a, b = m.get(ins.rs), m.get(ins.rt)
    words = (b & MASK32, a & MASK32, b >> 32, a >> 32)
    _q_set(m, ins.rd, sum(word << (32 * i) for i, word in enumerate(words)))


# Unaligned loads and stores, little-endian. The "left" forms move the bytes from
# the address down to the aligned boundary into the high end of the register;
# the "right" forms move the bytes from the address up into the low end. A pair
# (lwl at p+3, lwr at p) assembles the word at an unaligned p.

def _unaligned_address(m: Machine, ins) -> int:
    return (m.get32(ins.rs) + ins.imm) & MASK32


def _op_lwl(m: Machine, ins) -> None:
    addr = _unaligned_address(m, ins)
    keep = (3 - (addr & 3)) * 8
    word = m.memory.read_int(addr & ~3, 4, False)
    m.set32(ins.rt, (word << keep) | (m.get32(ins.rt) & ((1 << keep) - 1)))


def _op_lwr(m: Machine, ins) -> None:
    addr = _unaligned_address(m, ins)
    drop = (addr & 3) * 8
    word = m.memory.read_int(addr & ~3, 4, False)
    merged = ((word >> drop) | (m.get32(ins.rt) & ~(MASK32 >> drop))) & MASK32
    if drop == 0:
        m.set32(ins.rt, merged)
    else:
        # A partial word leaves the register's upper half as it was.
        m.set(ins.rt, (m.get(ins.rt) & ~MASK32) | merged)


def _op_ldl(m: Machine, ins) -> None:
    addr = _unaligned_address(m, ins)
    keep = (7 - (addr & 7)) * 8
    double = m.memory.read_int(addr & ~7, 8, False)
    m.set(ins.rt, ((double << keep) | (m.get(ins.rt) & ((1 << keep) - 1))) & MASK64)


def _op_ldr(m: Machine, ins) -> None:
    addr = _unaligned_address(m, ins)
    drop = (addr & 7) * 8
    double = m.memory.read_int(addr & ~7, 8, False)
    m.set(ins.rt, ((double >> drop) | (m.get(ins.rt) & ~(MASK64 >> drop))) & MASK64)


def _store_left(size: int):
    def run(m: Machine, ins) -> None:
        addr = _unaligned_address(m, ins)
        offset = addr & (size - 1)
        count = offset + 1
        value = (m.get(ins.rt) & ((1 << (8 * size)) - 1)) >> (8 * (size - count))
        m.memory.write(addr - offset, value.to_bytes(count, "little"))
    return run


def _store_right(size: int):
    def run(m: Machine, ins) -> None:
        addr = _unaligned_address(m, ins)
        count = size - (addr & (size - 1))
        value = m.get(ins.rt) & ((1 << (8 * count)) - 1)
        m.memory.write(addr, value.to_bytes(count, "little"))
    return run


def _q_shift(width: int, kind: str):
    """Shift each `width`-bit lane of rt by the sa field: sll, srl or sra."""
    mask = (1 << width) - 1

    def run(m: Machine, ins) -> None:
        value, amount, out = _q_get(m, ins.rt), ins.imm & (width - 1), 0
        for shift in range(0, 128, width):
            lane = (value >> shift) & mask
            if kind == "sll":
                lane = (lane << amount) & mask
            elif kind == "srl":
                lane >>= amount
            else:
                signed = lane - (1 << width) if lane >> (width - 1) else lane
                lane = (signed >> amount) & mask
            out |= lane << shift
        _q_set(m, ins.rd, out)
    return run


def _q_interleave(width: int):
    """pextlb/pextlh: the low doubleword's lanes of rt and rs, alternated."""
    mask = (1 << width) - 1

    def run(m: Machine, ins) -> None:
        a, b, out = m.get(ins.rs), m.get(ins.rt), 0
        for i in range(64 // width):
            out |= ((b >> (i * width)) & mask) << (2 * i * width)
            out |= ((a >> (i * width)) & mask) << ((2 * i + 1) * width)
        _q_set(m, ins.rd, out)
    return run


def _op_ppacb(m: Machine, ins) -> None:
    """The even bytes of rt, then of rs."""
    a, b, out = _q_get(m, ins.rs), _q_get(m, ins.rt), 0
    for i in range(8):
        out |= ((b >> (16 * i)) & 0xFF) << (8 * i)
        out |= ((a >> (16 * i)) & 0xFF) << (64 + 8 * i)
    _q_set(m, ins.rd, out)


def _op_pextub(m: Machine, ins) -> None:
    """pextlb on the upper doublewords: their bytes of rt and rs, alternated."""
    a, b, out = _q_get(m, ins.rs) >> 64, _q_get(m, ins.rt) >> 64, 0
    for i in range(8):
        out |= ((b >> (8 * i)) & 0xFF) << (16 * i)
        out |= ((a >> (8 * i)) & 0xFF) << (16 * i + 8)
    _q_set(m, ins.rd, out)


def _op_ppach(m: Machine, ins) -> None:
    """The even halfwords of rt, then of rs."""
    a, b, out = _q_get(m, ins.rs), _q_get(m, ins.rt), 0
    for i in range(4):
        out |= ((b >> (32 * i)) & 0xFFFF) << (16 * i)
        out |= ((a >> (32 * i)) & 0xFFFF) << (64 + 16 * i)
    _q_set(m, ins.rd, out)


def _op_ppacw(m: Machine, ins) -> None:
    """The even words of rt, then of rs (EE core manual, PPACW)."""
    a, b = _q_get(m, ins.rs), _q_get(m, ins.rt)
    lo = (b & 0xFFFFFFFF) | (((b >> 64) & 0xFFFFFFFF) << 32)
    hi = (a & 0xFFFFFFFF) | (((a >> 64) & 0xFFFFFFFF) << 32)
    _q_set(m, ins.rd, lo | (hi << 64))


def _op_qfsrv(m: Machine, ins) -> None:
    """The low 128 bits of rs:rt (256 bits, rs high) shifted right by SA."""
    value = (_q_get(m, ins.rs) << 128) | _q_get(m, ins.rt)
    _q_set(m, ins.rd, (value >> (m.sa & 0xFF)) & ((1 << 128) - 1))


def _op_mtsab(m: Machine, ins) -> None:
    m.sa = ((m.get(ins.rs) ^ ins.imm) & 0xF) * 8


def _op_mtsah(m: Machine, ins) -> None:
    m.sa = ((m.get(ins.rs) ^ ins.imm) & 0x7) * 16


def _op_rsqrt_s(m: Machine, ins) -> None:
    """fd = fs / sqrt(|ft|); a zero root gives the largest magnitude, fs's sign."""
    numerator = m.fpr[_fs(ins)]
    root = abs(_fp_from_bits(m.fpr[_ft(ins)])) ** 0.5
    if root == 0.0:
        m.fpr[_fd(ins)] = (numerator & 0x80000000) | 0x7F7FFFFF
    else:
        m.fpr[_fd(ins)] = _fp_bits(_f32(_fp_from_bits(numerator) / root))


_HANDLERS = {
    # MMI (128-bit integer SIMD) subset
    "pand": _q_bitwise(lambda a, b: a & b),
    "por": _q_bitwise(lambda a, b: a | b),
    "pxor": _q_bitwise(lambda a, b: a ^ b),
    "pnor": _q_bitwise(lambda a, b: ~(a | b)),
    "paddb": _q_perbyte(lambda a, b: a + b),
    "psubb": _q_perbyte(lambda a, b: a - b),
    "pcpyld": _op_pcpyld,
    "pcpyud": _op_pcpyud,
    "pcpyh": _op_pcpyh,
    "prot3w": _op_prot3w,
    "pmaxw": _q_lanes(32, lambda a, b: a if _s32(a) >= _s32(b) else b),
    "pminw": _q_lanes(32, lambda a, b: a if _s32(a) <= _s32(b) else b),
    "psubw": _q_lanes(32, lambda a, b: a - b),
    "paddw": _q_lanes(32, lambda a, b: a + b),
    "paddh": _q_lanes(16, lambda a, b: a + b),
    "ppacb": _op_ppacb, "pextub": _op_pextub,
    "padduw": _q_lanes(32, lambda a, b: min(a + b, MASK32)),
    "paddub": _q_lanes(8, lambda a, b: min(a + b, 0xFF)),
    # |x|, except that 0x80000000 has none and saturates to 0x7FFFFFFF.
    "pabsw": _q_lanes(32, lambda a, b: min(abs(_s32(b)), 0x7FFFFFFF)),
    "psrlh": _op_psrlh,
    "psllh": _q_shift(16, "sll"), "psrah": _q_shift(16, "sra"),
    "psllw": _q_shift(32, "sll"), "psrlw": _q_shift(32, "srl"), "psraw": _q_shift(32, "sra"),
    "pextlb": _q_interleave(8), "pextlh": _q_interleave(16),
    "ppach": _op_ppach, "ppacw": _op_ppacw, "qfsrv": _op_qfsrv, "mtsab": _op_mtsab, "mtsah": _op_mtsah,
    "rsqrt.s": _op_rsqrt_s,
    "pextlw": _op_pextlw,
    "pcgtb": _q_compare(8, True), "pcgth": _q_compare(16, True), "pcgtw": _q_compare(32, True),
    "pceqb": _q_compare(8, False), "pceqh": _q_compare(16, False), "pceqw": _q_compare(32, False),
    "pmaxh": _q_lanes(16, lambda a, b: a if _signed(16)(a) >= _signed(16)(b) else b),
    "pminh": _q_lanes(16, lambda a, b: a if _signed(16)(a) <= _signed(16)(b) else b),
    "pextuh": _q_upper_interleave(16), "pextuw": _q_upper_interleave(32),
    # Unaligned loads and stores
    "lwl": _op_lwl, "lwr": _op_lwr, "ldl": _op_ldl, "ldr": _op_ldr,
    "swl": _store_left(4), "swr": _store_right(4),
    "sdl": _store_left(8), "sdr": _store_right(8),
    # 128-bit load/store, used for register spills
    "lq": _op_lq,
    "sq": _op_sq,
    # Coprocessor 1 (single precision)
    "lwc1": _op_lwc1,
    "swc1": _op_swc1,
    "mtc1": _op_mtc1,
    "mfc1": _op_mfc1,
    "cfc1": _op_cfc1,
    "ctc1": _op_ctc1,
    "add.s": _fp_arith(lambda a, b: a + b),
    "sub.s": _fp_arith(lambda a, b: a - b),
    "mul.s": _fp_arith(lambda a, b: a * b),
    "div.s": _op_div_s,
    "max.s": _fp_arith(max),
    "min.s": _fp_arith(min),
    "mov.s": _fp_unary(lambda a: a),
    "neg.s": _fp_unary(lambda a: -a),
    "abs.s": _fp_unary(abs),
    "sqrt.s": _op_sqrt_s,
    "cvt.w.s": _op_cvt_w_s,
    "cvt.s.w": _op_cvt_s_w,
    "c.eq.s": _fp_compare(lambda a, b: a == b),
    "c.lt.s": _fp_compare(lambda a, b: a < b),
    "c.le.s": _fp_compare(lambda a, b: a <= b),
    "c.f.s": _fp_compare(lambda a, b: False),
    # I-type arithmetic and logic
    "addiu": _op_addiu,
    "addi": _op_addiu,
    "daddiu": lambda m, i: m.set(i.rt, (m.get(i.rs) + i.imm) & MASK64),
    "daddi": lambda m, i: m.set(i.rt, (m.get(i.rs) + i.imm) & MASK64),
    "lui": _op_lui,
    "andi": _op_andi,
    "ori": _op_ori,
    "xori": _op_xori,
    "slti": _op_slti,
    "sltiu": _op_sltiu,
    # Register arithmetic and logic
    "addu": _alu_rrr(lambda a, b: a + b),
    "add": _alu_rrr(lambda a, b: a + b),
    "subu": _alu_rrr(lambda a, b: a - b),
    "sub": _alu_rrr(lambda a, b: a - b),
    "and": _alu_d64(lambda a, b: a & b),
    "or": _alu_d64(lambda a, b: a | b),
    "xor": _alu_d64(lambda a, b: a ^ b),
    "nor": _alu_d64(lambda a, b: ~(a | b)),
    "daddu": _alu_d64(lambda a, b: a + b),
    "dadd": _alu_d64(lambda a, b: a + b),
    "dsubu": _alu_d64(lambda a, b: a - b),
    "dsub": _alu_d64(lambda a, b: a - b),
    "slt": _op_slt,
    "sltu": _op_sltu,
    "movz": _op_movz,
    "movn": _op_movn,
    # Shifts
    "sll": _shift(lambda v, n: v << n, False),
    "srl": _shift(lambda v, n: (v & MASK32) >> n, False),
    "sra": _shift(lambda v, n: _sign32(v) >> n, False),
    "sllv": _shift(lambda v, n: v << n, True),
    "srlv": _shift(lambda v, n: (v & MASK32) >> n, True),
    "srav": _shift(lambda v, n: _sign32(v) >> n, True),
    "dsll": _shift(lambda v, n: v << n, False, 64),
    "dsrl": _shift(lambda v, n: (v & MASK64) >> n, False, 64),
    "dsra": _shift(lambda v, n: _sign64(v) >> n, False, 64),
    "dsll32": _shift(lambda v, n: v << (n + 32), False, 64),
    "dsrl32": _shift(lambda v, n: (v & MASK64) >> (n + 32), False, 64),
    "dsra32": _shift(lambda v, n: _sign64(v) >> (n + 32), False, 64),
    "dsllv": _shift(lambda v, n: v << n, True, 64),
    "dsrlv": _shift(lambda v, n: (v & MASK64) >> n, True, 64),
    "dsrav": _shift(lambda v, n: _sign64(v) >> n, True, 64),
    # Multiply and divide
    "mult": _op_mult,
    "multu": _op_multu,
    "div": _op_div,
    "divu": _op_divu,
    "mfhi": _op_mfhi,
    "mflo": _op_mflo,
    "mthi": _op_mthi,
    "mtlo": _op_mtlo,
    "madd": _madd(True),
    "maddu": _madd(False),
    "mult1": _pipeline1(_op_mult),
    "multu1": _pipeline1(_op_multu),
    "div1": _pipeline1(_op_div),
    "divu1": _pipeline1(_op_divu),
    "mfhi1": _pipeline1(_op_mfhi),
    "mflo1": _pipeline1(_op_mflo),
    "mthi1": _pipeline1(_op_mthi),
    "mtlo1": _pipeline1(_op_mtlo),
    "madd1": _pipeline1(_madd(True)),
    "maddu1": _pipeline1(_madd(False)),
    # Loads
    "lb": _load(1, True),
    "lbu": _load(1, False),
    "lh": _load(2, True),
    "lhu": _load(2, False),
    "lw": _load(4, True),
    "lwu": _load(4, False),
    "ld": _load(8, False),
    # Stores
    "sb": _store(1),
    "sh": _store(2),
    "sw": _store(4),
    "sd": _store(8),
}

# -- system instructions, stubbed for testing ---------------------------------
#
# These exist so that *callees* reached during a test can run: a derived function's
# own code never contains them (the symbolic model has no rule for them), but the
# original functions it calls often do. Original and derivation always run the
# same stubs, so a stub's choice of behaviour cannot make them disagree - it only
# decides which functions can be tested at all. The assumptions are deliberately
# the plainest available:
#
#   syscall  - the kernel call does nothing and returns 0, except the services
#              in SYSCALL_RESULTS.
#   mfc0     - every system-control register reads as 0. Count in particular is
#              constant, so a busy-wait on it never ends and the test runs out
#              of budget, rather than finishing with a made-up time.
#   traps    - trap exactly as the hardware does, which ends the test.
#   calls outside the executable - return 0 and do nothing (Machine._stubbed_call).
#   ei/di    - do nothing. No interrupt ever arrives during a test, so whether
#              they are enabled changes nothing observable, and mfc0 reading
#              Status as 0 already reports them disabled either way.

# Services whose stub result is not 0. SifGetReg reads a SIF register, and code
# polls it until the IOP sets a flag (`while (!(SifGetReg(4) & 0x20000))`); with
# every bit set, the IOP is up and has answered, and the poll ends.
SYSCALL_RESULTS = {
    0x7A: 0xFFFFFFFF,     # SifGetReg
}


def _cop2(m: Machine, ins) -> None:
    """A COP2 instruction: VU0 in macro mode (vu0.py)."""
    name = ins.mnemonic
    try:
        if name in ("qmtc2", "ctc2"):
            vu0.move_to(m, ins.raw, name, m.get(ins.rt), m.regs_hi[ins.rt])
        elif name in ("qmfc2", "cfc2"):
            low, high = vu0.move_from(m, ins.raw, name)
            if name == "cfc2":
                m.set32(ins.rt, low)
            elif ins.rt:
                m.regs[ins.rt] = low & MASK64
                m.regs_hi[ins.rt] = high & MASK64
        elif name.split(".")[0] in vu0.SUPPORTED:
            vu0.execute(m, ins.raw, name)
        else:
            raise vu0.Refused(name)
    except vu0.Refused:
        raise UnsupportedInstruction(ins.addr, ins.raw, name)


def _op_lqc2(m: Machine, ins) -> None:
    addr = (m.get32(ins.rs) + ins.imm) & MASK32 & ~0xF
    lanes = [m.memory.read_int(addr + 4 * i, 4, False) for i in range(4)]
    if ins.rt:
        m.vf[ins.rt] = lanes


def _op_sqc2(m: Machine, ins) -> None:
    addr = (m.get32(ins.rs) + ins.imm) & MASK32 & ~0xF
    m.memory.write(addr, struct.pack("<4I", *m.vf[ins.rt]))


def is_supported(ins) -> bool:
    """Whether this interpreter runs the instruction (COP2 by its operation)."""
    if ins.raw >> 26 == 0x12:
        return (ins.mnemonic in ("qmtc2", "ctc2", "qmfc2", "cfc2")
                or ins.mnemonic.split(".")[0] in vu0.SUPPORTED)
    return ins.mnemonic in SUPPORTED_MNEMONICS


# Instructions with no effect an interpreter can observe: ordering barriers, and
# cache maintenance and prefetch, which matter only to real caches.
_NO_EFFECT = frozenset({"nop", "sync", "cache", "pref"})


def _op_syscall(m: Machine, ins) -> None:
    service = _sign32(m.get32(3))
    m.set32(2, SYSCALL_RESULTS.get(service, 0))


def _op_mfc0(m: Machine, ins) -> None:
    m.set(ins.rt, 0)


def _trap(condition):
    def run(m: Machine, ins) -> None:
        a, b = m.get(ins.rs) & MASK64, m.get(ins.rt) & MASK64
        if condition(a, b):
            raise EmulationError("trap (%s) at 0x%08X" % (ins.mnemonic, ins.addr))
    return run


def _s64(v):
    v &= MASK64
    return v - (1 << 64) if v >> 63 else v


# The EE FPU's accumulator: adda/mula/madda/msuba write it, madd/msub read it.
def _fp_accumulate(fn, write_fd: bool):
    def run(m: Machine, ins) -> None:
        a = _fp_from_bits(m.fpr[_fs(ins)])
        b = _fp_from_bits(m.fpr[_ft(ins)])
        value = _f32(fn(_fp_from_bits(m.fp_acc), a, b))
        if write_fd:
            m.fpr[_fd(ins)] = _fp_bits(value)
        else:
            m.fp_acc = _fp_bits(value)
    return run


_HANDLERS.update({
    "syscall": _op_syscall,
    "mfc0": _op_mfc0,
    "ei": lambda m, ins: None,
    "di": lambda m, ins: None,
    "lqc2": _op_lqc2,
    "sqc2": _op_sqc2,
    "break": _trap(lambda a, b: True),
    # A system-control register write; reads (mfc0) see 0 regardless.
    "mtc0": lambda m, ins: None,
    "teq": _trap(lambda a, b: a == b),
    "tne": _trap(lambda a, b: a != b),
    "tge": _trap(lambda a, b: _s64(a) >= _s64(b)),
    "tgeu": _trap(lambda a, b: a >= b),
    "tlt": _trap(lambda a, b: _s64(a) < _s64(b)),
    "tltu": _trap(lambda a, b: a < b),
    "adda.s": _fp_accumulate(lambda acc, a, b: a + b, False),
    "suba.s": _fp_accumulate(lambda acc, a, b: a - b, False),
    "mula.s": _fp_accumulate(lambda acc, a, b: a * b, False),
    "madda.s": _fp_accumulate(lambda acc, a, b: acc + a * b, False),
    "msuba.s": _fp_accumulate(lambda acc, a, b: acc - a * b, False),
    "madd.s": _fp_accumulate(lambda acc, a, b: acc + a * b, True),
    "msub.s": _fp_accumulate(lambda acc, a, b: acc - a * b, True),
})

# Control transfers are handled in _resolve_transfer rather than the dispatch
# table, so they have to be listed explicitly for coverage reporting to be honest.
CONTROL_MNEMONICS = frozenset({
    "j", "b", "jal", "jr", "jalr",
    "beq", "bne", "beql", "bnel",
    "blez", "bgtz", "blezl", "bgtzl",
    "bltz", "bgez", "bltzl", "bgezl", "bltzal", "bgezal",
    "bc1t", "bc1f", "bc1tl", "bc1fl",
    "bc0t", "bc0f", "bc0tl", "bc0fl",
})

SUPPORTED_MNEMONICS = frozenset(_HANDLERS) | CONTROL_MNEMONICS | _NO_EFFECT
