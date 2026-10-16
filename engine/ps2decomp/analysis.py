"""Code discovery for a stripped PS2 executable.

The retail Burnout 3 binary is one flat PROGBITS blob: no symbol table, and no
section headers marking where code stops and read-only data begins.  Everything
about the layout therefore has to be recovered by analysis.

The approach here is deliberately conservative, in the order that produces the
fewest false positives:

1. **Call-target harvesting.**  Every `jal` in the image names a function entry
   directly.  A `jal` that lands inside the image is near-proof of a function,
   because the compiler only emits it for real call sites.
2. **Recursive traversal.**  Starting from the ELF entry point and every
   harvested call target, walk the control-flow graph, following branches and
   jumps, and stop at returns.  This marks bytes that are *provably* reachable
   code rather than bytes that merely decode as code.
3. **Prologue scanning.**  In the gaps, look for the conventional frame setup
   (`addiu $sp, $sp, -N`) to catch functions only reached through function
   pointers - vtables, jump tables, and callback tables, which a 2004 C++ game
   uses heavily.
4. **Data classification.**  Whatever is left is scored: runs that decode poorly,
   or that look like pointers, floats, or strings, are recorded as data.

Each stage records *why* an address was classified, so the results stay
auditable instead of being an opaque guess.
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import struct

from .r5900 import DISASM, Flags, GPR

# Confidence levels attached to each discovered function.
CONF_ENTRY = "entry"          # the ELF entry point
CONF_CALLED = "called"        # at least one direct jal targets it
CONF_PROLOGUE = "prologue"    # found by frame-setup pattern
CONF_POINTER = "pointer"      # referenced by a plausible pointer word


@dataclasses.dataclass
class Function:
    """A discovered function: a half-open address range plus provenance."""

    start: int
    end: int = 0
    confidence: str = CONF_PROLOGUE
    callers: set = dataclasses.field(default_factory=set)
    calls: set = dataclasses.field(default_factory=set)
    instruction_count: int = 0
    has_return: bool = False
    uses_vu: bool = False
    name: str = ""

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)

    @property
    def label(self) -> str:
        return self.name or ("func_%08X" % self.start)


@dataclasses.dataclass
class Region:
    """A classified span of the image."""

    start: int
    end: int
    kind: str  # "code" | "data" | "unknown"

    @property
    def size(self) -> int:
        return self.end - self.start


class ImageAnalyzer:
    """Recovers functions and code/data boundaries from a flat loaded image."""

    def __init__(self, data: bytes, base_addr: int, entry_point: int = 0) -> None:
        self.data = data
        self.base = base_addr
        self.end = base_addr + len(data)
        self.entry_point = entry_point
        self.words = struct.unpack("<%dI" % (len(data) // 4), data[: (len(data) // 4) * 4])
        self.functions: dict = {}
        self.code_bytes = bytearray(len(data))  # 1 where traversal proved code
        self.call_sites: dict = collections.defaultdict(set)
        self.case_targets: set = set()    # labels reached through switch jump tables
        self.stats = collections.Counter()

    # -- address helpers ----------------------------------------------

    def contains(self, addr: int) -> bool:
        return self.base <= addr < self.end

    def word_at(self, addr: int) -> int:
        if not self.contains(addr) or (addr - self.base) % 4:
            return 0
        return self.words[(addr - self.base) // 4]

    def _index(self, addr: int) -> int:
        return addr - self.base

    # -- stage 1: harvest direct call targets --------------------------

    def harvest_call_targets(self) -> int:
        """Scan every aligned word for `jal` and record its target.

        This deliberately scans the whole image rather than only known code: a
        `jal` encoding appearing inside genuine data is rare, and the traversal
        stage discards targets that do not hold sane code.
        """
        found = 0
        for i, word in enumerate(self.words):
            if (word >> 26) != 0x03:  # jal
                continue
            addr = self.base + i * 4
            target = ((addr + 4) & 0xF0000000) | ((word & 0x03FFFFFF) << 2)
            if not self.contains(target):
                continue
            self.call_sites[target].add(addr)
            found += 1
        return found

    # -- stage 2: recursive control-flow traversal ---------------------

    def _looks_like_code(self, addr: int, probe: int = 8, called: bool = False) -> bool:
        """Cheap sanity check that `addr` begins a decodable instruction run.

        Decoding successfully is not enough on its own: a zero word decodes as a
        perfectly valid `nop`, so a run of zeroes in read-only data passes any
        test that only asks "does this decode?".  Real code therefore also has to
        carry some actual work - at least two non-`nop` instructions, and not be
        mostly padding.
        """
        valid = 0
        meaningful = 0
        for k in range(probe):
            a = addr + k * 4
            if not self.contains(a):
                break
            ins = DISASM.decode(self.word_at(a), a)
            if not ins.is_valid:
                return False
            valid += 1
            if not ins.is_nop:
                meaningful += 1
            if ins.ends_block:
                # The delay slot executes too, and is often the whole body: a getter
                # is `jr $ra; lw $v0, 8($a0)`. Not counting it rejected 424 call
                # targets - one of them called from 866 sites - as "not code".
                # An empty delay slot counts for nothing either way: it is neither
                # work nor padding, and counting it as padding tipped 37 real
                # entries over the padding ratio.
                slot = a + 4
                if ins.has_delay_slot and self.contains(slot):
                    slot_ins = DISASM.decode(self.word_at(slot), slot)
                    if not slot_ins.is_valid:
                        return False
                    if not slot_ins.is_nop:
                        valid += 1
                        meaningful += 1
                break
        if called and valid == 1 and meaningful == 1:
            # A `jal` target that is just `jr $ra; nop` or `j elsewhere; nop` is an
            # empty function or a thunk. Data that happens to decode that way is
            # not what a call instruction points at.
            first = DISASM.decode(self.word_at(addr), addr)
            if first.is_return or first.mnemonic in ("j", "b"):
                return True
        if valid < 2 or meaningful < 2:
            return False
        # Padding-dominated windows are data, not a function body.
        return meaningful * 2 > valid

    def traverse(self, start: int, func: Function) -> set:
        """Walk one function's control-flow graph, marking reachable bytes.

        Returns the set of addresses proven to belong to this function, which the
        caller turns into an extent.  The traversal itself must not decide the
        extent: a tail jump or a shared epilogue can land far away, and treating
        the furthest address reached as the end merges everything in between into
        one enormous bogus function.
        """
        queue = collections.deque([start])
        seen = set()
        max_addr = start
        while queue:
            addr = queue.popleft()
            while True:
                if not self.contains(addr) or addr in seen:
                    break
                seen.add(addr)
                idx = self._index(addr)
                ins = DISASM.decode(self.words[idx // 4], addr)
                if not ins.is_valid:
                    break
                self.code_bytes[idx : idx + 4] = b"\x01\x01\x01\x01"
                func.instruction_count += 1
                if addr > max_addr:
                    max_addr = addr
                if ins.mnemonic.startswith("v") or ins.mnemonic in ("qmfc2", "qmtc2"):
                    func.uses_vu = True

                # The delay slot always executes, so claim it before branching.
                if ins.has_delay_slot:
                    slot = addr + 4
                    if self.contains(slot):
                        self.code_bytes[self._index(slot) : self._index(slot) + 4] = b"\x01\x01\x01\x01"
                        seen.add(slot)
                        if slot > max_addr:
                            max_addr = slot

                if ins.is_call:
                    if ins.target >= 0 and self.contains(ins.target):
                        func.calls.add(ins.target)
                elif ins.target >= 0 and self.contains(ins.target):
                    # Intra-function branch or tail jump: queue the target.
                    queue.append(ins.target)

                if ins.mnemonic == "jr" and ins.rs != 31:
                    # A switch: its cases are this function's own code.
                    targets = self.jump_table(addr, ins.rs)
                    if targets:
                        queue.extend(targets)
                        self.case_targets.update(targets)
                        self.stats["jump_tables"] += 1

                if ins.is_return:
                    func.has_return = True
                    break
                if ins.ends_block:
                    break

                addr += 8 if ins.has_delay_slot else 4
        return seen

    # How far back from a `jr` its table and bound are looked for.
    JUMP_TABLE_WINDOW = 32
    MAX_CASES = 1024

    def jump_table(self, jr_addr: int, register: int):
        """The targets of the switch that ends in `jr register`, or None.

        GCC's switch on the R5900 bounds the index, then loads the case's
        address from a table and jumps to it:

            sltiu $at, $i, N;  beq $at, $zero, default; ...
            lui $t, %hi(T);  sll $i, $i, 2;  addiu $t, $t, %lo(T)
            addu $i, $i, $t;  lw $r, 0($i);  jr $r

        Walking back from the `jr` finds the load, the sum, the table's two
        halves and the bound; the table's N words are the cases. Without the
        bound the table's length is unknown, and nothing is claimed.
        """
        load = index = low = table = bound = None
        register_of_sum = register
        addr = jr_addr - 4
        for _ in range(self.JUMP_TABLE_WINDOW):
            if not self.contains(addr):
                break
            ins = DISASM.decode(self.word_at(addr), addr)
            if not ins.is_valid or ins.is_call:
                break
            m = ins.mnemonic
            if load is None:
                if m == "lw" and ins.rt == register_of_sum:
                    load = ins
                    register_of_sum = ins.rs
            elif table is None:
                if index is None and m in ("addu", "daddu") and ins.rd == register_of_sum:
                    index = (ins.rs, ins.rt)
                elif index and low is None and m == "addiu" and ins.rt in index:
                    # The lui is the addiu's source: GCC adds in place
                    # (`addiu $t, $t, lo`), CodeWarrior into another register
                    # (`lui $v0, hi; addiu $v1, $v0, lo; sll $v0, $i, 2`).
                    low = (ins.rs, ins.imm)
                elif m == "lui" and ((low and ins.rt == low[0]) or (index and not low and ins.rt in index)):
                    # The table's low half is in the addiu, or else in the load's
                    # own offset: `lui $t, hi; addu $t, $t, $i; lw $r, lo($t)`.
                    table = ((ins.imm << 16) + (low[1] if low else 0) + load.imm) & 0xFFFFFFFF
            elif m == "sltiu":
                bound = ins.imm
                break
            addr -= 4
        if table is None or not bound or bound > self.MAX_CASES:
            return None
        targets = []
        for i in range(bound):
            target = self.word_at(table + 4 * i)
            if not self.contains(target) or target % 4:
                return None             # not a table of code addresses after all
            targets.append(target)
        return targets

    # A function's pieces outside its range add up to no more than this; more is
    # a closure that leaked into other code. Burnout 3's largest, 0x00186850,
    # has 54 KB of them (a state machine's cases).
    MAX_CHUNK_BYTES = 0x10000

    def chunks(self, start: int, end: int, entries) -> list:
        """The function's out-of-line pieces: [(start, end)] outside [start, end).

        The compiler places a function's cold blocks after the hot code, among
        other functions' cold blocks, and reaches them by branch. Following the
        function's branches - conditional, `b` and switch tables, but not `j`,
        which is how it tail-calls, and stopping at the entries of other
        functions - gives every block that is its own. Those outside its range
        are its chunks, each ending in a transfer.
        """
        seen = self.closure(start, entries)
        if seen is None:
            return []                           # not code after all: claim nothing
        runs = []
        for a in sorted(x for x in seen if not (start <= x < end)):
            if runs and a == runs[-1][1]:
                runs[-1][1] = a + 4
            else:
                runs.append([a, a + 4])
        if sum(e - s for s, e in runs) > self.MAX_CHUNK_BYTES:
            return []
        return [(s, e) for s, e in runs]

    def closure(self, start: int, entries):
        """Every instruction a function reaches by its own branches (as
        `chunks` follows them), or None when that runs into what is not code."""
        seen = set()
        queue = [start]
        while queue:
            addr = queue.pop()
            while self.contains(addr) and addr not in seen:
                if addr != start and addr in entries:
                    break
                ins = DISASM.decode(self.word_at(addr), addr)
                if not ins.is_valid:
                    return None
                seen.add(addr)
                if ins.has_delay_slot:
                    seen.add(addr + 4)
                    if not ins.is_call and ins.mnemonic not in ("jr", "jalr", "j") and ins.target >= 0:
                        queue.append(ins.target)
                    if ins.mnemonic == "jr" and ins.rs != 31:
                        queue.extend(self.jump_table(addr, ins.rs) or [])
                if ins.is_return or ins.ends_block:
                    break
                addr += 8 if ins.has_delay_slot else 4
        return seen

    # -- stage 2b: pointer-table targets -------------------------------

    def _is_prologue(self, addr: int) -> bool:
        """True when `addr` holds `addiu $sp, $sp, -N`, or the large frame's
        `lui $at, 0xFFFF; ori $at, $at, lo; addu $sp, $sp, $at` (CodeWarrior's,
        past 32 KB: Mister Mosquito's task entries)."""
        word = self.word_at(addr)
        if (word & self._PROLOGUE_MASK) == self._PROLOGUE_MATCH and bool(word & 0x8000):
            return True
        return self._is_large_prologue(addr)

    def _is_large_prologue(self, addr: int) -> bool:
        if not (self.contains(addr) and self.contains(addr + 8)):
            return False
        return (self.word_at(addr) == 0x3C01FFFF and (self.word_at(addr + 4) >> 16) == 0x3421
                and self.word_at(addr + 8) in (0x03A1E821, 0x03A1E82D))

    def _follows_terminator(self, addr: int) -> bool:
        """True when the instruction two slots back ends a function.

        A function entry is normally preceded by the previous function's final
        delay slot, so the terminating `jr`/`j` sits two words back.
        """
        # Alignment padding (a few `nop`s) may come between: Burnout 3's
        # 0x002A6B80, a leaf reached only through a pointer, follows two, and was
        # missed - its switch's cases taken for functions instead.
        for padding in range(0, self.MAX_PADDING_WORDS + 1):
            terminator = addr - 8 - 4 * padding
            if not self.contains(terminator):
                return False
            if padding and self.word_at(addr - 4 * padding) != 0:
                return False                    # not padding: something else is there
            if DISASM.decode(self.word_at(terminator), terminator).ends_block:
                return True
        return False

    def harvest_pointer_targets(self) -> list:
        """Find function entries referenced only by pointer tables.

        Virtual calls and callback tables never produce a `jal`, so those targets
        are invisible to call-target harvesting.  Every aligned word that points
        at a plausible entry inside the image is considered, then filtered hard -
        a word has to land on a stack prologue or just past another function's
        terminator to be accepted, which is what keeps ordinary data from
        registering as code.
        """
        candidates = set()
        for word in self.words:
            if not self.contains(word) or word % 4:
                continue
            candidates.add(word)

        accepted = []
        for addr in sorted(candidates):
            if addr in self.functions:
                continue
            if not (self._is_prologue(addr) or self._follows_terminator(addr)):
                continue
            if not self._looks_like_code(addr):
                continue
            accepted.append(addr)
        return accepted

    # Alignment padding between functions is a short run of `nop`; bridging that
    # much keeps a function whole without letting a real gap merge two of them.
    MAX_PADDING_WORDS = 4

    def extent_from(self, start: int, visited: set) -> int:
        """Turn a traversal's visited set into a contiguous end address.

        The body of a function is the unbroken run of its own instructions. A gap
        means the run has ended - the next bytes belong to something else - except
        when the gap is only `nop` padding inserted for alignment.
        """
        addr = start
        end = start
        while self.contains(addr):
            if addr in visited:
                addr += 4
                end = addr
                continue
            # Look past a short run of nops for more of this function.
            probe = addr
            padding = 0
            while (
                self.contains(probe)
                and padding < self.MAX_PADDING_WORDS
                and self.word_at(probe) == 0
                and probe not in visited
            ):
                probe += 4
                padding += 1
            if padding and self.contains(probe) and probe in visited:
                addr = probe
                continue
            break
        return max(end, start + 4)

    # -- stage 3: prologue scan ----------------------------------------

    _PROLOGUE_MASK = 0xFFFF0000
    # addiu $sp, $sp, -N  ->  op=0x09 rs=29 rt=29, negative immediate
    _PROLOGUE_MATCH = (0x09 << 26) | (29 << 21) | (29 << 16)

    def scan_prologues(self) -> list:
        """Find `addiu $sp, $sp, -N` in spans not already proven to be code."""
        candidates = []
        # Skip only what lies inside a known function. Bytes merely *reached* by
        # another function's traversal - the target of a tail jump or a shared
        # epilogue - can still begin a function of their own; skipping every
        # visited byte lost 37 functions, one of them already verified, when
        # better call-target discovery made more traversals reach them.
        extents = sorted((f.start, f.end) for f in self.functions.values())
        starts = [start for start, _end in extents]

        def inside_function(addr):
            index = bisect.bisect_right(starts, addr) - 1
            return index >= 0 and extents[index][0] <= addr < extents[index][1]

        for i, word in enumerate(self.words):
            if word == 0x3C01FFFF:              # lui $at, 0xFFFF: perhaps a large frame
                if not self._is_large_prologue(self.base + i * 4):
                    continue
            elif (word & self._PROLOGUE_MASK) != self._PROLOGUE_MATCH:
                continue
            elif not (word & 0x8000):  # immediate must be negative
                continue
            addr = self.base + i * 4
            if inside_function(addr):
                continue
            if not self._looks_like_code(addr):
                continue
            candidates.append(addr)
        return candidates

    # -- driver ---------------------------------------------------------

    def run(self) -> None:
        self.stats["jal_sites"] = self.harvest_call_targets()
        self.stats["call_targets"] = len(self.call_sites)

        # Seed with the entry point and every direct call target.
        seeds = []
        if self.contains(self.entry_point):
            seeds.append((self.entry_point, CONF_ENTRY))
        for target in sorted(self.call_sites):
            seeds.append((target, CONF_CALLED))

        for addr, confidence in seeds:
            if addr in self.functions:
                continue
            if confidence != CONF_ENTRY and not self._looks_like_code(addr, called=True):
                self.stats["rejected_call_targets"] += 1
                continue
            func = Function(start=addr, confidence=confidence)
            func.callers = set(self.call_sites.get(addr, ()))
            self.functions[addr] = func
            visited = self.traverse(addr, func)
            func.end = self.extent_from(addr, visited)
            func.instruction_count = sum(
                1 for a in visited if func.start <= a < func.end
            )

        # Functions reached only through pointer tables emit no jal at all, so
        # harvest plausible pointer targets before falling back to a blind scan.
        for addr in self.harvest_pointer_targets():
            if addr in self.functions:
                continue
            func = Function(start=addr, confidence=CONF_POINTER)
            visited = self.traverse(addr, func)
            # A pointer-table entry is a real function - a virtual method or a
            # callback - so it has to return. Without a `jr $ra` the candidate is
            # almost always data that merely decodes: tables of small integers
            # read as instructions on $zero, or pointer arrays pointing at
            # themselves. Requiring a return discards those without needing a
            # hand-maintained list of data ranges.
            # One that opens a stack frame is code all the same: a task's entry
            # never returns (it kills itself, or loops forever).
            if not func.has_return and not self._is_prologue(addr):
                self.stats["pointer_rejected_no_return"] += 1
                continue
            self.functions[addr] = func
            func.end = self.extent_from(addr, visited)
            func.instruction_count = sum(
                1 for a in visited if func.start <= a < func.end
            )
            self.stats["pointer_found"] += 1

        # A final pass catches anything still missing.
        for addr in self.scan_prologues():
            if addr in self.functions:
                continue
            func = Function(start=addr, confidence=CONF_PROLOGUE)
            self.functions[addr] = func
            visited = self.traverse(addr, func)
            func.end = self.extent_from(addr, visited)
            func.instruction_count = sum(
                1 for a in visited if func.start <= a < func.end
            )
            self.stats["prologue_found"] += 1

        # A switch's cases are labels of the function that jumps to them. Their
        # table's words point at code just past a `jr` or `b`, which is what the
        # pointer harvest looks for; 2,258 of Burnout 3's 2,986 pointer-found
        # "functions" were cases. Called ones stay: a `jal` is evidence enough.
        for addr in sorted(self.case_targets):
            func = self.functions.get(addr)
            if func is not None and func.confidence in (CONF_POINTER, CONF_PROLOGUE) and not func.callers:
                del self.functions[addr]
                self.stats["switch_cases_not_functions"] += 1

        self._tighten_bounds()
        self._compute_stats()

    def _tighten_bounds(self) -> None:
        """Clamp each function to end no later than the next function start."""
        starts = sorted(self.functions)
        for i, addr in enumerate(starts):
            func = self.functions[addr]
            limit = starts[i + 1] if i + 1 < len(starts) else self.end
            # Clamp only. The extent already reflects the function's own
            # contiguous code, so growing it here would reintroduce the merging
            # this pass exists to prevent.
            if func.end > limit:
                func.end = limit
            if func.end <= func.start:
                func.end = min(func.start + 4, limit)

    def _compute_stats(self) -> None:
        self.stats["functions"] = len(self.functions)
        self.stats["code_bytes"] = sum(self.code_bytes)
        self.stats["image_bytes"] = len(self.data)
        self.stats["with_return"] = sum(1 for f in self.functions.values() if f.has_return)
        self.stats["uses_vu"] = sum(1 for f in self.functions.values() if f.uses_vu)
        for f in self.functions.values():
            self.stats["conf_" + f.confidence] += 1

    # -- reporting -------------------------------------------------------

    def regions(self) -> list:
        """Collapse the code bitmap into contiguous code/unknown regions."""
        out = []
        start = 0
        current = self.code_bytes[0] if self.code_bytes else 0
        for i in range(1, len(self.code_bytes)):
            if self.code_bytes[i] != current:
                out.append(Region(self.base + start, self.base + i, "code" if current else "unknown"))
                start = i
                current = self.code_bytes[i]
        out.append(
            Region(self.base + start, self.base + len(self.code_bytes), "code" if current else "unknown")
        )
        return out

    def coverage(self) -> float:
        total = len(self.data)
        return (sum(self.code_bytes) / total * 100.0) if total else 0.0
