"""Names for functions in a stripped executable, each with the evidence behind it.

Retail PS2 executables carry no symbols, but some functions can be named with
certainty:

* **Kernel stubs.** `addiu $v1, $zero, N; syscall; jr $ra` calls kernel service
  N, and the service numbers are platform ABI (`engine/data/ee_syscalls.csv`).
* **The signature library.** Code already named in another game, and
  recognised here by its signature (`signatures.py`).
* **Diagnostic strings.** SDK and middleware functions identify themselves in
  their error messages - `"sceDbcInitSocket: rpc error"` - so a string that
  opens with a function-shaped name and a colon names the one function that
  prints it (`diagnostic_names`).
* **SIF RPC calls.** Every game talks to the IOP through the SDK's
  `sceSifCallRpc`, which takes eight register arguments, and binds each server
  with `sceSifBindRpc`, passing a server ID (`0x80000001` is file I/O) in `$a1`.
  The shape of their call sites names them (`rpc_names`).
* **Hand-assigned names.** The ones in the game's status table that are not
  `generated::` placeholders.

A game's names live in `config/names.csv` as (address, name, source), so every
name can be traced back to its evidence and a wrong one is easy to find. Nothing
is renamed without a source.
"""

from __future__ import annotations

import bisect
import collections
import csv
import os
import re
import struct

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_SYSCALL = 0x0000000C
_JR_RA = 0x03E00008


def syscall_table(path: str = os.path.join(DATA, "ee_syscalls.csv")) -> dict:
    """{service number: name}; negative numbers are interrupt-context variants."""
    table = {}
    with open(path, encoding="utf-8") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in rows:
            table[int(row["service"], 0)] = row["name"]
    return table


def syscall_stubs(database: dict, image: bytes, base: int) -> dict:
    """{function address: service number} for every kernel stub.

    A stub is `addiu $v1, $zero, N` then `syscall`, followed by `jr $ra` within the
    next two instructions. Nothing else may come before the syscall.
    """
    found = {}
    for func in database["functions"]:
        start, end = func["addr"], func["end"]
        if end - start < 12 or end - start > 24:
            continue
        words = struct.unpack_from("<%dI" % ((end - start) // 4), image, start - base)
        first = words[0]
        if first >> 16 != 0x2403 or words[1] != _SYSCALL:   # addiu $v1, $zero, imm
            continue
        if _JR_RA not in words[2:4]:
            continue
        service = first & 0xFFFF
        found[start] = service - 0x10000 if service & 0x8000 else service
    return found


# "Name: message". The name must look like a function, not a module tag: either a
# lowercase library prefix then a capital (sceDbcInitSocket, lgAudOpen), or three
# or more capitalised words (_LobbyStatbookCreate). Two-word names are refused
# because module tags use that shape too ("ProtoMangle: ...", "DirtyElf: ...").
_SELF_NAMED = re.compile(r"^(_?(?:[a-z]+[A-Z]|[A-Z][a-z0-9]+[A-Z][a-z0-9]+[A-Z])[A-Za-z0-9_]*)\s?:")


def diagnostic_names(database: dict) -> dict:
    """{function address: (name, the string)} from self-identifying messages.

    A name is used only when exactly one function references strings carrying it,
    and that function carries no other such name.
    """
    strings = {entry["addr"]: entry["text"] for entry in database["strings"]}
    spans = sorted((f["addr"], f["end"]) for f in database["functions"])
    starts = [start for start, _end in spans]
    owners = collections.defaultdict(set)      # name -> functions
    evidence = {}
    for xref in database["xrefs"]:
        text = strings.get(xref["to"])
        match = _SELF_NAMED.match(text) if text else None
        if not match:
            continue
        index = bisect.bisect_right(starts, xref["from"]) - 1
        if index < 0 or xref["from"] >= spans[index][1]:
            continue
        owners[match.group(1)].add(spans[index][0])
        evidence.setdefault(match.group(1), text.strip())
    by_function = collections.defaultdict(set)
    for name, functions in owners.items():
        for address in functions:
            by_function[address].add(name)
    named = {}
    for address, names in by_function.items():
        name = next(iter(names))
        if len(names) == 1 and len(owners[name]) == 1:
            named[address] = (name, evidence[name])
    return named


# sceSifCallRpc(client, fno, mode, send, ssize, receive, rsize, end, end_param):
# its call sites set all of $a0-$a3 and $t0-$t3. It must be the target of more
# such calls than any other function, by this factor, and from this many callers.
_RPC_CALL_LEAD = 2
_RPC_CALL_CALLERS = 8
# sceSifBindRpc(client, server ID, mode): the SDK's servers have IDs 0x8000xxxx.
_SDK_SERVERS = range(0x80000000, 0x80010000)
_RPC_BIND_SERVERS = 3
_WINDOW = 12            # instructions searched before a call for its arguments


def _call_sites(database: dict, image: bytes, base: int):
    """(caller, call address, target) for every `jal` inside a known function."""
    for func in database["functions"]:
        start, end = func["addr"], func["end"]
        words = struct.unpack_from("<%dI" % ((end - start) // 4), image, start - base)
        for index, word in enumerate(words):
            if word >> 26 == 3:
                at = start + 4 * index
                yield start, at, (at & 0xF0000000) | ((word & 0x3FFFFFF) << 2)


def _argument_setup(image: bytes, base: int, start: int, at: int):
    """(registers written, register constants) in the straight line code before
    the call at `at`, counting its delay slot.

    Constants are tracked through lui/ori/addiu only; a branch or jump ends the
    straight line, so anything before it is forgotten.
    """
    written, constants = set(), {}
    first = max(start, at - 4 * _WINDOW)
    for address in list(range(first, at, 4)) + [at + 4]:
        word = struct.unpack_from("<I", image, address - base)[0]
        op, rs, rt, imm = word >> 26, (word >> 21) & 31, (word >> 16) & 31, word & 0xFFFF
        if op in (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07) or (
                op == 0 and word & 0x3F in (8, 9)):
            written, constants = set(), {}
            continue
        if op == 0:
            target = (word >> 11) & 31
        elif op in (0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x19,
                    0x20, 0x21, 0x23, 0x24, 0x25, 0x27, 0x37):
            target = rt
        else:
            continue
        written.add(target)
        source = 0 if rs == 0 else constants.get(rs)
        if op == 0x0F:
            value = imm << 16
        elif op == 0x0D and source is not None:
            value = source | imm
        elif op == 0x09 and source is not None:
            value = (source + imm - (0x10000 if imm & 0x8000 else 0)) & 0xFFFFFFFF
        else:
            value = None
        constants[target] = value
    return written, constants


def rpc_names(database: dict, image: bytes, base: int) -> dict:
    """{function address: (name, evidence)} for sceSifCallRpc and sceSifBindRpc."""
    eight = collections.defaultdict(set)        # target -> callers
    servers = collections.defaultdict(set)      # target -> server IDs passed
    for caller, at, target in _call_sites(database, image, base):
        written, constants = _argument_setup(image, base, caller, at)
        if {8, 9, 10, 11} <= written:
            eight[target].add(caller)
        if constants.get(5) in _SDK_SERVERS:
            servers[target].add(constants[5])
    named = {}
    ranked = sorted(eight.items(), key=lambda item: -len(item[1]))
    if not ranked or len(ranked[0][1]) < _RPC_CALL_CALLERS:
        return named
    call, callers = ranked[0]
    runner_up = len(ranked[1][1]) if len(ranked) > 1 else 0
    if len(callers) < _RPC_CALL_LEAD * runner_up:
        return named
    named[call] = ("sceSifCallRpc", "called with eight arguments by %d functions "
                   "(next most: %d)" % (len(callers), runner_up))
    binds = sorted(((len(ids), target) for target, ids in servers.items()
                    if target != call), reverse=True)
    if binds and binds[0][0] >= _RPC_BIND_SERVERS and (
            len(binds) == 1 or binds[1][0] < binds[0][0]):
        count, bind = binds[0]
        named[bind] = ("sceSifBindRpc", "passed %d SDK server IDs (%s...)" % (
            count, ", ".join("0x%08X" % sid for sid in sorted(servers[bind])[:3])))
    return named


# Address-derived names the lifter gives functions it cannot name (lift.py).
_PLACEHOLDER = re.compile(r"^F_[0-9A-F]{8}$")


def status_names(status_path: str) -> dict:
    """Hand-assigned names from a status table: {address: name}."""
    names = {}
    if not os.path.exists(status_path):
        return names
    with open(status_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            if name and not name.startswith("generated::"):
                name = name.split("::")[-1]
                if not _PLACEHOLDER.match(name):
                    names[int(row["address"], 16)] = name
    return names


def load(path: str) -> dict:
    """A game's names file: {address: (name, source)}."""
    names = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                names[int(row["address"], 16)] = (row["name"], row["source"])
    return names


def save(path: str, names: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("address,name,source\n")
        for address in sorted(names):
            name, source = names[address]
            handle.write("0x%08X,%s,%s\n" % (address, name, source))
