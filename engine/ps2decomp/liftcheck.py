"""Compile lifted functions and run them against the original machine code.

The lifted C++ is built into a shared library with a small host: memory reads and
writes go to the interpreter's memory, and a call to anything goes to the
interpreter with the whole register context (Machine.run_to_return). So each
lifted function is tested on its own - its callees are the original code - and
what is compared is everything it can affect: all registers, HI/LO, the FPU, and
every byte of memory written.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess

# The runtime rounds floats as the console does (toward zero, denormals
# flushed: eefpu.cpp, vu0.cpp); the interpreter it is compared against rounds
# to nearest. Verification checks the translation, not the FPU model: the
# compiled code runs in the interpreter's mode. Set before any harness loads.
os.environ["PS2RT_EE_ROUND_NEAREST"] = "1"

from . import cppcheck, layout, vu0
from .emu import SYSCALL_RESULTS, EmulationError, ExecutionLimit, RETURN_SENTINEL

MASK64 = (1 << 64) - 1


class Context(ctypes.Structure):
    """Mirror of lifted::Context in ps2rt/lifted.h."""
    _fields_ = [
        ("r", ctypes.c_uint64 * 32),
        ("rh", ctypes.c_uint64 * 32),
        ("hi", ctypes.c_uint64), ("lo", ctypes.c_uint64),
        ("hi1", ctypes.c_uint64), ("lo1", ctypes.c_uint64),
        ("f", ctypes.c_uint32 * 32),
        ("acc", ctypes.c_uint32),
        ("fcc", ctypes.c_uint32),
        ("budget", ctypes.c_int64),
        ("vf", (ctypes.c_uint32 * 4) * 32),
        ("vc", ctypes.c_uint32 * 32),
        ("vacc", ctypes.c_uint32 * 4),
        ("sa", ctypes.c_uint32),
    ]


READ = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.POINTER(ctypes.c_uint32))
WRITE = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32,
                         ctypes.POINTER(ctypes.c_uint32))
CALL = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(Context),
                        ctypes.c_uint32, ctypes.c_uint32)

LOOP_BUDGET = 10_000_000

DEFINITION = re.compile(r"^void (F_([0-9A-F]{8}))\(lifted::Context& c\) \{$", re.M)


def harness_source(names) -> str:
    """The host: memory and calls forwarded to Python, and one entry point."""
    syscalls = " ".join("case %d: value = 0x%Xu; break;" % (service, result & 0xFFFFFFFF)
                        for service, result in sorted(SYSCALL_RESULTS.items()))
    lines = [
        '#pragma optimize("", off)',
        '#include "ps2rt/lifted.h"',
        "",
        "typedef u64 (*ReadFn)(u32 address, u32 size, u32 is_signed, u32* fault);",
        "typedef void (*WriteFn)(u32 address, u64 value, u32 size, u32* fault);",
        "typedef u32 (*CallFn)(u32 address, lifted::Context* context, u32 return_address,",
        "                      u32 entry_sp);",
        "static ReadFn g_read;",
        "static WriteFn g_write;",
        "static CallFn g_call;",
        "",
        "namespace generated {",
        "static u64 Read(u32 a, u32 size, u32 is_signed) {",
        "    u32 fault = 0;",
        "    u64 value = g_read(a, size, is_signed, &fault);",
        "    if (fault) throw lifted::Abort{a, 4};",
        "    return value;",
        "}",
        "static void Write(u32 a, u64 v, u32 size) {",
        "    u32 fault = 0;",
        "    g_write(a, v, size, &fault);",
        "    if (fault) throw lifted::Abort{a, 4};",
        "}",
    ]
    lines += cppcheck.port_memory("Read(a, size, is_signed)", "Write(a, v, size)")
    lines += [
        "}",
        "",
        "namespace lifted {",
        "void Call(Context& c, u32 target) {",
        "    if (g_call(target, &c, (u32)c.r[31], (u32)c.r[29])) throw Abort{target, 3};",
        "}",
        "void TailCall(Context& c, u32 target, u32 return_address, u32 entry_sp) {",
        "    if (g_call(target, &c, return_address, entry_sp)) throw Abort{target, 3};",
        "}",
        "void Vcallms(Context&, u32 address) { throw Abort{address, 5}; }",
        "void Syscall(Context& c) {",
        "    u32 value = 0;",
        "    switch ((s32)(u32)c.r[3]) { %s default: break; }" % syscalls,
        "    c.r[2] = Sx32(value);",
        "}",
        "}",
        "",
    ]
    lines += ["void %s(lifted::Context& c);" % name for name in names]
    lines += [
        "",
        'extern "C" __declspec(dllexport) void lift_set_host(ReadFn r, WriteFn w, CallFn c) {',
        "    g_read = r; g_write = w; g_call = c;",
        "}",
        "",
        "/* 1 ran, 0 unknown address, 2 abandoned (*reason says why). */",
        'extern "C" __declspec(dllexport) int lift_run(u32 address, lifted::Context* c, u32* reason) {',
        "    try {",
        "        switch (address) {",
    ]
    for name in names:
        lines.append("        case 0x%su: %s(*c); return 1;" % (name[2:], name))
    lines += [
        "        }",
        "    } catch (const lifted::Abort& abort) {",
        "        *reason = abort.reason;",
        "        return 2;",
        "    }",
        "    return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


# Compiler resource failures - out of heap, a DLL it could not load - say nothing
# about the code; the build is retried with fewer compilers running at once.
RESOURCE_ERRORS = ("C1060", "C1356", "C1076", "C3859")


def build(sources, work_dir: str, parallel: int = 8) -> str:
    """Compile lifted sources with the host into a DLL; returns its path.

    `parallel` caps how many compilers run at once: each can need gigabytes at /O2
    on these files, and one per core exhausted memory on a 16-core machine.
    """
    try:
        return _build(sources, work_dir, parallel)
    except cppcheck.CompileError as exc:
        if parallel > 1 and any(code in str(exc) for code in RESOURCE_ERRORS):
            return _build(sources, work_dir, 2)
        raise


def _build(sources, work_dir: str, parallel: int) -> str:
    vcvars = cppcheck.find_vcvars()
    # Absolute: the build script changes into it, and a relative path (from a
    # relative --game) would then name a directory inside itself.
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    names = []
    for path in sources:
        with open(path, encoding="utf-8") as handle:
            names += [match.group(1) for match in DEFINITION.finditer(handle.read())]
    harness = os.path.join(work_dir, "lift_harness.cpp")
    with open(harness, "w", encoding="utf-8") as handle:
        handle.write(harness_source(names))
    runtime = os.path.join(cppcheck.RUNTIME, "src")
    all_sources = [os.path.abspath(p) for p in sources] + [
        harness, os.path.join(runtime, "eefpu.cpp"), os.path.join(runtime, "lifted_support.cpp"),
        os.path.join(runtime, "vu0.cpp")]
    cppcheck._BUILDS += 1
    dll = os.path.join(work_dir, "lifted_%d_%d.dll" % (os.getpid(), cppcheck._BUILDS))
    script = os.path.join(work_dir, "lift_build.bat")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write("@echo off\n")
        handle.write('call "%s" >nul\n' % cppcheck.find_vcvars())
        handle.write('cd /d "%s"\n' % work_dir)
        # /O2 for the same reason as the derived code's check; /EHsc for Abort;
        # /bigobj because a lifted file has far more sections than the default allows.
        handle.write('cl /nologo /std:c++17 /O2 /EHsc /bigobj /MP%d /W3 /LD /I"%s" %s /Fe"%s" '
                     '>lift_build.log 2>&1\n'
                     % (parallel, os.path.join(cppcheck.RUNTIME, "include"),
                        " ".join('"%s"' % p for p in all_sources), dll))
        handle.write("exit /b %errorlevel%\n")
    result = subprocess.run(["cmd", "/c", script], capture_output=True, text=True)
    if result.returncode != 0:
        with open(os.path.join(work_dir, "lift_build.log"), encoding="utf-8",
                  errors="replace") as handle:
            log = handle.read()
        errors = [line for line in log.splitlines() if " error " in line]
        raise cppcheck.CompileError("\n".join(errors) or log[-2000:])
    return dll


def machine_state(machine) -> tuple:
    """Everything a function can change in the register file."""
    return (tuple(machine.regs), tuple(machine.regs_hi), machine.hi, machine.lo,
            machine.hi1, machine.lo1, tuple(f & 0xFFFFFFFF for f in machine.fpr),
            machine.fp_acc & 0xFFFFFFFF, int(bool(machine.fp_cond)),
            tuple(tuple(v) for v in machine.vf), tuple(machine.vc), tuple(machine.vacc),
            machine.sa)


def context_state(context: Context) -> tuple:
    regs = list(context.r)
    regs[0] = 0
    return (tuple(regs), tuple(context.rh), context.hi, context.lo, context.hi1,
            context.lo1, tuple(context.f), context.acc, int(context.fcc != 0),
            tuple(tuple(v) for v in context.vf), tuple(context.vc), tuple(context.vacc),
            context.sa)


def load_machine(machine, context: Context) -> None:
    machine.regs = [v & MASK64 for v in context.r]
    machine.regs[0] = 0
    machine.regs_hi = list(context.rh)
    machine.hi, machine.lo, machine.hi1, machine.lo1 = (
        context.hi, context.lo, context.hi1, context.lo1)
    machine.fpr = list(context.f)
    machine.fp_acc = context.acc
    machine.fp_cond = context.fcc != 0
    machine.vf = [list(v) for v in context.vf]
    machine.vc = list(context.vc)
    machine.vacc = list(context.vacc)
    machine.sa = context.sa


def store_context(context: Context, machine) -> None:
    for i in range(32):
        context.r[i] = machine.regs[i] & MASK64
        context.rh[i] = machine.regs_hi[i] & MASK64
        context.f[i] = machine.fpr[i] & 0xFFFFFFFF
    context.hi, context.lo = machine.hi & MASK64, machine.lo & MASK64
    context.hi1, context.lo1 = machine.hi1 & MASK64, machine.lo1 & MASK64
    context.acc = machine.fp_acc & 0xFFFFFFFF
    context.fcc = 1 if machine.fp_cond else 0
    for i in range(32):
        for lane in range(4):
            context.vf[i][lane] = machine.vf[i][lane] & 0xFFFFFFFF
        context.vc[i] = machine.vc[i] & 0xFFFFFFFF
    for lane in range(4):
        context.vacc[lane] = machine.vacc[lane] & 0xFFFFFFFF
    context.sa = machine.sa


def entry_context(args, stack_top=None) -> Context:
    """The state Machine.call starts a function in, as a context."""
    context = Context()
    for i, value in enumerate(args):
        if i < 8:
            context.r[(4, 5, 6, 7, 8, 9, 10, 11)[i]] = value & MASK64
        else:
            context.f[12 + i - 8] = value & 0xFFFFFFFF
    context.r[28] = layout.CURRENT.gp
    context.r[29] = layout.CURRENT.stack_top if stack_top is None else stack_top
    context.r[31] = RETURN_SENTINEL
    context.budget = LOOP_BUDGET
    context.vf[0][3] = vu0.ONE
    context.vc[vu0.REG_R] = vu0.ONE
    return context


class LiftedFunctions:
    """A built library of lifted functions, hosted on an interpreter's memory."""

    REASONS = {1: "trap", 2: "loop budget", 3: "callee failed", 4: "memory fault",
               5: "COP2 operation"}

    def __init__(self, dll_path: str, machine, budget: int = 100_000) -> None:
        self.machine = machine
        self.budget = budget
        self._read = READ(self._on_read)
        self._write = WRITE(self._on_write)
        self._call = CALL(self._on_call)
        self.library = ctypes.CDLL(dll_path)
        self.library.lift_set_host(self._read, self._write, self._call)
        self.library.lift_run.restype = ctypes.c_int
        self.library.lift_run.argtypes = [ctypes.c_uint32, ctypes.POINTER(Context),
                                          ctypes.POINTER(ctypes.c_uint32)]

    def _on_read(self, address, size, signed, fault):
        try:
            return self.machine.memory.read_int(address, size, bool(signed)) & MASK64
        except EmulationError:
            fault[0] = 1
            return 0

    def _on_write(self, address, value, size, fault):
        try:
            self.machine.memory.write_int(address, value & ((1 << (8 * size)) - 1), size)
        except EmulationError:
            fault[0] = 1

    def _on_call(self, target, context, return_address, entry_sp):
        try:
            load_machine(self.machine, context.contents)
            self.machine.run_to_return(target, self.budget, return_address, entry_sp)
            store_context(context.contents, self.machine)
            return 0
        except (EmulationError, ExecutionLimit) as exc:
            self.callee_error = "0x%08X: %s" % (target, exc)
            return 1

    def run(self, address: int, context: Context):
        """Run one lifted function; returns None if it ran, else why it did not."""
        reason = ctypes.c_uint32(0)
        self.callee_error = None
        status = self.library.lift_run(address, ctypes.byref(context), ctypes.byref(reason))
        if status == 1:
            return None
        if status == 0:
            return "not in the library"
        why = self.REASONS.get(reason.value, "abandoned")
        if reason.value == 3 and self.callee_error:
            why += " (%s)" % self.callee_error
        return why
