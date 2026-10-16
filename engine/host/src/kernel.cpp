/*
 * The EE kernel, as the game calls it: `syscall` with the service number in $v1
 * and arguments in $a0-$a3 ($t0, $t1 beyond). A negative number is the same
 * service called from interrupt context. Results come back in $v0.
 *
 * What a program needs to start - its stack and heap, cache flushes, the SIF
 * registers, the syscall table it may extend. Threads, semaphores and interrupt
 * handlers are threads.cpp's. Every service is noted the first time it is used,
 * so the report shows what the game asked for.
 */

#include <cstdio>
#include <map>
#include <string>

#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

struct Kernel {
    u32 heap_end = 0;
    u32 stack_base = 0;
    u64 gs_imr = 0;
    std::map<u32, u32> sif_registers;
    // The console's settings, as Get/SetOsdConfigParam pass them: bit 0 S/PDIF,
    // 1-2 screen (0 4:3), 3 video output, 4 non-Japanese language, 5-12 PS1
    // driver settings, 13-15 version, 16-20 language (1 English), 21-31 time
    // zone offset in minutes. A US console in English, at GMT.
    u32 osd_config = (1u << 4) | (1u << 13) | (1u << 16);
};

// The kernel's syscall table, in guest kernel RAM as on the console: SetSyscall
// writes a handler into it, and a program may find it there and read or patch it
// (Burnout 3's start-up searches kernel memory for the handlers it installed, to
// learn where the table is). An entry of zero is a service the host provides.
constexpr u32 kSyscallTable = 0x80001000u;
constexpr u32 kSyscallEntries = 256;

u32 TableEntry(s32 number) {
    return u32(number) < kSyscallEntries
               ? u32(GuestMemory().Read(kSyscallTable + 4 * u32(number), 4)) : 0;
}

Kernel& K() {
    static Kernel kernel;
    return kernel;
}

u32 Arg(const lifted::Context& c, int i) {
    static const int regs[] = {4, 5, 6, 7, 8, 9};     // $a0-$a3, $t0, $t1
    return u32(c.r[regs[i]]);
}

void Return(lifted::Context& c, u64 value) { c.r[2] = u64(s64(s32(u32(value)))); }

struct Service {
    s32 number;
    const char* name;
};

// The documented EE kernel services (engine/data/ee_syscalls.csv names them for
// the analysis; these are the ones the host knows by name).
const Service kServices[] = {
    {0x04, "Exit"}, {0x05, "RFU005"}, {0x06, "LoadExecPS2"}, {0x07, "ExecPS2"},
    {0x0D, "SetVTLBRefillHandler"}, {0x0E, "SetVCommonHandler"},
    {0x10, "AddIntcHandler"}, {0x11, "RemoveIntcHandler"},
    {0x12, "AddDmacHandler"}, {0x13, "RemoveDmacHandler"},
    {0x14, "_EnableIntc"}, {0x15, "_DisableIntc"}, {0x16, "_EnableDmac"},
    {0x17, "_DisableDmac"}, {0x20, "CreateThread"}, {0x21, "DeleteThread"},
    {0x22, "StartThread"}, {0x23, "ExitThread"}, {0x24, "ExitDeleteThread"},
    {0x25, "TerminateThread"}, {0x29, "ChangeThreadPriority"},
    {0x2B, "RotateThreadReadyQueue"}, {0x2F, "GetThreadId"},
    {0x30, "ReferThreadStatus"}, {0x32, "SleepThread"}, {0x33, "WakeupThread"}, {0x34, "iWakeupThread"},
    {0x37, "SuspendThread"}, {0x39, "ResumeThread"}, {0x3C, "SetupThread"},
    {0x3D, "SetupHeap"}, {0x3E, "EndOfHeap"}, {0x40, "CreateSema"},
    {0x41, "DeleteSema"}, {0x42, "SignalSema"}, {0x43, "iSignalSema"},
    {0x44, "WaitSema"}, {0x45, "PollSema"}, {0x47, "ReferSemaStatus"},
    {0x4A, "SetOsdConfigParam"}, {0x4B, "GetOsdConfigParam"}, {0x5A, "kCopy"},
    {0x64, "FlushCache"},
    {0x70, "GsGetIMR"}, {0x71, "GsPutIMR"}, {0x73, "SetVSyncFlag"},
    {0x74, "SetSyscall"}, {0x76, "SifDmaStat"}, {0x77, "SifSetDma"},
    {0x78, "SifSetDChain"}, {0x79, "SifSetReg"}, {0x7A, "SifGetReg"},
    {0x7C, "Deci2Call"}, {0xFC, "SetAlarm"}, {0xFD, "iSetAlarm"},
    {0xFE, "ReleaseAlarm"}, {0xFF, "iReleaseAlarm"},
};

}  // namespace

const char* SyscallName(s32 service) {
    const s32 number = service < 0 ? -service : service;
    for (const Service& s : kServices)
        if (s.number == number) return s.name;
    return "?";
}

void KernelSyscall(lifted::Context& c) {
    const s32 service = s32(u32(c.r[3]));
    const s32 number = service < 0 ? -service : service;
    RunStats().syscalls[service]++;
    char text[160];
    std::snprintf(text, sizeof text, "kernel %s (0x%02X) a0=0x%08X a1=0x%08X a2=0x%08X a3=0x%08X",
                  SyscallName(service), unsigned(number), Arg(c, 0), Arg(c, 1), Arg(c, 2),
                  Arg(c, 3));
    if (Tracing()) std::fprintf(stderr, "[%s]\n", text);
    else Note("syscall " + std::to_string(service), text);
    Kernel& k = K();
    // A service the program installed itself (SetSyscall) runs its own handler,
    // in kernel mode on the console - here, as any other guest function.
    if (const u32 handler = TableEntry(number)) {
        Dispatch(c, handler);
        return;
    }
    u64 result = 0;
    if (ThreadSyscall(c, service, result)) {
        Return(c, result);
        return;
    }
    switch (number) {
        case 0x04:                                   // Exit(status)
            throw GuestExit{s32(Arg(c, 0))};
        case 0x06: case 0x07:                        // LoadExecPS2 / ExecPS2
            throw Unimplemented{std::string("the kernel service ") + SyscallName(service) +
                                ", which starts another program"};
        case 0x3C: {                                 // SetupThread(gp, stack, size, args, root)
            const u32 stack = Arg(c, 1), size = Arg(c, 2);
            // A stack of -1 means "at the top of RAM"; the thread's stack pointer
            // is the top of its stack, less the kernel's argument area.
            const u32 base = stack == 0xFFFFFFFFu ? kRdramSize - size : stack;
            k.stack_base = base;
            Return(c, base + size - 0x2A0);
            return;
        }
        case 0x3D: {                                 // SetupHeap(start, size)
            const u32 start = Arg(c, 0), size = Arg(c, 1);
            k.heap_end = size == 0xFFFFFFFFu ? k.stack_base : start + size;
            Return(c, k.heap_end);
            return;
        }
        case 0x3E:                                   // EndOfHeap()
            Return(c, k.heap_end);
            return;
        case 0x70:                                   // GsGetIMR
            Return(c, k.gs_imr);
            return;
        case 0x71:                                   // GsPutIMR(imr)
            k.gs_imr = Arg(c, 0);
            Return(c, 0);
            return;
        case 0x74:                                   // SetSyscall(service, handler)
            // The SDK patches the kernel's alarms (SetAlarm, iSetAlarm, ReleaseAlarm,
            // iReleaseAlarm, and their timer 3 handler at 0x12C) with code that
            // drives the kernel's own interrupt dispatch. The host keeps those
            // calls: its alarms are the same API, run by its own scheduler.
            if (Arg(c, 0) >= 0xFC && Arg(c, 0) <= 0xFF) {
                std::snprintf(text, sizeof text, "kernel: the program's patch of syscall 0x%02X is not "
                              "installed: the host's alarms stand", unsigned(Arg(c, 0)));
                Note("syscall patch " + std::to_string(Arg(c, 0)), text);
            } else if (Arg(c, 0) < kSyscallEntries) {
                GuestMemory().Write(kSyscallTable + 4 * Arg(c, 0), Arg(c, 1), 4);
            }
            Return(c, 0);
            return;
        case 0x5A: {                                 // kCopy(dest, src, size): from kernel memory
            Memory& m = GuestMemory();
            for (u32 i = 0; i + 4 <= Arg(c, 2); i += 4)
                m.Write(Arg(c, 0) + i, m.Read(Arg(c, 1) + i, 4), 4);
            Return(c, 0);
            return;
        }
        case 0x76:                                   // SifDmaStat(id): < 0 when done
            Return(c, u64(-1));
            return;
        case 0x77: {                                 // SifSetDma(transfers*, count)
            if (Tracing()) {
                // struct SifDmaTransfer { void* src; void* dest; int size; int attr; }
                Memory& m = GuestMemory();
                for (u32 i = 0; i < Arg(c, 1); ++i) {
                    const u32 t = Arg(c, 0) + 16 * i;
                    const u32 src = u32(m.Read(t, 4)), dest = u32(m.Read(t + 4, 4));
                    const u32 size = u32(m.Read(t + 8, 4)), attr = u32(m.Read(t + 12, 4));
                    std::fprintf(stderr, "  [sif dma] src=0x%08X dest=0x%08X size=%u attr=0x%X:",
                                 src, dest, size, attr);
                    for (u32 w = 0; w < size && w < 64; w += 4)
                        std::fprintf(stderr, " %08X", u32(m.Read(src + w, 4)));
                    std::fprintf(stderr, "\n");
                }
            }
            SifDma(Arg(c, 0), Arg(c, 1));
            Return(c, 1);
            return;
        }
        case 0x78:                                   // SifSetDChain
            Return(c, 1);
            return;
        case 0x79:                                   // SifSetReg(reg, value)
            k.sif_registers[Arg(c, 0)] = Arg(c, 1);
            Return(c, 1);
            return;
        case 0x7A:                                   // SifGetReg(reg)
            // Every status bit set: the IOP reports itself up and ready, as the
            // verification interpreter does (engine/ps2decomp/emu.py).
            Return(c, 0xFFFFFFFFu);
            return;
        case 0x4A:                                   // SetOsdConfigParam(config*)
            k.osd_config = u32(GuestMemory().Read(Arg(c, 0), 4));
            Return(c, 0);
            return;
        case 0x4B:                                   // GetOsdConfigParam(config*)
            GuestMemory().Write(Arg(c, 0), k.osd_config, 4);
            Return(c, 0);
            return;
        case 0x7C: {                                 // Deci2Call(function, params*)
            // DECI2 talks to a development tool's debugger. A retail console has
            // none attached, so every request fails - and code that sends its
            // TTY output over DECI2 gives up rather than waiting on a reply that
            // never comes. kputs (0x10) is the kernel's own debug print: the text
            // is the game's, so it goes to the log.
            if (Arg(c, 0) == 0x10) {
                Memory& m = GuestMemory();
                std::string text;
                for (u32 p = u32(m.Read(Arg(c, 1), 4)); text.size() < 1024; ++p) {
                    const char ch = char(m.Read(p, 1));
                    if (!ch) break;
                    text += ch;
                }
                while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
                std::fprintf(stderr, "[guest] %s\n", text.c_str());
                Return(c, 0);
                return;
            }
            Return(c, 0xFFFFFFFFu);
            return;
        }
        default:
            // The rest - handler enables, cache flushes, thread status, priorities -
            // have no effect the program depends on yet: succeed quietly.
            Return(c, 0);
            return;
    }
}

}  // namespace ps2host
