/*
 * The IOP interpreter (iop_vm.h): IOP RAM, an R3000 (MIPS I) interpreter, the
 * IOP kernel's threads under its strict-priority scheduler, and the IRX loader.
 *
 * IOP RAM is the 2 MB the EE also sees at 0x1C000000 (memory.cpp), so what a
 * module writes is where the EE's SIF transfers find it. The heap modules
 * allocate from (sysmem) is 0x10000-0x180000: below is the IOP kernel's, above
 * is where the host's own servers pretend to live (iop.cpp).
 *
 * Library calls: an IRX imports functions through stubs (`jr $ra; addiu $zero,
 * $zero, N` after a table naming the library). The loader rewrites each stub's
 * first word as a `syscall` carrying a host function's id, so calling the stub
 * runs the host's implementation (iop_libs.cpp) and returns to $ra.
 *
 * Not modelled: the R3000's load delay slots (compiled code never depends on
 * them), its caches, and IOP hardware registers - modules reach the hardware
 * through libraries the host implements; a direct access stops the run, naming
 * the address.
 */

#include "iop_vm_internal.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <set>
#include <thread>

#include "ps2host/iop_vm.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {
namespace iop {

namespace {

constexpr u32 kHleCode = 0x80000u;          // syscall codes at or above this are host calls
constexpr u64 kSlice = 20'000'000;          // instructions a call may run, so a busy IOP can't hang the EE
// A thread runs at most this many instructions before the scheduler looks again:
// interrupts (alarms, sound, DMA completions) arrive between quanta, as the
// real IOP takes them whenever they come.
constexpr u64 kQuantum = 4096;
constexpr u32 kInterruptStack = 0x0000FF00u;    // below the heap, in the kernel's area

std::set<std::string>& RunList() {
    static std::set<std::string> names;
    return names;
}

std::string Upper(std::string text) {
    for (char& ch : text) ch = char(std::toupper(static_cast<unsigned char>(ch)));
    return text;
}

std::string Hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%08X", value);
    return text;
}

[[noreturn]] void Stop(const std::string& what) { throw Unimplemented{"IOP: " + what}; }

s32 Sx16(u32 value) { return s32(s16(u16(value))); }

}  // namespace

// ---------------------------------------------------------------- memory

// Where the IOP is, for messages: the running thread, or a handler.
std::string Vm::Where() {
    if (in_interrupt || !current) return " (in a handler)";
    return " (IOP thread " + std::to_string(current) + ", $ra " + Hex(threads[current].c.r[31]) + ")";
}

u8* Vm::At(u32 address, u32 size) {
    const u32 physical = address & 0x1FFFFFFFu;
    if (physical < 0x00800000u) {                       // RAM and its mirrors
        const u32 offset = physical & (kRamSize - 1);
        if (offset + size <= kRamSize) return ram + offset;
    }
    return nullptr;
}

u32 Vm::Read(u32 address, u32 size) {
    const u8* p = At(address, size);
    if (!p) Stop("a read of " + Hex(address) + " (IOP hardware or unmapped), by the instruction at " +
                 Hex(executing) + Where());
    u32 value = 0;
    std::memcpy(&value, p, size);
    return value;
}

void Vm::Write(u32 address, u32 value, u32 size) {
    u8* p = At(address, size);
    if (!p) Stop("a write of " + Hex(value) + " to " + Hex(address) +
                 " (IOP hardware or unmapped), by the instruction at " + Hex(executing) + Where());
    std::memcpy(p, &value, size);
}

std::string Vm::String(u32 address, u32 limit) {
    std::string out;
    for (u32 i = 0; i < limit; ++i) {
        const u8* p = At(address + i, 1);
        if (!p || !*p) break;
        out += char(*p);
    }
    return out;
}

u32 Vm::Allocate(u32 size) {
    size = (size + 15) & ~15u;
    for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
        if (it->second < size) continue;
        const u32 address = it->first, left = it->second - size;
        free_blocks.erase(it);
        if (left) free_blocks[address + size] = left;
        used_blocks[address] = size;
        std::memset(ram + address, 0, size);
        return address;
    }
    return 0;
}

void Vm::Free(u32 address) {
    auto used = used_blocks.find(address);
    if (used == used_blocks.end()) return;
    u32 start = address, size = used->second;
    used_blocks.erase(used);
    auto next = free_blocks.find(start + size);
    if (next != free_blocks.end()) {
        size += next->second;
        free_blocks.erase(next);
    }
    auto prev = free_blocks.lower_bound(start);
    if (prev != free_blocks.begin() && (--prev)->first + prev->second == start) {
        prev->second += size;
        return;
    }
    free_blocks[start] = size;
}

u32 Vm::LargestFree() const {
    u32 best = 0;
    for (const auto& [address, size] : free_blocks) best = size > best ? size : best;
    return best;
}

u64 Vm::NowUs() const {
    static const u64 start = ClockNs();
    return (ClockNs() - start) / 1000;
}

Vm& TheVm() {
    static Vm vm;
    if (!vm.ram) vm.ram = GuestMemory().Translate(kIopRamBase, kRamSize);
    return vm;
}

u32 Arg(Vm& vm, const Context& c, int i) {
    return i < 4 ? c.r[4 + i] : vm.Read(c.r[29] + 16 + 4 * u32(i - 4), 4);
}

// ---------------------------------------------------------------- threads

void Vm::MakeReady(Thread& t) {
    t.ready = true;
    t.wait = Wait::None;
    t.queued = ++counter;
    if (current && !in_interrupt && t.priority < threads[current].priority) reschedule = true;
    if (in_interrupt) reschedule = true;
}

bool Vm::Block(Context& c, Wait why, u32 object) {
    Thread* t = Current();
    if (in_interrupt || !t) return false;
    (void)c;
    t->ready = false;
    t->wait = why;
    t->object = object;
    return true;
}

// ---------------------------------------------------------------- interpreter

namespace {

// Runs `c` until it reaches `stop`, a library call blocks or yields, or
// `limit` instructions have run. Returns what ended it.
Outcome Execute(Vm& vm, Context& c, u32 stop, u64 limit, bool* stopped) {
    *stopped = false;
    for (u64 n = 0; n < limit; ++n) {
        if (c.pc == stop) {
            *stopped = true;
            return Outcome::Return;
        }
        if (c.pc == kThreadExit || c.pc == kRpcReturn) {
            *stopped = true;
            return Outcome::Block;
        }
        const u32 pc = c.pc;
        vm.executing = pc;
        const u32 w = vm.Read(pc, 4);
        c.pc = c.npc;
        c.npc += 4;
        ++vm.instructions;
        const u32 op = w >> 26, rs = (w >> 21) & 31, rt = (w >> 16) & 31, rd = (w >> 11) & 31;
        const u32 sa = (w >> 6) & 31, imm = w & 0xFFFF;
        const s32 simm = Sx16(imm);
        u32* r = c.r;
        auto set = [&](u32 reg, u32 value) { if (reg) r[reg] = value; };
        auto branch = [&](bool taken) { if (taken) c.npc = pc + 4 + u32(simm << 2); };
        switch (op) {
            case 0x00:
                switch (w & 0x3F) {
                    case 0x00: set(rd, r[rt] << sa); break;                          // sll
                    case 0x02: set(rd, r[rt] >> sa); break;                          // srl
                    case 0x03: set(rd, u32(s32(r[rt]) >> sa)); break;                // sra
                    case 0x04: set(rd, r[rt] << (r[rs] & 31)); break;                // sllv
                    case 0x06: set(rd, r[rt] >> (r[rs] & 31)); break;                // srlv
                    case 0x07: set(rd, u32(s32(r[rt]) >> (r[rs] & 31))); break;      // srav
                    case 0x08: c.npc = r[rs]; break;                                 // jr
                    case 0x09: {                                                     // jalr
                        const u32 t = r[rs];
                        set(rd, pc + 8);
                        c.npc = t;
                        if (vm.trace) { auto& e = vm.calls[t]; ++e.first; e.second = pc; }
                        break;
                    }
                    case 0x0C: {                                                     // syscall
                        const u32 code = (w >> 6) & 0xFFFFF;
                        if (code < kHleCode || code - kHleCode >= vm.libraries.size())
                            Stop("syscall " + Hex(code) + " at " + Hex(pc));
                        const u32 a0 = r[4], a1 = r[5], a2 = r[6], a3 = r[7];
                        const Outcome outcome = vm.libraries[code - kHleCode](vm, c);
                        if (vm.trace) {
                            const auto& stub = vm.stubs[pc];
                            std::fprintf(stderr, "[iop] %s %s#%u %s(%08X, %08X, %08X, %08X) -> %08X%s from %08X\n",
                                         vm.in_interrupt ? "handler" : ("t" + std::to_string(vm.current)).c_str(),
                                         stub.first.c_str(), unsigned(stub.second),
                                         ExportName(stub.first, stub.second).c_str(), a0, a1, a2, a3,
                                         r[2], outcome == Outcome::Block ? " (waits)" : "", r[31] - 8);
                        }
                        c.pc = r[31];
                        c.npc = r[31] + 4;
                        if (outcome != Outcome::Return) return outcome;
                        if (vm.reschedule && !vm.in_interrupt) return Outcome::Yield;
                        break;
                    }
                    case 0x0D: Stop("break at " + Hex(pc));
                    case 0x10: set(rd, c.hi); break;                                 // mfhi
                    case 0x11: c.hi = r[rs]; break;                                  // mthi
                    case 0x12: set(rd, c.lo); break;                                 // mflo
                    case 0x13: c.lo = r[rs]; break;                                  // mtlo
                    case 0x18: {                                                     // mult
                        const s64 p = s64(s32(r[rs])) * s64(s32(r[rt]));
                        c.lo = u32(p); c.hi = u32(u64(p) >> 32); break;
                    }
                    case 0x19: {                                                     // multu
                        const u64 p = u64(r[rs]) * u64(r[rt]);
                        c.lo = u32(p); c.hi = u32(p >> 32); break;
                    }
                    case 0x1A: {                                                     // div
                        const s32 a = s32(r[rs]), b = s32(r[rt]);
                        if (b == 0) { c.lo = a < 0 ? 1u : 0xFFFFFFFFu; c.hi = u32(a); }
                        else if (a == INT32_MIN && b == -1) { c.lo = u32(a); c.hi = 0; }
                        else { c.lo = u32(a / b); c.hi = u32(a % b); }
                        break;
                    }
                    case 0x1B: {                                                     // divu
                        const u32 a = r[rs], b = r[rt];
                        if (b == 0) { c.lo = 0xFFFFFFFFu; c.hi = a; }
                        else { c.lo = a / b; c.hi = a % b; }
                        break;
                    }
                    case 0x20: case 0x21: set(rd, r[rs] + r[rt]); break;             // add(u)
                    case 0x22: case 0x23: set(rd, r[rs] - r[rt]); break;             // sub(u)
                    case 0x24: set(rd, r[rs] & r[rt]); break;
                    case 0x25: set(rd, r[rs] | r[rt]); break;
                    case 0x26: set(rd, r[rs] ^ r[rt]); break;
                    case 0x27: set(rd, ~(r[rs] | r[rt])); break;
                    case 0x2A: set(rd, s32(r[rs]) < s32(r[rt]) ? 1 : 0); break;
                    case 0x2B: set(rd, r[rs] < r[rt] ? 1 : 0); break;
                    default: Stop("instruction " + Hex(w) + " at " + Hex(pc));
                }
                break;
            case 0x01: {                                                             // regimm
                const bool link = (rt & 0x1E) == 0x10;
                const bool taken = (rt & 1) ? s32(r[rs]) >= 0 : s32(r[rs]) < 0;
                if (link) r[31] = pc + 8;
                branch(taken);
                break;
            }
            case 0x02: c.npc = (pc & 0xF0000000u) | ((w & 0x03FFFFFFu) << 2); break;  // j
            case 0x03:                                                               // jal
                r[31] = pc + 8;
                c.npc = (pc & 0xF0000000u) | ((w & 0x03FFFFFFu) << 2);
                if (vm.trace) { auto& e = vm.calls[c.npc]; ++e.first; e.second = pc; }
                break;
            case 0x04: branch(r[rs] == r[rt]); break;
            case 0x05: branch(r[rs] != r[rt]); break;
            case 0x06: branch(s32(r[rs]) <= 0); break;
            case 0x07: branch(s32(r[rs]) > 0); break;
            case 0x08: case 0x09: set(rt, r[rs] + u32(simm)); break;                 // addi(u)
            case 0x0A: set(rt, s32(r[rs]) < simm ? 1 : 0); break;
            case 0x0B: set(rt, r[rs] < u32(simm) ? 1 : 0); break;
            case 0x0C: set(rt, r[rs] & imm); break;
            case 0x0D: set(rt, r[rs] | imm); break;
            case 0x0E: set(rt, r[rs] ^ imm); break;
            case 0x0F: set(rt, imm << 16); break;
            case 0x10:                                                               // cop0
                if (rs == 0x00) set(rt, rd == 12 ? 0x401u : 0); // mfc0: Status, interrupts on
                else if (rs == 0x04) {}                          // mtc0: no effect here
                else if (rs == 0x10 && (w & 0x3F) == 0x10) {}    // rfe
                else Stop("COP0 instruction " + Hex(w) + " at " + Hex(pc));
                break;
            case 0x20: set(rt, u32(s32(s8(vm.Read(r[rs] + u32(simm), 1))))); break;   // lb
            case 0x21: set(rt, u32(Sx16(vm.Read(r[rs] + u32(simm), 2)))); break;      // lh
            case 0x22: case 0x26: {                                                  // lwl / lwr
                const u32 a = r[rs] + u32(simm), word = vm.Read(a & ~3u, 4), k = a & 3;
                if (op == 0x22) set(rt, (r[rt] & (0x00FFFFFFu >> (8 * k))) | (word << (24 - 8 * k)));
                else set(rt, (r[rt] & ~(0xFFFFFFFFu >> (8 * k))) | (word >> (8 * k)));
                break;
            }
            case 0x23: set(rt, vm.Read(r[rs] + u32(simm), 4)); break;                // lw
            case 0x24: set(rt, vm.Read(r[rs] + u32(simm), 1)); break;                // lbu
            case 0x25: set(rt, vm.Read(r[rs] + u32(simm), 2)); break;                // lhu
            case 0x28: vm.Write(r[rs] + u32(simm), r[rt], 1); break;                 // sb
            case 0x29: vm.Write(r[rs] + u32(simm), r[rt], 2); break;                 // sh
            case 0x2A: case 0x2E: {                                                  // swl / swr
                const u32 a = r[rs] + u32(simm), word = vm.Read(a & ~3u, 4), k = a & 3;
                const u32 value = op == 0x2A
                    ? (word & ~(0xFFFFFFFFu >> (24 - 8 * k))) | (r[rt] >> (24 - 8 * k))
                    : (word & ~(0xFFFFFFFFu << (8 * k))) | (r[rt] << (8 * k));
                vm.Write(a & ~3u, value, 4);
                break;
            }
            case 0x2B: vm.Write(r[rs] + u32(simm), r[rt], 4); break;                 // sw
            default: Stop("instruction " + Hex(w) + " at " + Hex(pc));
        }
    }
    return Outcome::Return;          // the quantum ran out: still ready, not rotated
}

}  // namespace

void Vm::Step(Context& c, u64 count) {
    bool stopped = false;
    Execute(*this, c, kReturnFromCall, count, &stopped);
}

u32 Vm::Call(u32 function, std::vector<u32> args) {
    Context c;
    for (std::size_t i = 0; i < args.size() && i < 4; ++i) c.r[4 + i] = args[i];
    c.r[29] = kInterruptStack - 0x10;
    c.r[31] = kReturnFromCall;
    c.pc = function;
    c.npc = function + 4;
    const bool nested = in_interrupt;
    const u32 thread = current;
    in_interrupt = true;
    current = 0;
    bool stopped = false;
    const Outcome outcome = Execute(*this, c, kReturnFromCall, kSlice, &stopped);
    in_interrupt = nested;
    current = thread;
    if (!stopped) Stop(outcome == Outcome::Block ? "a handler at " + Hex(function) + " blocked"
                                                 : "a handler at " + Hex(function) + " did not return");
    return c.r[2];
}

// ---------------------------------------------------------------- scheduling

void Run(u64 slice) {
    Vm& vm = TheVm();
    if (vm.threads.empty() && vm.alarms.empty() && vm.completions.empty()) return;
    const u64 start = vm.instructions;
    for (;;) {
        SoundInterrupts(vm);
        // Completion callbacks of finished SIF DMA (sceSifSetDmaIntr), as interrupts.
        while (!vm.completions.empty()) {
            const auto done = vm.completions.front();
            vm.completions.erase(vm.completions.begin());
            vm.Call(done.first, {done.second});
        }
        const u64 now = vm.NowUs();
        for (auto& [id, t] : vm.threads)
            if (t.wait == Wait::Delay && now >= t.until_us) {
                t.c.r[2] = 0;
                vm.MakeReady(t);
            }
        for (auto it = vm.alarms.begin(); it != vm.alarms.end();) {
            if (now < it->second.due_us) { ++it; continue; }
            const Alarm alarm = it->second;
            it = vm.alarms.erase(it);
            const u32 again = vm.Call(alarm.handler, {alarm.arg});
            if (again) vm.alarms[alarm.handler] = {now + again / 37, alarm.handler, alarm.arg};
        }
        Thread* best = nullptr;
        for (auto& [id, t] : vm.threads)
            if (t.ready && (!best || t.priority < best->priority ||
                            (t.priority == best->priority && t.queued < best->queued)))
                best = &t;
        if (!best || vm.instructions - start > slice) return;
        vm.current = best->id;
        vm.reschedule = false;
        bool stopped = false;
        const Outcome outcome = Execute(vm, best->c, kReturnFromCall, kQuantum, &stopped);
        if (best->c.pc == kRpcReturn) {                      // a server function returned
            const RpcRequest& done = best->serving;
            Memory& ee = GuestMemory();
            const u32 answer = best->c.r[2];
            for (u32 i = 0; i < done.receive_size && done.receive && answer; ++i)
                ee.Write(done.receive + i, vm.Read(answer + i, 1), 1);
            const RpcServerEntry& entry = vm.rpc_servers[done.server];
            QueueRpcEnd(done.packet, entry.server, entry.buffer, entry.cbuffer);
            best->ready = false;
            best->wait = Wait::Rpc;
            ServeNext(vm, *best);
        } else if (best->c.pc == kThreadExit) {              // returned from its entry
            best->ready = false;
            best->wait = Wait::Dormant;
        } else if (outcome == Outcome::Yield && best->ready) {
            best->queued = ++vm.counter;                     // back of its priority's queue
        }
        // A thread that has just entered sceSifRpcLoop serves what is waiting.
        if (!best->ready && best->wait == Wait::Rpc) ServeNext(vm, *best);
        vm.current = 0;
    }
}

// ---------------------------------------------------------------- modules

void SetTrace(bool on) { TheVm().trace = on; }

void RunModule(const std::string& file_name) { RunList().insert(Upper(file_name)); }

bool RunsModule(const std::string& path) {
    std::string name = Upper(path);
    const std::size_t slash = name.find_last_of("\\/:");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    const std::size_t version = name.find(';');
    if (version != std::string::npos) name = name.substr(0, version);
    return RunList().count(name) != 0;
}

bool LoadModule(const std::string& path, const std::vector<u8>& image,
                const std::vector<u8>& args, s32* result) {
    Vm& vm = TheVm();
    auto u16at = [&](std::size_t o) { u16 v = 0; std::memcpy(&v, &image[o], 2); return v; };
    auto u32at = [&](std::size_t o) { u32 v = 0; std::memcpy(&v, &image[o], 4); return v; };
    if (image.size() < 0x34 || u32at(0) != 0x464C457Fu) return false;
    const u32 entry = u32at(0x18), phoff = u32at(0x1C), shoff = u32at(0x20);
    const u16 phentsize = u16at(0x2A), phnum = u16at(0x2C), shentsize = u16at(0x2E), shnum = u16at(0x30);
    // The loadable segment: the module's code and data, linked at 0.
    u32 offset = 0, filesz = 0, memsz = 0;
    for (u16 i = 0; i < phnum; ++i) {
        const std::size_t ph = phoff + std::size_t(i) * phentsize;
        if (u32at(ph) == 1) { offset = u32at(ph + 4); filesz = u32at(ph + 16); memsz = u32at(ph + 20); break; }
    }
    if (!memsz || offset + filesz > image.size()) return false;
    const u32 base = vm.Allocate(memsz);
    if (!base) Stop("no IOP memory for " + path);
    std::memcpy(vm.ram + base, &image[offset], filesz);
    // Relocations: every SHT_REL section, against the base.
    for (u16 i = 0; i < shnum; ++i) {
        const std::size_t sh = shoff + std::size_t(i) * shentsize;
        if (u32at(sh + 4) != 9) continue;                       // SHT_REL
        const u32 rel = u32at(sh + 16), size = u32at(sh + 20);
        for (u32 k = 0; k + 8 <= size; k += 8) {
            const u32 where = base + u32at(rel + k), type = u32at(rel + k + 4) & 0xFF;
            const u32 word = vm.Read(where, 4);
            switch (type) {
                case 2: vm.Write(where, word + base, 4); break;                          // R_MIPS_32
                case 4: vm.Write(where, (word & 0xFC000000u) |
                                        ((word + (base >> 2)) & 0x03FFFFFFu), 4); break;  // R_MIPS_26
                case 5: {                                                                // R_MIPS_HI16
                    // Paired with the next LO16: the carry depends on the full value.
                    u32 lo = 0;
                    for (u32 j = k + 8; j + 8 <= size; j += 8)
                        if ((u32at(rel + j + 4) & 0xFF) == 6) {
                            lo = vm.Read(base + u32at(rel + j), 4) & 0xFFFF;
                            break;
                        }
                    const u32 full = (word << 16) + u32(Sx16(lo)) + base;
                    vm.Write(where, (word & 0xFFFF0000u) | (((full + 0x8000u) >> 16) & 0xFFFF), 4);
                    break;
                }
                case 6: vm.Write(where, (word & 0xFFFF0000u) | ((word + base) & 0xFFFF), 4); break;
                case 0: break;
                default: Stop(path + ": relocation type " + std::to_string(type));
            }
        }
    }
    // Imports: patch each stub into a call of the host's implementation.
    std::vector<std::string> missing;
    for (u32 a = base; a + 20 <= base + filesz; a += 4) {
        if (vm.Read(a, 4) != 0x41E00000u) continue;
        const std::string library = vm.String(a + 12, 8);
        u32 stub = a + 20;
        while (vm.Read(stub, 4) == 0x03E00008u && (vm.Read(stub + 4, 4) >> 16) == 0x2400u) {
            const u32 index = vm.Read(stub + 4, 4) & 0xFFFF;
            Library function = FindLibrary(library, index);
            const std::string name = ExportName(library, index);
            if (!function) {
                missing.push_back(library + "#" + std::to_string(index) + " " + name);
                function = [library, index, name](Vm&, Context&) -> Outcome {
                    Stop("library function " + library + "#" + std::to_string(index) + " (" +
                         name + ") is not implemented");
                };
            }
            vm.stubs[stub] = {library, index};
            vm.libraries.push_back(function);
            vm.Write(stub, ((kHleCode + u32(vm.libraries.size() - 1)) << 6) | 0x0C, 4);
            stub += 8;
        }
    }
    for (const std::string& name : missing)
        Note("iop missing " + name, "IOP: " + path + " imports " + name + ", which the host lacks");
    // Start it: _start(argc, argv), argv[0] the path, then the argument strings.
    std::vector<std::string> words{path};
    std::string current;
    for (u8 ch : args) {
        if (ch) current += char(ch);
        else if (!current.empty()) { words.push_back(current); current.clear(); }
    }
    if (!current.empty()) words.push_back(current);
    u32 total = 4 * u32(words.size() + 1);
    for (const auto& w : words) total += u32(w.size()) + 1;
    const u32 argv = vm.Allocate(total);
    u32 text = argv + 4 * u32(words.size() + 1);
    for (std::size_t i = 0; i < words.size(); ++i) {
        vm.Write(argv + 4 * u32(i), text, 4);
        std::memcpy(vm.ram + text, words[i].c_str(), words[i].size() + 1);
        text += u32(words[i].size()) + 1;
    }
    char line[160];
    std::snprintf(line, sizeof line, "IOP: running %s at 0x%08X (%u bytes), entry 0x%08X",
                  path.c_str(), unsigned(base), unsigned(memsz), unsigned(base + entry));
    Note("iop run " + path, line);
    // _start runs as a thread, as the IOP's module loader runs it: it may wait,
    // and it may ask for its own thread id. Its return value is the result.
    Thread loader;
    loader.id = vm.next_id++;
    loader.entry = base + entry;
    loader.priority = 8;
    loader.stack_size = 0x1000;
    loader.stack = vm.Allocate(loader.stack_size);
    loader.c.r[4] = u32(words.size());
    loader.c.r[5] = argv;
    loader.c.r[29] = loader.stack + loader.stack_size - 16;
    loader.c.r[31] = kThreadExit;
    loader.c.pc = loader.entry;
    loader.c.npc = loader.entry + 4;
    Thread& started = vm.threads[loader.id] = loader;
    vm.MakeReady(started);
    const u64 deadline = vm.NowUs() + 5000000;      // it may wait on timers: real time
    while (vm.threads[loader.id].wait != Wait::Dormant && vm.NowUs() < deadline) {
        Run();
        if (vm.threads[loader.id].wait != Wait::Dormant)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    Thread& done = vm.threads[loader.id];
    if (done.wait != Wait::Dormant) Stop(path + "'s start routine did not finish");
    *result = s32(done.c.r[2]);
    vm.Free(done.stack);
    vm.threads.erase(loader.id);
    Run();
    return true;
}

void ServeNext(Vm& vm, Thread& t) {
    auto& pending = vm.rpc_pending[t.rpc_queue];
    if (pending.empty() || t.ready || t.wait != Wait::Rpc) return;
    t.serving = pending.front();
    pending.pop_front();
    const RpcServerEntry& entry = vm.rpc_servers[t.serving.server];
    t.c.r[4] = t.serving.procedure;
    t.c.r[5] = entry.buffer;
    t.c.r[6] = t.serving.size;
    t.c.r[31] = kRpcReturn;
    t.c.pc = entry.function;
    t.c.npc = entry.function + 4;
    vm.MakeReady(t);
}

bool RpcServer(u32 sid, u32* server, u32* buffer, u32* cbuffer) {
    for (const auto& [address, entry] : TheVm().rpc_servers)
        if (entry.sid == sid) {
            *server = address;
            *buffer = entry.buffer;
            *cbuffer = entry.cbuffer;
            return true;
        }
    return false;
}

bool OwnsServer(u32 server) { return TheVm().rpc_servers.count(server) != 0; }

void RpcCall(u32 server, u32 procedure, u32 size, u32 receive, u32 receive_size,
             const std::vector<u8>& packet) {
    Vm& vm = TheVm();
    const RpcServerEntry& entry = vm.rpc_servers[server];
    char text[160];
    std::snprintf(text, sizeof text, "IOP: RPC 0x%08X procedure %u (%u bytes in, %u out) served by a module",
                  unsigned(entry.sid), unsigned(procedure), unsigned(size), unsigned(receive_size));
    Note("iop rpc call " + std::to_string(entry.sid) + " " + std::to_string(procedure), text);
    if (vm.trace) {
        // The request as it sits in the server's buffer: words, and any text.
        std::string words, chars;
        for (u32 i = 0; i < size && i < 64; i += 4) {
            char w[12];
            std::snprintf(w, sizeof w, " %08X", unsigned(vm.Read(entry.buffer + i, 4)));
            words += w;
        }
        for (u32 i = 0; i < size && i < 64; ++i) {
            const char ch = char(vm.Read(entry.buffer + i, 1));
            chars += (ch >= 32 && ch < 127) ? ch : '.';
        }
        std::fprintf(stderr, "[iop] RPC 0x%08X procedure %u:%s  \"%s\"\n", unsigned(entry.sid),
                     unsigned(procedure), words.c_str(), chars.c_str());
    }
    RpcRequest request{server, procedure, size, receive, receive_size, packet};
    vm.rpc_pending[entry.queue].push_back(request);
    auto thread = vm.rpc_queues.find(entry.queue);
    if (thread != vm.rpc_queues.end()) ServeNext(vm, vm.threads[thread->second]);
    Run();
}

u32 Allocate(u32 size) { return TheVm().Allocate(size); }
void Free(u32 address) { TheVm().Free(address); }

bool HasCommandHandler(u32 command) { return TheVm().command_handlers.count(command) != 0; }

void Command(u32 command, const std::vector<u8>& packet) {
    Vm& vm = TheVm();
    const auto found = vm.command_handlers.find(command);
    if (found == vm.command_handlers.end()) return;
    if (!vm.command_buffer) vm.command_buffer = vm.Allocate(0x100);
    if (packet.size() >= 0x18) {
        u32 words[4];
        std::memcpy(words, &packet[0x10], std::min<std::size_t>(16, packet.size() - 0x10));
        char text[160];
        std::snprintf(text, sizeof text, "IOP: command %u from the EE: %08X %08X %08X %08X",
                      unsigned(command), words[0], words[1], words[2], words[3]);
        Note("iop command " + std::to_string(command) + " " + std::to_string(words[0]), text);
    }
    std::memcpy(vm.ram + vm.command_buffer, packet.data(), packet.size() < 0x100 ? packet.size() : 0x100);
    vm.Call(found->second.first, {vm.command_buffer, found->second.second});
    Run();
}

std::string Report() {
    Vm& vm = TheVm();
    std::string out;
    if (vm.trace && !vm.calls.empty()) {
        // The IOP's busiest call targets (with --iop-trace): where its time goes.
        std::vector<std::pair<u64, u32>> hot;
        for (const auto& [target, e] : vm.calls) hot.push_back({e.first, target});
        std::sort(hot.rbegin(), hot.rend());
        out += "  IOP calls, most frequent:\n";
        for (std::size_t i = 0; i < hot.size() && i < 30; ++i) {
            char line[96];
            std::snprintf(line, sizeof line, "    0x%08X x%llu (from 0x%08X)\n", unsigned(hot[i].second),
                          static_cast<unsigned long long>(hot[i].first), unsigned(vm.calls[hot[i].second].second));
            out += line;
        }
    }
    static const char* waits[] = {"ready", "waiting on a semaphore", "waiting on an event flag",
                                  "sleeping", "delayed", "dormant", "serving RPC"};
    for (auto& [id, t] : vm.threads) {
        char line[160];
        std::snprintf(line, sizeof line, "  IOP thread %u: %s, priority %d, entry 0x%08X, at 0x%08X\n",
                      id, t.ready ? "ready" : waits[int(t.wait)], t.priority, t.entry, t.c.pc);
        out += line;
    }
    return out;
}

}  // namespace iop
}  // namespace ps2host
