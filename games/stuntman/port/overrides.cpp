/*
 * Stuntman's host implementations of guest functions.
 *
 * An override replaces the translated function at its address: the way a port
 * redirects code that drives hardware the host provides differently, and the
 * way the PC features hook in (a settings menu in the game's own menus,
 * widescreen, the mouse's targets). Register them here with
 * ps2host::Override(address, function, "why"); the original stays callable as
 * F_<ADDRESS>(c). See the engine's docs/PORTING.md and the ps2-port skill.
 *
 * Override as little as possible: the game's own code is the point.
 */

#include <cstdio>
#include <cstring>
#include <string>

#include "ps2host/host.h"
#include "ps2host/iop_vm.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace {

std::string GuestString(u32 address, std::size_t limit = 4096) {
    ps2host::Memory& m = ps2host::GuestMemory();
    std::string text;
    for (u32 p = address; address && text.size() < limit; ++p) {
        const char ch = char(m.Read(p, 1));
        if (!ch) break;
        text += ch;
    }
    return text;
}

// printf (0x00253020): the SDK's sends its text over DECI2, which a retail
// console - and the host - does not have, so nothing was ever seen. Here it is
// formatted from the game's own arguments and logged as [guest] lines, as the
// kernel's kputs is. The arguments, under the EE's EABI: $a1-$a3 and $t0-$t3
// (r5-r11), then the stack in 8-byte slots; a float is promoted to a double,
// which travels whole in one 64-bit register. Handled: %d %i %u %x %X %o %c %s
// %p %f %e %g %%, with flags, width and precision, and h/l/ll.
void GuestPrintf(lifted::Context& c) {
    ps2host::Memory& m = ps2host::GuestMemory();
    int next = 5;
    u32 stack = u32(c.r[29]);
    auto argument = [&]() -> u64 {
        if (next <= 11) return c.r[next++];
        const u64 value = m.Read(stack, 4) | (m.Read(stack + 4, 4) << 32);
        stack += 8;
        return value;
    };
    const std::string format = GuestString(u32(c.r[4]));
    std::string out;
    for (std::size_t i = 0; i < format.size(); ++i) {
        if (format[i] != '%') {
            out += format[i];
            continue;
        }
        std::size_t j = i + 1;
        while (j < format.size() && std::strchr("-+ #0123456789.*", format[j])) ++j;
        std::string spec = format.substr(i, j - i);
        int longs = 0;
        while (j < format.size() && std::strchr("hlLqjzt", format[j])) {
            if (format[j] == 'l' || format[j] == 'q' || format[j] == 'L') ++longs;
            ++j;
        }
        if (j >= format.size()) {
            out += format.substr(i);
            break;
        }
        const char conversion = format[j];
        char buffer[512];
        buffer[0] = 0;
        // A '*' width or precision is an argument of its own.
        std::string resolved;
        for (char ch : spec) {
            if (ch == '*') resolved += std::to_string(s32(u32(argument())));
            else resolved += ch;
        }
        switch (conversion) {
            case 'd': case 'i': {
                const u64 v = argument();
                std::snprintf(buffer, sizeof buffer, (resolved + "lld").c_str(),
                              longs ? (long long)s64(v) : (long long)s32(u32(v)));
                break;
            }
            case 'u': case 'x': case 'X': case 'o': {
                const u64 v = argument();
                std::snprintf(buffer, sizeof buffer, (resolved + "ll" + conversion).c_str(),
                              longs ? (unsigned long long)v : (unsigned long long)u32(v));
                break;
            }
            case 'c':
                std::snprintf(buffer, sizeof buffer, (resolved + "c").c_str(), int(u8(argument())));
                break;
            case 's':
                std::snprintf(buffer, sizeof buffer, (resolved + "s").c_str(),
                              GuestString(u32(argument()), 400).c_str());
                break;
            case 'p':
                std::snprintf(buffer, sizeof buffer, "0x%08X", unsigned(u32(argument())));
                break;
            case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': {
                const u64 bits = argument();
                double value;
                std::memcpy(&value, &bits, 8);
                std::snprintf(buffer, sizeof buffer, (resolved + conversion).c_str(), value);
                break;
            }
            case '%':
                out += '%';
                break;
            default:                                         // shown as written
                out += format.substr(i, j + 1 - i);
                break;
        }
        out += buffer;
        i = j;
    }
    std::size_t from = 0;
    while (from < out.size()) {
        std::size_t end = out.find('\n', from);
        if (end == std::string::npos) end = out.size();
        std::string line = out.substr(from, end - from);
        while (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) std::fprintf(stderr, "[guest] %s\n", line.c_str());
        from = end + 1;
    }
    std::fflush(stderr);
    c.r[2] = u64(s64(s32(out.size())));                     // printf's result: the characters
}

}  // namespace

void RegisterGameOverrides() {
    // ps2host::Override(0x00123456u, SomeHook, "what it does and why");

    ps2host::Override(0x00253020u, GuestPrintf, "printf: the game's messages to the log as [guest] lines");

    // The game's own IOP modules run in the IOP interpreter (PORTING.md):
    // XAVSTRM.IRX ("xavstr v1.1") serves RPC 0x7F7F7F7F, whose client
    // (0x002039D0) requires version 0x101 and stops for ever otherwise.
    ps2host::iop::RunModule("XAVSTRM.IRX");
    // IOPHEAP.IRX is the game's own ("Stuntman_IOP_Heap v1.0", not Sony's
    // iopheap): it serves RPC 0x21166779, procedures 256 and 257.
    ps2host::iop::RunModule("IOPHEAP.IRX");
    // SOUNDM.IRX: the game's sound driver ("StuntSound v2.243").
    ps2host::iop::RunModule("SOUNDM.IRX");
}
