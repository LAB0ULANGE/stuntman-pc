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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "ps2host/host.h"
#include "ps2host/iop_vm.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"

// The translated originals, which stay callable (port.py's dispatch.cpp).
void F_00251FE0(lifted::Context& c);
void F_002521E0(lifted::Context& c);
void F_00252208(lifted::Context& c);
void F_001D0DA0(lifted::Context& c);
void F_001D4FA8(lifted::Context& c);
void F_001D6008(lifted::Context& c);
void F_001D6440(lifted::Context& c);

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

// newlib's malloc, memalign and free, as the game patches them. At start-up
// 0x00102860 (the game heap's set-up, from the entry) has 0x001027A8 write a
// jump over the call each makes into newlib's allocator (FINDINGS.md):
//   0x002521F4 malloc    jal _malloc_r 0x00252470   -> jal 0x001027E0
//   0x00251FF8 memalign  jal _memalign_r 0x00252010 -> jal 0x00102810
//   0x00252210 free      j _free_r 0x002513F0       -> j 0x00102838
// so every block comes from, and goes back to, the game's allocator
// (0x00102360, 0x00102638). The translation is the executable's, unpatched:
// these run the patched code, instruction for instruction - the word in memory
// read at each call. Before the patch they run the original; a word that is
// neither stops the run rather than guess.
constexpr u32 kImpurePtr = 0x0027919Cu;      // _impure_ptr, newlib's reentrancy struct

bool Patched(u32 at, u32 patched, u32 original) {
    const u32 word = u32(ps2host::GuestMemory().Read(at, 4));
    if (word == patched) return true;
    if (word == original) return false;
    char text[160];
    std::snprintf(text, sizeof text, "newlib's allocator call at 0x%08X is 0x%08X, neither the executable's "
                  "(0x%08X) nor the game's patch (0x%08X)", unsigned(at), unsigned(word), unsigned(original),
                  unsigned(patched));
    throw ps2host::Unimplemented{text};
}

u64 ImpurePtr() { return u64(s64(s32(u32(ps2host::GuestMemory().Read(kImpurePtr, 4))))); }

// The frame malloc and memalign keep: $ra saved at $sp, 16 bytes.
void CallFramed(lifted::Context& c, u32 target, u32 return_address) {
    ps2host::Memory& m = ps2host::GuestMemory();
    c.r[29] -= 0x10;
    m.Write(u32(c.r[29]), c.r[31], 8);
    lifted::Call(c, target, return_address);
    c.r[31] = m.Read(u32(c.r[29]), 8);
    c.r[29] += 0x10;
}

void PatchedMalloc(lifted::Context& c) {                     // 0x002521E0
    if (!Patched(0x002521F4u, 0x0C0409F8u, 0x0C09491Cu)) return F_002521E0(c);
    c.r[5] = c.r[4];
    c.r[2] = 0x00280000u;
    c.r[4] = ImpurePtr();
    CallFramed(c, 0x001027E0u, 0x002521FCu);
}

void PatchedMemalign(lifted::Context& c) {                   // 0x00251FE0
    if (!Patched(0x00251FF8u, 0x0C040A04u, 0x0C094804u)) return F_00251FE0(c);
    c.r[2] = c.r[4];
    c.r[3] = 0x00280000u;
    c.r[6] = c.r[5];
    c.r[4] = ImpurePtr();
    c.r[5] = c.r[2];
    CallFramed(c, 0x00102810u, 0x00252000u);
}

void PatchedFree(lifted::Context& c) {                       // 0x00252208, a jump: $ra stays the caller's
    if (!Patched(0x00252210u, 0x08040A0Eu, 0x080944FCu)) return F_00252208(c);
    c.r[5] = c.r[4];
    c.r[2] = 0x00280000u;
    c.r[4] = ImpurePtr();
    lifted::Call(c, 0x00102838u);
}

// The light table (0x001D0DA0). Each frame the game builds the 256-entry
// CLUT a full-screen pass multiplies the picture by, indexed by the light
// level each pixel keeps in its alpha: max(the ambient light, the level),
// the ambient being the brightest of the scene's ambient colour (0x005C2180,
// three floats; 0.37 in the first stunts - 37 %). Shadowed pixels are marked
// below it, so they come out at the ambient: dark. The PC settings raise
// the ambient for the table alone (the colour put back after): shadows=off
// lights everything fully, ambient_light=NN (a percentage) keeps at least
// that much light in the shadows.
constexpr u32 kAmbient = 0x005C2180u;

float AmbientWanted() {
    if (!ps2host::SettingOn("shadows", true)) return 1.0f;
    const std::string percent = ps2host::Setting("ambient_light", "");
    if (percent.empty()) return 0.0f;
    try {
        return std::stof(percent) / 100.0f;
    } catch (...) {
        return 0.0f;
    }
}

void LightTable(lifted::Context& c) {
    ps2host::Memory& m = ps2host::GuestMemory();
    static const float wanted = AmbientWanted();
    u64 saved[2] = {0, 0};
    if (wanted > 0) {
        saved[0] = m.Read(kAmbient, 8);
        saved[1] = m.Read(kAmbient + 8, 4);
        for (u32 i = 0; i < 3; ++i) {
            const u32 w = u32(m.Read(kAmbient + 4 * i, 4));
            float v;
            std::memcpy(&v, &w, 4);
            if (!(v >= wanted)) {
                u32 raised;
                std::memcpy(&raised, &wanted, 4);
                m.Write(kAmbient + 4 * i, raised, 4);
            }
        }
    }
    F_001D0DA0(c);
    if (wanted > 0) {
        m.Write(kAmbient, saved[0], 8);
        m.Write(kAmbient + 8, saved[1], 4);
    }
}

// The picture's wait (0x001D4FA8): for the blank counter (0x00272D70) to be
// N past the last picture's (0x00272D90), N the word 0x005C21B4 (2, from the
// game's configuration table, written by 0x00165220). The main loop
// (0x00119790) updates the game once per blank elapsed, at most 3 a picture,
// so N sets the pictures a second, not the game's speed: frame_rate=50 makes
// it 1 - a picture every blank of PAL's 50 (PeterDelta's "50 FPS" patch for
// PCSX2 writes the same word).
constexpr u32 kBlanksPerPicture = 0x005C21B4u;

void PictureWait(lifted::Context& c) {
    static const bool fifty = ps2host::Setting("frame_rate", "25") == "50";
    if (fifty) ps2host::GuestMemory().Write(kBlanksPerPicture, 1, 4);
    F_001D4FA8(c);
}

// widescreen=on: the scene seen at the picture's shape (aspect=: 16:9,
// 21:9, 32:9, the window's own with auto). The camera (0x001D4C88, every
// picture) passes its horizontal field of view in $f12 (60 degrees in the
// stunts) and its aspect in $f13 (4/3) to the projection (0x001D6440: x
// scale 256/tan(fov/2), y scale that times the aspect, what VU1 draws with)
// and to the view's clipping planes (0x001D6008); nothing else calls them.
// Both given tan(fov/2) and the aspect times the picture's over 4/3 (16/9:
// a third wider): the picture as tall as before (Hor+), the planes as wide.
// The display shows it at that shape; the 2D (the gauges, the menus) is
// stretched with it. The shape is asked every picture, so a window resized
// or made fullscreen with aspect=auto is followed.
bool Widescreen() {
    static const bool on = ps2host::SettingOn("widescreen", false);   // first asked by the camera, the settings read
    return on;
}

void Widen(lifted::Context& c) {
    if (!Widescreen()) return;
    float fov, aspect;
    std::memcpy(&fov, &c.f[12], 4);
    std::memcpy(&aspect, &c.f[13], 4);
    const float wider = ps2host::PictureAspect() / aspect;
    if (!(wider > 1.0f)) return;                // already that wide
    fov = 2.0f * std::atan(std::tan(fov * 0.5f) * wider);
    aspect *= wider;
    std::memcpy(&c.f[12], &fov, 4);
    std::memcpy(&c.f[13], &aspect, 4);
}

void WideProjection(lifted::Context& c) {
    Widen(c);
    F_001D6440(c);
}

void WideClipping(lifted::Context& c) {
    Widen(c);
    F_001D6008(c);
}

}  // namespace

void RegisterGameOverrides() {
    ps2host::Override(0x001D4FA8u, PictureWait, "the picture's wait: a picture every blank with frame_rate=50");
    ps2host::Override(0x001D0DA0u, LightTable, "the light table: shadows and ambient light from the PC settings");
    ps2host::Override(0x001D6440u, WideProjection, "the projection: the scene at the picture's shape with widescreen=on");
    ps2host::Override(0x001D6008u, WideClipping, "the view's clipping planes: as wide as the projection");
    // ps2host::Override(0x00123456u, SomeHook, "what it does and why");

    ps2host::Override(0x00253020u, GuestPrintf, "printf: the game's messages to the log as [guest] lines");
    ps2host::Override(0x002521E0u, PatchedMalloc, "malloc, as the game patches it: its own allocator");
    ps2host::Override(0x00251FE0u, PatchedMemalign, "memalign, as the game patches it: its own allocator");
    ps2host::Override(0x00252208u, PatchedFree, "free, as the game patches it: its own allocator");

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
