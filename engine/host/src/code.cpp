/*
 * The program's code, against what the program makes of it.
 *
 * A game can patch its own instructions: Stuntman writes jumps over the
 * calls inside newlib's malloc, memalign and free at start-up, sending them to
 * its own allocator. A console runs the patched code; the translated code is
 * the executable's as it was on the disc, and does not follow. Nothing in the
 * run says so - until a pointer one allocator gave is handed to the other.
 *
 * So the executable sections are kept as loaded and compared with memory: at
 * the first frame (a game patches itself while starting), then for the
 * report. Each word found changed is logged once, with the function that
 * stored it: stores into the code are watched until the first frame (they go
 * the slow way meanwhile); one made later is named by --watch-write.
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ps2host/host.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

struct CodeSection {
    u32 address;
    std::vector<u8> bytes;
};
std::vector<CodeSection> g_sections;
std::map<u32, u32> g_logged;                 // word -> the value last logged changed
bool g_watching = false;

std::string Named(u32 address) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%08X", unsigned(address));
    const char* name = FunctionName(address);
    return name ? std::string(text) + " " + name : text;
}

// A jump's target (j, jal: the patches a game makes), or the word.
std::string Instruction(u32 address, u32 word) {
    char text[32];
    const u32 op = word >> 26;
    if (op == 2 || op == 3)
        std::snprintf(text, sizeof text, "%s 0x%08X", op == 2 ? "j" : "jal",
                      unsigned(((address + 4) & 0xF0000000u) | ((word & 0x03FFFFFFu) << 2)));
    else
        std::snprintf(text, sizeof text, "0x%08X", unsigned(word));
    return text;
}

std::string Line(u32 address, u32 was, u32 now) {
    char head[96];
    std::snprintf(head, sizeof head, "code changed by the program at 0x%08X: %08X -> %08X", unsigned(address),
                  unsigned(was), unsigned(now));
    std::string line = std::string(head) + " (" + Instruction(address, was) + " -> " + Instruction(address, now) + ")";
    const std::pair<u32, u32> writer = GuestMemory().CodeWriter(address);
    if (writer.first)
        line += ", stored by " + Named(writer.first) + (writer.second ? " < " + Named(writer.second) : "");
    else
        line += ", stored after the first frame (--watch-write " + Named(address).substr(0, 10) + " names who)";
    return line + "; the translated code still runs the old instruction - TO RECHECK";
}

}  // namespace

void KeepCode(u32 address, const u8* bytes, u32 size) {
    g_sections.push_back({address, std::vector<u8>(bytes, bytes + size)});
    u32 start = address, end = address + size;
    for (const CodeSection& s : g_sections) {
        start = std::min(start, s.address);
        end = std::max(end, s.address + u32(s.bytes.size()));
    }
    GuestMemory().WatchCode(start, end);
    g_watching = true;
}

std::vector<std::string> CheckCode() {
    std::vector<std::string> lines;
    for (const CodeSection& s : g_sections) {
        const u32 size = u32(s.bytes.size());
        const u8* now = GuestMemory().Translate(s.address, size);
        if (!now || std::memcmp(now, s.bytes.data(), size) == 0) continue;
        for (u32 at = 0; at + 4 <= size; at += 4) {
            u32 was, is;
            std::memcpy(&was, s.bytes.data() + at, 4);
            std::memcpy(&is, now + at, 4);
            if (was == is) continue;
            auto logged = g_logged.find(s.address + at);
            if (logged != g_logged.end() && logged->second == is) continue;
            g_logged[s.address + at] = is;
            lines.push_back(Line(s.address + at, was, is));
        }
    }
    return lines;
}

void CodeFrame(u64 frame) {
    (void)frame;
    if (!g_watching) return;
    for (const std::string& line : CheckCode()) Note(line, line);
    GuestMemory().WatchCode(0, 0);
    g_watching = false;
}

std::string CodeReport() {
    for (const std::string& line : CheckCode()) Note(line, line);
    std::string text;
    for (const CodeSection& s : g_sections) {
        const u8* now = GuestMemory().Translate(s.address, u32(s.bytes.size()));
        if (!now) continue;
        for (u32 at = 0; at + 4 <= s.bytes.size(); at += 4) {
            u32 was, is;
            std::memcpy(&was, s.bytes.data() + at, 4);
            std::memcpy(&is, now + at, 4);
            if (was != is) text += "  " + Line(s.address + at, was, is) + "\n";
        }
    }
    return text;
}

void ForgetCode() {
    g_sections.clear();
    g_logged.clear();
    GuestMemory().WatchCode(0, 0);
    g_watching = false;
}

}  // namespace ps2host
