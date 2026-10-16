/*
 * The CD/DVD services the IOP's file system module (cdvdfsv) provides to the EE,
 * as host RPC servers reading the user's disc image (disc.cpp).
 *
 * Each server's protocol is read from the game's own EE-side library, which
 * builds the request and reads the answer: Burnout 3's copy of the SDK's
 * libcdvd is at 0x0011E9A8-0x0011FD00. The servers are added one at a time, as
 * a game is seen to reach them.
 */

#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>
#include <vector>

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

std::vector<u8> Word(u32 value) {
    return {u8(value), u8(value >> 8), u8(value >> 16), u8(value >> 24)};
}

u32 At(const std::vector<u8>& data, std::size_t offset) {
    u32 value = 0;
    if (offset + 4 <= data.size()) std::memcpy(&value, &data[offset], 4);
    return value;
}

// sceCdDiskReady (server 0x8000059C, procedure 0): sent the wait mode, answers
// the drive's state in one word - 2, SCECdComplete, when a disc is ready, and 6,
// SCECdNotReady, when there is none.
std::vector<u8> DiskReady(u32 /*procedure*/, const std::vector<u8>& /*data*/) {
    return Word(GuestDisc().IsOpen() ? 2 : 6);
}

// sceCdSearchFile (server 0x80000597, procedure 0). The request (0x12C bytes):
//   0x000  sceCdlFILE { u32 lsn; u32 size; char name[16]; u8 date[8]; }
//   0x024  the path, 256 bytes, NUL-terminated: "\\DATA\\CARS.BIN;1"
//   0x124  the EE address of this buffer, where the answer's file record goes
// The answer word is 1 when the file was found, 0 when not; the library then
// copies the record from the buffer to its caller.
std::vector<u8> SearchFile(u32 /*procedure*/, const std::vector<u8>& data) {
    if (data.size() < 0x128) return Word(0);
    const char* begin = reinterpret_cast<const char*>(&data[0x24]);
    const std::string path(begin, strnlen(begin, 256));
    Disc::File file;
    const bool found = GuestDisc().IsOpen() && GuestDisc().Find(path, &file);
    Note("cd search " + path, "CD: search " + path + (found ? "" : " (not found)"));
    if (!found) return Word(0);
    const u32 record = At(data, 0x124);
    Memory& m = GuestMemory();
    m.Write(record, file.lsn, 4);
    m.Write(record + 4, file.size, 4);
    // The name, as the directory has it: the path's last part.
    const std::size_t slash = path.find_last_of("\\/");
    const std::string name = path.substr(slash == std::string::npos ? 0 : slash + 1);
    for (u32 i = 0; i < 16; ++i)
        m.Write(record + 8 + i, i < name.size() ? u8(name[i]) : 0, 1);
    for (u32 i = 0; i < 8; ++i) m.Write(record + 24 + i, 0, 1);
    return Word(1);
}

u8 Bcd(int value) { return u8(((value / 10) << 4) | (value % 10)); }

// S-commands (server 0x80000593): the procedure is the command; each answers a
// result word, 1 for success, and the command's data after it. Read from the
// game's wrappers, each of which names its command and reply size.
std::vector<u8> SCommand(u32 procedure, const std::vector<u8>& /*data*/) {
    switch (procedure) {
        case 0x01: {                                    // sceCdReadClock: result, sceCdCLOCK
            // With --virtual-time the clock starts at a fixed date and follows
            // the virtual clock: games seed their random choices from it
            // (Burnout 3 its EA Trax playlist), and a scripted run must repeat.
            std::tm local{};
            if (VirtualClock()) {
                const std::time_t start = 1094558400;   // 2004-09-07 12:00:00 UTC, Burnout 3's release
                const std::time_t now = start + std::time_t(ClockNs() / 1000000000ull);
                gmtime_s(&local, &now);
            } else {
                const std::time_t now = std::time(nullptr);
                localtime_s(&local, &now);
            }
            std::vector<u8> reply = Word(1);
            // sceCdCLOCK { stat, second, minute, hour, pad, day, month, year }, BCD
            const u8 clock[8] = {0, Bcd(local.tm_sec), Bcd(local.tm_min), Bcd(local.tm_hour), 0,
                                 Bcd(local.tm_mday), Bcd(local.tm_mon + 1),
                                 Bcd(local.tm_year % 100)};
            reply.insert(reply.end(), clock, clock + 8);
            return reply;
        }
        case 0x03:                                      // sceCdGetDiskType
            return Word(!GuestDisc().IsOpen() ? 0x00 : GuestDisc().IsCd() ? 0x12 : 0x14);  // none, SCECdPS2CD, SCECdPS2DVD
        case 0x05: {                                    // sceCdTrayReq: result, tray state
            std::vector<u8> reply = Word(1);
            const std::vector<u8> closed = Word(0);
            reply.insert(reply.end(), closed.begin(), closed.end());
            return reply;
        }
        case 0x22:                                      // sceCdMmode: the media mode is set
            return Word(1);
        case 0x04:                                      // sceCdGetError: SCECdErNO
            // SDK 2.3's library reads one word, the error (Mister Mosquito,
            // 0x0010D9E0); every read here succeeds.
            return Word(0);
        default:
            Note("cd scmd " + std::to_string(procedure),
                 "CD: S-command " + std::to_string(procedure) + " (answered with zeros)");
            return {};
    }
}

// sceCdInit (server 0x80000592, procedure 0): sent the init mode; the drive is
// ready at once.
std::vector<u8> CdInit(u32 /*procedure*/, const std::vector<u8>& /*data*/) { return Word(1); }

// N-commands (server 0x80000595): the procedure is the command. Reads finish at
// once - the sectors are in EE memory when the call's end arrives.
//   1  sceCdRead: { u32 lsn, sectors, EE buffer, mode (trycount, spindlctrl,
//      datapattern, -), u32 EE address of the library's unaligned-ends record,
//      - }. cdvdfsv sends a buffer's 64-byte-aligned middle by DMA and its
//      ragged ends through that record { u32 length1, length2, dest1, dest2,
//      u8 bytes1[64], bytes2[64] }, which the library's completion copies out:
//      here every byte goes straight to the buffer and the record says none.
//  14  sceCdDiskReady (N): 2, SCECdComplete.
// How long a read takes the drive: a 24x CD's 3.6 MB/s, after a seek. Games
// time things by it - Mister Mosquito's stage loader moves a sound bank a
// jingle still plays from, which the seconds its reads take let end first.
u64 ReadTimeNs(std::size_t bytes) {
    constexpr u64 kSeekNs = 2000000, kBytesPerSecond = 3686400;
    return kSeekNs + u64(bytes) * 1000000000ull / kBytesPerSecond;
}

std::vector<u8> NCommand(u32 procedure, const std::vector<u8>& data) {
    Memory& m = GuestMemory();
    switch (procedure) {
        case 1: {
            if (data.size() < 20) return Word(0);
            const u32 lsn = At(data, 0), sectors = At(data, 4), buffer = At(data, 8);
            const u32 pattern = (At(data, 12) >> 16) & 0xFF, ends = At(data, 16);
            std::vector<u8> bytes(std::size_t(sectors) * Disc::PatternSize(pattern));
            const bool ok = GuestDisc().IsOpen() &&
                            GuestDisc().ReadPattern(lsn, sectors, pattern, bytes.data());
            char text[160];
            std::snprintf(text, sizeof text, "CD: read %u sectors at %u to 0x%08X (pattern %u, ends 0x%08X)%s",
                          sectors, lsn, buffer, pattern, ends, ok ? "" : " - past the disc's end");
            Note("cd read " + std::to_string(lsn) + " " + std::to_string(buffer), text);
            if (!ok) return Word(0);
            for (std::size_t i = 0; i < bytes.size(); ++i) m.Write(buffer + u32(i), bytes[i], 1);
            if (ends)
                for (u32 k = 0; k < 16; ++k) m.Write(ends + k, 0, 1);
            DelayRpcReply(ReadTimeNs(bytes.size()));
            return Word(1);
        }
        case 13: {
            // sceCdReadIOPMem: { lsn, sectors, IOP buffer, mode }: the sectors
            // into IOP memory, for a module there.
            if (data.size() < 12) return Word(0);
            const u32 lsn = At(data, 0), sectors = At(data, 4), buffer = At(data, 8) & 0x1FFFFF;
            const u32 pattern = data.size() >= 16 ? (At(data, 12) >> 16) & 0xFF : 0;
            const u32 size = sectors * Disc::PatternSize(pattern);
            std::vector<u8> bytes(size);
            const bool ok = buffer + size <= kIopRamSize && GuestDisc().IsOpen() &&
                            GuestDisc().ReadPattern(lsn, sectors, pattern, bytes.data());
            char text[160];
            std::snprintf(text, sizeof text, "CD: read %u sectors at %u into IOP memory at 0x%06X%s",
                          sectors, lsn, buffer, ok ? "" : " - failed");
            Note("cd read iop " + std::to_string(lsn), text);
            if (!ok) return Word(0);
            for (u32 i = 0; i < size; ++i) m.Write(kIopRamBase + buffer + i, bytes[i], 1);
            DelayRpcReply(ReadTimeNs(size));
            return Word(1);
        }
        case 14:
            return Word(2);
        default:
            Note("cd ncmd " + std::to_string(procedure),
                 "CD: N-command " + std::to_string(procedure) + " (answered with zeros)");
            return {};
    }
}

}  // namespace

void InstallIopServers() {
    InstallFileIo();
    InstallLoadFile();
    InstallDbc();
    InstallIopHeap();
    InstallMcserv();
    RegisterRpcServer(0x8000059Cu, DiskReady);
    RegisterRpcServer(0x8000059Au, DiskReady);      // the same, as SDK 2.3's libcdvd binds it
    RegisterRpcServer(0x80000592u, CdInit);
    RegisterRpcServer(0x80000595u, NCommand);
    RegisterRpcServer(0x80000597u, SearchFile);
    RegisterRpcServer(0x80000593u, SCommand);
}

}  // namespace ps2host
