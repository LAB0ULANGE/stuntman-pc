/*
 * The IOP's file I/O service (server 0x80000001), on the host: files on the game
 * disc ("cdrom0:\\DATA\\CARS.BIN;1") read from the user's disc image.
 *
 * The protocol, as Burnout 3's EE-side library (0x0010FFA0-0x00112F9C) speaks
 * it: every request starts with
 *   0x00  a semaphore the EE created for this request
 *   0x04  the EE address of the result word
 *   0x08  the result's size (4)
 * then the procedure's arguments from 0x0C. The RPC's own answer only
 * acknowledges the request (nonzero); the IOP does the work, writes the result
 * and signals the semaphore, and the EE waits on that. Here the work is done at
 * once, so the semaphore is signalled before the EE gets to wait on it.
 *
 *   procedure 0  open   flags 0x0C, mode 0x10, path 0x14 (1,024 bytes) -> descriptor
 *   procedure 1  close  descriptor 0x0C                                -> 0
 *   procedure 2  read   descriptor 0x0C, EE buffer 0x10, size 0x14      -> bytes read
 *   procedure 4  lseek  descriptor 0x0C, offset 0x10, whence 0x14       -> position
 *
 * Any other procedure is noted and fails (-1); they are added as games need them.
 *
 * Procedure 0xFF is the library's initialisation, answered directly (no result
 * word, no semaphore): the module's release, 4 characters, as loadfile answers
 * its own. Stuntman's library (0x002602E8) keeps it and refuses every open
 * until it is the release it was built for ("2430", its IOPRP243.IMG's
 * "fileio  2430" tag) - with zeros it retried its first open for ever. Before
 * any reboot the release is the ROM-booted IOP's, taken as loadfile's ("2800").
 * TO RECHECK: this answer was eight zero bytes until Stuntman; run
 * engine/tools/regress_all.py on burnout3 once it is at hand.
 * Its request's first word is the EE address of the library's completion
 * record (Stuntman: 0x002AE580).
 *
 * An asynchronous request (a descriptor opened "no wait") carries its
 * semaphore negated, and the EE does not wait on it: it keeps the request in a
 * table and polls that (Stuntman's 0x002610E8, table at 0x00279DE8). What ends
 * it is the module's completion, not the RPC: the record written to the EE -
 *   0x00  the semaphore as sent      0x08  the result's address
 *   0x04  the procedure              0x0C  the result's size
 *   0x10  the result                 0x14  per procedure (read: the unaligned
 *                                          head and tail, none here)
 * - then SIF command 0x80000011, whose EE handler (Stuntman's 0x0025FDD8) frees
 * the table's entry. A synchronous request is answered as before: its result
 * word and its semaphore.
 */

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

constexpr s32 kNoSuchFile = -2, kBadDescriptor = -9;

char g_release[4] = {'2', '8', '0', '0'};   // until a reboot image says otherwise
u32 g_completion = 0;                       // the EE's completion record (procedure 0xFF)
constexpr u32 kCompletionCommand = 0x80000011u;
constexpr u32 kCompletionSize = 0xA4;       // what the library writes back from the cache

struct OpenFile {
    Disc::File file;
    u32 position = 0;
};

std::map<s32, OpenFile>& Files() {
    static std::map<s32, OpenFile> files;
    return files;
}

u32 At(const std::vector<u8>& data, std::size_t offset) {
    u32 value = 0;
    if (offset + 4 <= data.size()) std::memcpy(&value, &data[offset], 4);
    return value;
}

s32 Open(const std::vector<u8>& data) {
    if (data.size() < 0x14) return kNoSuchFile;
    const char* begin = reinterpret_cast<const char*>(&data[0x14]);
    const std::string path(begin, strnlen(begin, std::min<std::size_t>(1024, data.size() - 0x14)));
    // "cdrom0:\\DATA\\CARS.BIN;1": the device, then the path on it.
    const std::size_t colon = path.find(':');
    const std::string device = colon == std::string::npos ? "" : path.substr(0, colon);
    const std::string rest = colon == std::string::npos ? path : path.substr(colon + 1);
    Disc::File file;
    if (device.rfind("cdrom", 0) != 0 || !GuestDisc().IsOpen() || !GuestDisc().Find(rest, &file)) {
        Note("open " + path, "file I/O: open " + path + " (not found)");
        return kNoSuchFile;
    }
    static s32 next = 3;
    Files()[next] = OpenFile{file, 0};
    Note("open " + path, "file I/O: open " + path);
    return next++;
}

s32 Read(const std::vector<u8>& data) {
    auto found = Files().find(s32(At(data, 0x0C)));
    if (found == Files().end()) return kBadDescriptor;
    OpenFile& f = found->second;
    const u32 buffer = At(data, 0x10);
    u32 size = At(data, 0x14);
    size = std::min(size, f.file.size > f.position ? f.file.size - f.position : 0u);
    // Whole sectors from the disc, then the bytes asked for out of them.
    const u32 first = f.position / kSectorSize, last = (f.position + size + kSectorSize - 1) / kSectorSize;
    std::vector<u8> sectors(std::size_t(last - first) * kSectorSize);
    if (size && !GuestDisc().Read(f.file.lsn + first, last - first, sectors.data())) return -5;
    Memory& m = GuestMemory();
    const u32 skip = f.position - first * kSectorSize;
    if (u8* out = m.Translate(buffer, size); out && size) {
        std::memcpy(out, sectors.data() + skip, size);
    } else {
        for (u32 i = 0; i < size; ++i) m.Write(buffer + i, sectors[skip + i], 1);
    }
    f.position += size;
    return s32(size);
}

s32 Seek(const std::vector<u8>& data) {
    auto found = Files().find(s32(At(data, 0x0C)));
    if (found == Files().end()) return kBadDescriptor;
    OpenFile& f = found->second;
    const s32 offset = s32(At(data, 0x10));
    const u32 whence = At(data, 0x14);
    const s64 base = whence == 0 ? 0 : whence == 1 ? s64(f.position) : s64(f.file.size);
    const s64 position = std::max<s64>(0, base + offset);
    f.position = u32(position);
    return s32(position);
}

std::vector<u8> FileIo(u32 procedure, const std::vector<u8>& data) {
    if (procedure == 0xFF) {                                // the service's initialisation
        g_completion = At(data, 0);
        return FileIoVersionAnswer();
    }
    s32 result;
    switch (procedure) {
        case 0: result = Open(data); break;
        case 1: result = Files().erase(s32(At(data, 0x0C))) ? 0 : kBadDescriptor; break;
        case 2: result = Read(data); break;
        case 4: result = Seek(data); break;
        default:
            Note("fileio " + std::to_string(procedure),
                 "file I/O: procedure " + std::to_string(procedure) + " (not implemented: fails)");
            result = -1;
            break;
    }
    // The result where the EE asked for it, then its semaphore: the work is done.
    if (const u32 where = At(data, 0x04)) GuestMemory().Write(where, u32(result), 4);
    const s32 semaphore = s32(At(data, 0x00));
    if (semaphore > 0) SignalSemaphore(u32(semaphore));
    if (semaphore < 0 && g_completion) {
        // An asynchronous request: its completion, after the RPC's answer.
        std::vector<u8> record(kCompletionSize, 0);
        const u32 words[5] = {u32(semaphore), procedure, At(data, 0x04), At(data, 0x08), u32(result)};
        std::memcpy(record.data(), words, sizeof words);
        SendCmdToEe(kCompletionCommand, {}, g_completion, record);
    } else if (semaphore < 0) {
        Note("fileio async", "file I/O: an asynchronous request before the library's initialisation "
                             "(no completion record to write)");
    }
    return {1, 0, 0, 0};
}

}  // namespace

void SetFileIoRelease(const char release[4]) { std::memcpy(g_release, release, 4); }

std::vector<u8> FileIoVersionAnswer() {
    std::vector<u8> answer(8, 0);
    std::memcpy(answer.data(), g_release, 4);
    return answer;
}

void InstallFileIo() { RegisterRpcServer(0x80000001u, FileIo); }

}  // namespace ps2host
