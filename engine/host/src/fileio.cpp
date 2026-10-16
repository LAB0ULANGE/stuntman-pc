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
    if (procedure == 0xFF) return std::vector<u8>(8, 0);   // the service's initialisation
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
    return {1, 0, 0, 0};
}

}  // namespace

void InstallFileIo() { RegisterRpcServer(0x80000001u, FileIo); }

}  // namespace ps2host
