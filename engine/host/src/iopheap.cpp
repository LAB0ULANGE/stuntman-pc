/*
 * The IOP's heap service (server 0x80000003, iopheap), on the host: the EE
 * allocating IOP memory, for buffers it then hands to IOP modules by address.
 * It shares the IOP interpreter's allocator (iop_vm.h), so what the EE gets is
 * memory no module the host runs is using.
 *
 * The protocol, as Burnout 3's EE-side library (0x00113120-0x0011320C) speaks it:
 *   procedure 1  allocate: size at +0                       -> address (0: none)
 *   procedure 2  free: address at +0                        -> 0
 *   procedure 4  allocate as sysmem does: size at +0, mode at +4, address at +8
 *                                                         -> address (0: none)
 *   procedure 3  sceSifLoadIopHeap: a file into IOP memory: the address at +0,
 *                the path at +4 ("cdrom0:\SOUND\SE.BIN;1", 252 bytes)
 *                                                         -> 0 (< 0: failed)
 *                (Mister Mosquito's library, 0x00109D80)
 * Any other procedure is noted and fails.
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/iop_vm.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

u32 At(const std::vector<u8>& data, std::size_t offset) {
    u32 value = 0;
    if (offset + 4 <= data.size()) std::memcpy(&value, &data[offset], 4);
    return value;
}

std::vector<u8> Word(u32 value) {
    return {u8(value), u8(value >> 8), u8(value >> 16), u8(value >> 24)};
}

std::vector<u8> IopHeap(u32 procedure, const std::vector<u8>& data) {
    char text[320];
    switch (procedure) {
        case 1:
        case 4: {
            const u32 size = At(data, 0), address = iop::Allocate(size);
            std::snprintf(text, sizeof text, "IOP heap: %u bytes at 0x%08X for the EE",
                          unsigned(size), unsigned(address));
            Note("iopheap alloc " + std::to_string(address), text);
            return Word(address);
        }
        case 2:
            iop::Free(At(data, 0));
            return Word(0);
        case 3: {
            const u32 address = At(data, 0);
            std::string path;
            for (std::size_t i = 4; i < data.size() && data[i]; ++i) path += char(data[i]);
            const std::size_t colon = path.find(':');
            const std::string on_disc = colon == std::string::npos ? path : path.substr(colon + 1);
            std::vector<u8> bytes;
            const bool ok = GuestDisc().IsOpen() && GuestDisc().ReadFile(on_disc, &bytes);
            std::snprintf(text, sizeof text, "IOP heap: %s loaded at 0x%08X (%u bytes)%s", path.c_str(),
                          unsigned(address), unsigned(bytes.size()), ok ? "" : " - not on the disc");
            Note("iopheap load " + path, text);
            if (!ok) return Word(0xFFFFFFFFu);
            Memory& m = GuestMemory();
            for (std::size_t i = 0; i < bytes.size(); ++i) m.Write(kIopRamBase + address + u32(i), bytes[i], 1);
            return Word(0);
        }
        default:
            Note("iopheap " + std::to_string(procedure),
                 "IOP: iopheap procedure " + std::to_string(procedure) + " is not implemented (failed)");
            return Word(0xFFFFFFFFu);
    }
}

}  // namespace

void InstallIopHeap() { RegisterRpcServer(0x80000003u, IopHeap); }

}  // namespace ps2host
