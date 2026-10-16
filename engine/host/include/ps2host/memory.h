/*
 * Guest memory: the Emotion Engine's address space, as the game sees it.
 *
 *   RDRAM        32 MB at physical 0, seen through the kuseg (0x00000000),
 *                uncached (0x20000000), uncached-accelerated (0x30000000),
 *                kseg0 (0x80000000) and kseg1 (0xA0000000) windows
 *   scratchpad   16 KB at 0x70000000
 *   EE registers 64 KB at 0x10000000: timers, IPU, GIF, VIF, DMA, INTC, SIF
 *   VU memory    64 KB at 0x11000000: VU0/VU1 micro and data memory
 *   GS registers 8 KB at 0x12000000, the GS's privileged registers
 *   IOP RAM      2 MB at 0x1C000000, the I/O processor's memory as the EE sees it
 *   BIOS         4 MB at 0x1FC00000, read-only and zero unless supplied
 *
 * Everything else faults. Register windows are plain memory by default; a
 * Device can claim a register to give reads and writes their hardware meaning
 * (a DMA channel completing, the GS reporting a vertical blank).
 */

#ifndef PS2HOST_MEMORY_H
#define PS2HOST_MEMORY_H

#include <cstddef>
#include <functional>
#include <vector>

#include "ps2rt/types.h"

namespace ps2host {

constexpr u32 kRdramSize = 32u << 20;
constexpr u32 kScratchpadBase = 0x70000000u, kScratchpadSize = 16u << 10;
constexpr u32 kEeRegisterBase = 0x10000000u, kEeRegisterSize = 64u << 10;
constexpr u32 kVuMemoryBase = 0x11000000u, kVuMemorySize = 64u << 10;
constexpr u32 kGsRegisterBase = 0x12000000u, kGsRegisterSize = 8u << 10;
constexpr u32 kIopRamBase = 0x1C000000u, kIopRamSize = 2u << 20;
constexpr u32 kBiosBase = 0x1FC00000u, kBiosSize = 4u << 20;

/* A guest access that hits no memory. */
struct MemoryFault {
    u32 address;
    bool write;
};

class Memory {
public:
    Memory();

    /* Where `size` bytes at `address` live, or null if they span no memory. */
    u8* Translate(u32 address, u32 size);

    u64 Read(u32 address, u32 size);
    void Write(u32 address, u64 value, u32 size);

    /* Device registers: a read hook returns the value to read; a write hook
       receives the value written (already stored as plain memory). Hooks are
       per 32-bit word, in the EE and GS register windows. */
    using ReadHook = std::function<u64(u32 address, u64 stored)>;
    using WriteHook = std::function<void(u32 address, u64 value)>;
    void OnRead(u32 address, ReadHook hook);
    void OnWrite(u32 address, WriteHook hook);

    u8* rdram() { return rdram_.data(); }
    u8* scratchpad() { return scratchpad_.data(); }

    /* Log every store that touches one of these guest addresses (--watch-write),
       with the value and the function storing it. */
    void WatchWrites(const std::vector<u32>& addresses);
    /* Whether any store is being watched: if not, a store to main RAM or the
       scratchpad (no register hooks there) may go straight through Translate. */
    bool watching() const { return !watched_.empty(); }

private:
    std::vector<u8> rdram_, scratchpad_, ee_registers_, vu_memory_, gs_registers_, iop_ram_, bios_;
    std::vector<ReadHook> read_hooks_;      /* per register word, both windows */
    std::vector<WriteHook> write_hooks_;
    std::vector<u32> watched_;               /* physical addresses */
    void Watched(u32 address, u64 value, u32 size);
};

/* The one guest address space; the generated Read/Write hooks use it. */
Memory& GuestMemory();

}  // namespace ps2host

#endif  /* PS2HOST_MEMORY_H */
