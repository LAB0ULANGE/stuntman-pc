#include "ps2host/memory.h"

#include <cstdio>
#include <cstring>

#include "ps2host/run.h"
#include "ps2host/vu.h"
#include "ps2rt/generated.h"

namespace ps2host {

namespace {

// Hooks are indexed by register word: the EE window's words, then the GS's.
constexpr u32 kEeWords = kEeRegisterSize / 4;
constexpr u32 kGsWords = kGsRegisterSize / 4;

// kseg0 and kseg1 addresses, as the physical address they reach.
u32 Physical(u32 address) {
    return (address >= 0x80000000u && address < 0xC0000000u) ? address & 0x1FFFFFFFu : address;
}

int HookIndex(u32 address) {
    if (address - kEeRegisterBase < kEeRegisterSize) return (address - kEeRegisterBase) / 4;
    if (address - kGsRegisterBase < kGsRegisterSize)
        return kEeWords + (address - kGsRegisterBase) / 4;
    return -1;
}

}  // namespace

Memory::Memory()
    : rdram_(kRdramSize), scratchpad_(kScratchpadSize), ee_registers_(kEeRegisterSize),
      vu_memory_(kVuMemorySize), gs_registers_(kGsRegisterSize), iop_ram_(kIopRamSize),
      bios_(kBiosSize),
      read_hooks_(kEeWords + kGsWords), write_hooks_(kEeWords + kGsWords) {}

u8* Memory::Translate(u32 address, u32 size) {
    // kseg0 (0x80000000) and kseg1 (0xA0000000) see the whole low 512 MB of the
    // physical address space - RDRAM and the device registers alike.
    address = Physical(address);
    const u32 last = address + size - 1;
    // RDRAM and its user-mode mirrors: the top bits select the window, the rest
    // the byte.
    switch (address >> 28) {
        case 0x0: case 0x2: case 0x3: {
            const u32 offset = address & 0x0FFFFFFFu;
            if (offset < kRdramSize && (last & 0x0FFFFFFFu) < kRdramSize)
                return rdram_.data() + offset;
            break;
        }
        default:
            break;
    }
    // The other windows, by the address's top byte (each lies within one):
    // the scratchpad and the device registers are reached a great deal.
    auto in = [&](u32 base, u32 size, std::vector<u8>& bytes) -> u8* {
        return address - base < size && last - base < size ? bytes.data() + (address - base) : nullptr;
    };
    switch (address >> 24) {
        case kScratchpadBase >> 24: return in(kScratchpadBase, kScratchpadSize, scratchpad_);
        case kEeRegisterBase >> 24: return in(kEeRegisterBase, kEeRegisterSize, ee_registers_);
        case kVuMemoryBase >> 24: return in(kVuMemoryBase, kVuMemorySize, vu_memory_);
        case kGsRegisterBase >> 24: return in(kGsRegisterBase, kGsRegisterSize, gs_registers_);
        case 0x1C: case 0x1D: return in(kIopRamBase, kIopRamSize, iop_ram_);
        case 0x1F: return in(kBiosBase, kBiosSize, bios_);
        default: return nullptr;
    }
}

namespace {
/* Registers and memory VIF1's worker changes or reads (vif1_worker.cpp):
   GIF and VIF1 registers and FIFOs, VU1's memories (not VU0's: the EE uses
   them all the time). Not the DMA channels and the DMAC's D_STAT: a transfer
   ends on the EE's thread, the worker running on from a copy of its data.
   Of the GIF's registers only GIF_STAT, the one the worker changes; VIF1's
   only written: VIF1's model keeps its state to itself, and what a read
   finds there is the same with the worker done or not. */
bool Vif1Sees(u32 a, bool write) {
    return (a - 0x10003020u < 4u) || (write && a - 0x10003C00u < 0x400u) || (a - 0x10005000u < 0x2000u) ||
           (a - 0x11008000u < 0x8000u);
}
}  // namespace

u64 Memory::Read(u32 address, u32 size) {
    address = Physical(address);
    // GIF_STAT, while VIF1's worker has work: what it will read once the
    // worker is done, worked out rather than waited for (Vif1Guess).
    const bool status = Vif1Guessable(address, size) && !OnVif1Worker();
    if (Vif1Busy() && Vif1Sees(address, false)) {
        u32 guess = 0;
        if (status && Vif1Guess(address, &guess)) return guess;
        JoinVif1("read", address);
    }
    u8* bytes = Translate(address, size);
    if (!bytes) throw MemoryFault{address, false};
    u64 value = 0;
    std::memcpy(&value, bytes, size);        // the EE and the host are both little-endian
    const int index = HookIndex(address & ~3u);
    if (index >= 0) {
        if (!OnVif1Worker()) RunStats().register_reads[address & ~3u]++;
        if (read_hooks_[index]) value = read_hooks_[index](address, value);
    }
    if (status) Vif1Settled(address, u32(value));     // read with the queue run
    return value;
}

void Memory::WatchWrites(const std::vector<u32>& addresses) {
    watched_ = addresses;
    generated::ram_watched = !addresses.empty();     // writes come here to be seen
}

void Memory::Watched(u32 address, u64 value, u32 size) {
    const std::vector<u32>& calls = CurrentCalls();
    const u32 in = calls.empty() ? 0 : calls.back();
    // The callers too, innermost first: the function writing is often a helper.
    char chain[128] = "";
    for (std::size_t i = 1, n = 0; i < calls.size() && i <= 10; ++i)
        n += std::snprintf(chain + n, sizeof chain - n, " < %08X", unsigned(calls[calls.size() - 1 - i]));
    std::fprintf(stderr, "[write] 0x%08X <- 0x%llX (%u bytes) in 0x%08X%s\n", unsigned(address),
                 static_cast<unsigned long long>(value), unsigned(size), unsigned(in), chain);
}

void Memory::Write(u32 address, u64 value, u32 size) {
    address = Physical(address);
    // VIF1's FIFO excepted: a quadword stored there goes to the worker in turn (devices.cpp).
    if (Vif1Busy() && Vif1Sees(address, true) && address - 0x10005000u >= 0x1000u) JoinVif1("write", address);
    // Into VU0's or VU1's instruction memory: their compiled programs are
    // compared with it again (Vu::MicroWritten).
    if (address - 0x11000000u < 0x10000u && ((address >> 14) & 1) == 0) {
        if (address < 0x11008000u) Vu0().MicroWritten();
        else Vu1().MicroWritten();
    }
    if (!watched_.empty())
        for (u32 w : watched_)
            if (w >= address && w < address + size) Watched(address, value, size);
    u8* bytes = Translate(address, size);
    if (!bytes) throw MemoryFault{address, true};
    if (address >= kBiosBase && address - kBiosBase < kBiosSize) return;   // ROM
    std::memcpy(bytes, &value, size);
    // A write wider than a word reaches each register word it covers, in order.
    for (u32 offset = 0; offset < size; offset += 4) {
        const u32 word = (address & ~3u) + offset;
        const int index = HookIndex(word);
        if (index < 0) continue;
        if (!OnVif1Worker()) RunStats().register_writes[word]++;
        if (write_hooks_[index]) write_hooks_[index](word, offset ? value >> (8 * offset) : value);
    }
}

void Memory::OnRead(u32 address, ReadHook hook) {
    const int index = HookIndex(address & ~3u);
    if (index >= 0) read_hooks_[index] = std::move(hook);
}

void Memory::OnWrite(u32 address, WriteHook hook) {
    const int index = HookIndex(address & ~3u);
    if (index >= 0) write_hooks_[index] = std::move(hook);
}

Memory& GuestMemory() {
    static Memory memory;
    return memory;
}

// Main RAM for the derived code's inline loads and stores (generated.h).
namespace {
const bool g_ram_given = (generated::ram_base = GuestMemory().rdram(),
                          generated::scratchpad_base = GuestMemory().scratchpad(), true);
}  // namespace

}  // namespace ps2host

// What generated and lifted code reads and writes outside main RAM (main RAM
// itself it reaches inline, generated.h), through the devices' hooks.
namespace generated {

u8* ram_base = nullptr;
bool ram_watched = false;
u8* scratchpad_base = nullptr;

u64 PortRead(u32 address, u32 size, bool is_signed) {
    u64 value = ps2host::GuestMemory().Read(address, size);
    if (is_signed && size < 8) {
        const int shift = 64 - 8 * int(size);
        value = u64(s64(value << shift) >> shift);
    }
    return value;
}

void PortWrite(u32 address, u64 value, u32 size) { ps2host::GuestMemory().Write(address, value, size); }

}  // namespace generated
