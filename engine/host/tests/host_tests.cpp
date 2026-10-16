/*
 * The host's own tests, independent of any game: built with every port and run
 * by engine/tools/port.py before the port itself.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ps2host/disc.h"
#include "ps2host/dma.h"
#include "ps2host/gs.h"
#include "ps2host/ipu.h"
#include "iop_vm_internal.h"
#include "spu2.h"
#include "ps2host/mc2.h"
#include "ps2host/memory.h"
#include "ps2host/vif.h"
#include "ps2host/path3.h"
#include "ps2host/run.h"
#include "ps2host/texture_pack.h"
#include "ps2host/vu.h"
#include "ps2host/vu_core.h"
#include "ps2rt/generated.h"

#if PS2HOST_VULKAN
#include "ps2gpu/gs_renderer.h"
#endif

/* vu_corpus.cpp (vurecomp.py --corpus, as the build goes): the random programs, compiled. */
const ps2host::VuCompiled* VuGeneratedPrograms(std::size_t* count);

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

void MirrorsReachTheSameRam() {
    ps2host::Memory& m = ps2host::GuestMemory();
    m.Write(0x00100000u, 0x11223344u, 4);
    Check(m.Read(0x20100000u, 4) == 0x11223344u, "uncached mirror");
    Check(m.Read(0x30100000u, 4) == 0x11223344u, "uncached-accelerated mirror");
    Check(m.Read(0x80100000u, 4) == 0x11223344u, "kseg0");
    Check(m.Read(0xA0100000u, 4) == 0x11223344u, "kseg1");
}

void LoadsExtendAsTheEeDoes() {
    generated::Write32(0x00100010u, 0x80000001u);
    Check(generated::ReadS32(0x00100010u) == 0xFFFFFFFF80000001ull, "lw sign-extends");
    Check(generated::ReadU32(0x00100010u) == 0x80000001ull, "lwu zero-extends");
    generated::Write8(0x00100014u, 0xF0);
    Check(generated::ReadS8(0x00100014u) == 0xFFFFFFFFFFFFFFF0ull, "lb sign-extends");
}

void OutsideMemoryFaults() {
    bool faulted = false;
    try {
        ps2host::GuestMemory().Read(0x40000000u, 4);
    } catch (const ps2host::MemoryFault& fault) {
        faulted = fault.address == 0x40000000u && !fault.write;
    }
    Check(faulted, "an unmapped read faults");
}

void ScratchpadAndHooks() {
    ps2host::Memory& m = ps2host::GuestMemory();
    m.Write(ps2host::kScratchpadBase + 8, 0x55u, 1);
    Check(m.Read(ps2host::kScratchpadBase + 8, 1) == 0x55u, "scratchpad");
    int writes = 0;
    m.OnWrite(0x10003000u, [&](u32, u64 value) { writes += value == 7; });
    m.OnRead(0x10003000u, [](u32, u64 stored) { return stored + 1; });
    m.Write(0x10003000u, 7, 4);
    Check(writes == 1, "a write hook sees the value");
    Check(m.Read(0x10003000u, 4) == 8, "a read hook shapes the value");
}

void Kseg1ReachesDeviceRegisters() {
    ps2host::Memory& m = ps2host::GuestMemory();
    m.Write(0xB0001010u, 0x1234u, 4);                 // T1_MODE through kseg1
    Check(m.Read(0x10001010u, 4) == 0x1234u, "kseg1 reaches the EE registers");
    Check(m.Read(0x80100000u, 4) == m.Read(0x00100000u, 4), "kseg0 reaches RDRAM");
}

// A directory record: its length, extent, size, flags and name.
void Record(std::vector<u8>& image, u32 at, u32 lsn, u32 size, u8 flags, const std::string& name) {
    const u8 length = u8(33 + name.size() + (name.size() % 2 == 0 ? 1 : 0));
    image[at] = length;
    for (int i = 0; i < 4; ++i) {
        image[at + 2 + i] = u8(lsn >> (8 * i));
        image[at + 10 + i] = u8(size >> (8 * i));
    }
    image[at + 25] = flags;
    image[at + 32] = u8(name.size());
    std::memcpy(&image[at + 33], name.data(), name.size());
}

void DiscFindsFilesByPath() {
    using ps2host::kSectorSize;
    std::vector<u8> image(24 * kSectorSize, 0);
    u8* pvd = &image[16 * kSectorSize];
    pvd[0] = 1;
    std::memcpy(pvd + 1, "CD001", 5);
    Record(image, 16 * kSectorSize + 156, 20, kSectorSize, 2, std::string(1, '\0'));
    u32 at = 20 * kSectorSize;
    Record(image, at, 20, kSectorSize, 2, std::string(1, '\0')); at += 34;
    Record(image, at, 20, kSectorSize, 2, std::string(1, '\1')); at += 34;
    Record(image, at, 21, 5, 0, "FILE.BIN;1"); at += image[at];
    Record(image, at, 22, kSectorSize, 2, "DIR");
    Record(image, 22 * kSectorSize, 23, 3, 0, "INNER.DAT;1");
    std::memcpy(&image[21 * kSectorSize], "hello", 5);
    ps2host::Disc disc;
    disc.Use(image);
    ps2host::Disc::File file;
    Check(disc.Find("\\FILE.BIN;1", &file) && file.lsn == 21 && file.size == 5, "a file in the root");
    Check(disc.Find("/dir/inner.dat", &file) && file.lsn == 23 && file.size == 3,
          "a file in a directory, any case, no version");
    Check(!disc.Find("\\MISSING.BIN", &file), "a missing file");
    u8 sector[ps2host::kSectorSize];
    Check(disc.Read(21, 1, sector) && std::memcmp(sector, "hello", 5) == 0, "reading a sector");
    Check(!disc.Read(30, 1, sector), "reading past the end");
    Check(disc.IsCd(), "a small .iso counts as a CD");

    // The same disc as a raw CD image: Mode 2 sectors, data after the subheader.
    std::vector<u8> raw(image.size() / kSectorSize * ps2host::kRawSectorSize, 0);
    for (std::size_t i = 0; i < image.size() / kSectorSize; ++i) {
        u8* s = &raw[i * ps2host::kRawSectorSize];
        std::memset(s + 1, 0xFF, 10);
        s[15] = 2;
        s[16] = 0xAB;                                   // a subheader byte to see in 2340/2328 reads
        std::memcpy(s + 24, &image[i * kSectorSize], kSectorSize);
    }
    ps2host::Disc cd;
    cd.Use(raw);
    Check(cd.IsCd(), "a raw image is a CD");
    Check(cd.Find("/dir/inner.dat", &file) && file.lsn == 23 && file.size == 3, "a raw image's files");
    Check(cd.Read(21, 1, sector) && std::memcmp(sector, "hello", 5) == 0, "a raw sector's data");
    std::vector<u8> big(2340);
    Check(cd.ReadPattern(21, 1, 2, big.data()) && big[3] == 2 && big[4] == 0xAB &&
              std::memcmp(&big[12], "hello", 5) == 0,
          "a 2340-byte read: header, subheader, data");
    Check(cd.ReadPattern(21, 1, 1, big.data()) && std::memcmp(&big[0], "hello", 5) == 0,
          "a 2328-byte read starts at the data");
    Check(disc.ReadPattern(21, 1, 2, big.data()) && big[3] == 2 && std::memcmp(&big[12], "hello", 5) == 0,
          "an .iso's 2340-byte read has a made header");
    Check(!cd.Read(24, 1, sector), "reading past a raw image's end");
}

void BiosIsReadOnly() {
    ps2host::Memory& m = ps2host::GuestMemory();
    m.Write(ps2host::kBiosBase, 0xFFu, 1);
    Check(m.Read(ps2host::kBiosBase, 1) == 0, "the BIOS window ignores writes");
}

void WideWritesReachEveryWordsHook() {
    ps2host::Memory& m = ps2host::GuestMemory();
    u32 seen[2] = {0, 0};
    m.OnWrite(0x10006000u, [&seen](u32, u64 value) { seen[0] = u32(value); });
    m.OnWrite(0x10006004u, [&seen](u32, u64 value) { seen[1] = u32(value); });
    m.Write(0x10006000u, 0x2222222211111111ull, 8);
    Check(seen[0] == 0x11111111u && seen[1] == 0x22222222u, "a 64-bit write reaches both words");
    m.OnWrite(0x10006000u, nullptr);
    m.OnWrite(0x10006004u, nullptr);
}

void VifSkipsDataAndAppliesMaskPath3() {
    ps2host::VifStream vif;
    int masks = 0;
    bool masked = false;
    vif.on_mask_path3 = [&](bool on) { ++masks; masked = on; };
    // DIRECT of one quadword whose data looks like MSKPATH3: it must be skipped.
    vif.Word(0x50000001u);
    for (int i = 0; i < 4; ++i) vif.Word(0x06008000u);
    Check(masks == 0, "DIRECT data is not decoded");
    vif.Word(0x06008000u);
    Check(masks == 1 && masked, "MSKPATH3 masks");
    vif.Word(0x06000000u);
    Check(masks == 2 && !masked, "MSKPATH3 unmasks");
    // UNPACK V4-32 of 2 elements: 8 words. V2-16 of 3: 3 words.
    vif.Word(0x6C020000u);
    for (int i = 0; i < 8; ++i) vif.Word(0x06008000u);
    vif.Word(0x65030000u);
    for (int i = 0; i < 3; ++i) vif.Word(0x06008000u);
    Check(masks == 2 && vif.data_words() == 4 + 8 + 3, "UNPACK lengths");
    // STCYCL CL=1 WL=2: an UNPACK V4-32 of 4 reads 2 elements (filling writes).
    vif.Word(0x01000201u);
    vif.Word(0x6C040000u);
    for (int i = 0; i < 8; ++i) vif.Word(0);
    vif.Word(0x06008000u);
    Check(masks == 3, "UNPACK with WL > CL reads CL of every WL");
    // MPG of 1 instruction pair: 2 words.
    vif.Word(0x4A010000u);
    vif.Word(0x06000000u);
    vif.Word(0x06000000u);
    Check(masks == 3, "MPG data is not decoded");
}

void PutTag(u32 at, u32 qwc, u32 id, u32 address, u32 upper0, u32 upper1) {
    ps2host::Memory& m = ps2host::GuestMemory();
    m.Write(at, qwc | (id << 28), 4);
    m.Write(at + 4, address, 4);
    m.Write(at + 8, upper0, 4);
    m.Write(at + 12, upper1, 4);
}

void DmaChainsAreWalked() {
    ps2host::Memory& m = ps2host::GuestMemory();
    const u32 chain = 0x00200000u, table = 0x00210000u, sub = 0x00220000u;
    // cnt (1 qword after it) -> ref (1 qword at table) -> call sub -> end
    PutTag(chain, 1, 1, 0, 0xA0000001u, 0xA0000002u);
    for (u32 i = 0; i < 4; ++i) m.Write(chain + 16 + 4 * i, 0xD0 + i, 4);
    PutTag(chain + 32, 1, 3, table, 0, 0);
    for (u32 i = 0; i < 4; ++i) m.Write(table + 4 * i, 0xE0 + i, 4);
    PutTag(chain + 48, 0, 5, sub, 0, 0);
    PutTag(sub, 0, 6, 0, 0, 0);                     // ret: back to chain + 64
    PutTag(chain + 64, 0, 7, 0, 0, 0);              // end
    ps2host::DmaChannelState ch{0x145u, 0, 0, chain, 0, 0};    // chain, TTE, STR
    std::vector<u32> words;
    const ps2host::DmaOutcome out = ps2host::SendDma(ch, [&words](u32 w) { words.push_back(w); });
    Check(out.ended && out.tags == 5 && out.quadwords == 2, "a chain is walked to its end tag");
    Check(words.size() == 2 * 5 + 8 && words[0] == 0xA0000001u && words[2] == 0xD0u &&
          words[8] == 0xE0u, "tags' upper halves, then each tag's data, in order");
    Check(((ch.chcr >> 28) & 7) == 7 && ((ch.chcr >> 4) & 3) == 0,
          "the TAG field and the call depth are left as at the end");
    // A chain that never ends is stopped, not followed forever.
    PutTag(chain, 0, 2, chain, 0, 0);               // next -> itself
    ps2host::DmaChannelState loop{0x105u, 0, 0, chain, 0, 0};
    Check(!ps2host::SendDma(loop, [](u32) {}, 100).ended, "a looping chain is cut off");
}

// VIF1's interrupt bit (vif.h): the scanner stops after the command carrying
// it, its data included, however the words arrive; a data word that looks
// like a VIFcode is data.
void VifInterruptBitStopsAfterItsCommand() {
    ps2host::VifInterruptScanner scan;
    const u32 kFlushaI = 0x93000000u, kNop = 0, kStcycl11 = 0x01000101u;
    const u32 plain[] = {kNop, kFlushaI, kNop};
    auto stop = scan.Scan(plain, 3);
    Check(stop.stall && stop.words == 2, "a FLUSHA with the bit: the stall right after it");
    stop = scan.Scan(plain + 2, 1);
    Check(!stop.stall && stop.words == 1, "then the stream goes on");
    // UNPACK V4-32 of 2 elements (8 words) with the bit: after its data.
    const u32 unpack[] = {kStcycl11, 0xEC020000u, 1, 2, 3, kFlushaI, 5, 6, 7, 8, kNop};
    stop = scan.Scan(unpack, 11);
    Check(stop.stall && stop.words == 10, "an UNPACK with the bit: after its data, not at a look-alike in it");
    stop = scan.Scan(unpack + 10, 1);
    ps2host::VifInterruptScanner split;
    stop = split.Scan(unpack, 5);
    Check(!stop.stall && stop.words == 5, "its data split across transfers: no stall yet");
    stop = split.Scan(unpack + 5, 6);
    Check(stop.stall && stop.words == 5, "the stall where its data ends");
    // DIRECT of 1 quadword with the bit; an UNPACK without it, whose data
    // holds 0x93000000, does not stall.
    ps2host::VifInterruptScanner direct;
    const u32 words[] = {0xD0000001u, 9, 9, 9, 9, 0x6C010000u, kFlushaI, 0, 0, 0, kNop};
    stop = direct.Scan(words, 11);
    Check(stop.stall && stop.words == 5, "a DIRECT with the bit: after its quadword");
    stop = direct.Scan(words + 5, 6);
    Check(!stop.stall && stop.words == 6, "data that looks like a VIFcode is data");
}

// A transfer to a peripheral that stalls (dma.h, SendDmaBlocksUntil): it
// stops there, keeps what it read past the stall, and goes on from there.
void DmaStallsAndGoesOn() {
    ps2host::Memory& m = ps2host::GuestMemory();
    const u32 chain = 0x00230000u;
    // cnt: its upper half NOP, FLUSHA|i (Stuntman's tags), one quadword of
    // data; then end, its upper half NOP NOP.
    PutTag(chain, 1, 1, 0, 0x00000000u, 0x93000000u);
    for (u32 i = 0; i < 4; ++i) m.Write(chain + 16 + 4 * i, 0xD0 + i, 4);
    PutTag(chain + 32, 0, 7, 0, 0, 0);
    ps2host::VifInterruptScanner scan;
    std::vector<u32> got;
    const ps2host::DmaStallingSink sink = [&](const u32* words, u32 count, u32) {
        const auto stop = scan.Scan(words, count);
        got.insert(got.end(), words, words + stop.words);
        return ps2host::DmaTaken{stop.words, stop.stall};
    };
    ps2host::DmaChannelState ch{0x145u, 0, 0, chain, 0, 0};
    ps2host::DmaOutcome out = ps2host::SendDmaBlocksUntil(ch, sink);
    Check(out.stalled && !out.ended && got.size() == 2 && got[1] == 0x93000000u,
          "a chain stalls at an interrupt bit in a tag's upper half");
    Check(ch.madr == chain + 16 && ch.qwc == 1 && ch.tadr == chain + 32 && out.leftover_count == 0,
          "with that tag's data owed");
    out = ps2host::SendDmaBlocksUntil(ch, sink);
    Check(!out.stalled && out.ended && got.size() == 2 + 4 + 2 && got[2] == 0xD0u,
          "going on: the tag's data, then the rest to the end");
    // Normal mode, stalled inside a quadword: the rest of it is kept.
    const u32 data = 0x00231000u;
    const u32 normal[8] = {0, 0x93000000u, 0x11, 0x22, 0, 0, 0, 0};
    for (u32 i = 0; i < 8; ++i) m.Write(data + 4 * i, normal[i], 4);
    ps2host::VifInterruptScanner fresh;
    scan = fresh;
    got.clear();
    ps2host::DmaChannelState n{0x101u, data, 2, 0, 0, 0};
    out = ps2host::SendDmaBlocksUntil(n, sink);
    Check(out.stalled && got.size() == 2 && n.madr == data + 16 && n.qwc == 1 && out.leftover_count == 2 &&
              out.leftover[0] == 0x11 && out.leftover[1] == 0x22,
          "stalled inside a quadword: the rest of it kept, the next one owed");
    // A chain stalled in its end tag's data has ended once that data is sent.
    PutTag(chain, 1, 7, 0, 0x93000000u, 0);
    scan = fresh;
    got.clear();
    ps2host::DmaChannelState e{0x145u, 0, 0, chain, 0, 0};
    out = ps2host::SendDmaBlocksUntil(e, sink);
    Check(out.stalled && e.end_pending && e.qwc == 1 && out.leftover_count == 1, "stalled in an end tag");
    out = ps2host::SendDmaBlocksUntil(e, sink);
    Check(out.ended && !out.stalled && e.qwc == 0 && !e.end_pending, "its data sent, the chain has ended");
}

// --watch-read (memory.h): a load of a watched address - inline main-RAM loads
// included - is logged with its value, others not; the right value either way.
void WatchedReadsAreLogged() {
    ps2host::Memory& m = ps2host::GuestMemory();
    const u32 watched = 0x00240010u;
    m.Write(watched, 0x12345678u, 4);
    m.Write(watched + 16, 0x9ABCDEF0u, 4);
    const std::filesystem::path log = std::filesystem::temp_directory_path() / "ps2host_tests_watch_read.txt";
    std::fflush(stderr);
    const int saved = _dup(_fileno(stderr));
    FILE* to = std::fopen(log.u8string().c_str(), "w");
    _dup2(_fileno(to), _fileno(stderr));
    m.WatchReads({watched});
    const bool slow = generated::ram_base == nullptr;
    const u64 seen = generated::ReadU32(watched), other = generated::ReadU32(watched + 16);
    std::fflush(stderr);
    _dup2(saved, _fileno(stderr));
    _close(saved);
    std::fclose(to);
    m.WatchReads({});
    std::string text;
    {
        std::ifstream in(log);
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    std::filesystem::remove(log);
    Check(slow && generated::ram_base == m.rdram(), "watching a read sends loads the slow way, and only then");
    Check(seen == 0x12345678u && other == 0x9ABCDEF0u, "watched or not, a load reads its value");
    Check(text.find("[read] 0x00240010 -> 0x12345678") != std::string::npos &&
              text.find("0x00240020") == std::string::npos,
          "the watched load is logged, the other not");
}

// Every format's layout maps a page's pixels onto distinct bytes (nibbles).
void GsLayoutsArePermutations() {
    using namespace ps2host::gs;
    struct Case { u32 psm, w, h, bw; };
    const Case cases[] = {{PSMCT32, 64, 32, 1}, {PSMZ32, 64, 32, 1}, {PSMCT16, 64, 64, 1},
                          {PSMCT16S, 64, 64, 1}, {PSMZ16, 64, 64, 1}, {PSMZ16S, 64, 64, 1},
                          {PSMT8, 128, 64, 2}, {PSMT4, 128, 128, 2}};
    for (const Case& c : cases) {
        std::vector<int> seen(8192 * 2, 0);
        bool ok = true;
        const u32 bytes = c.psm == PSMT4 ? 0 : c.psm == PSMT8 ? 1 : (c.psm & 2) ? 2 : 4;
        for (u32 y = 0; y < c.h; ++y)
            for (u32 x = 0; x < c.w; ++x) {
                u32 nibble = 0;
                const u32 a = Memory::Address(0, c.bw, x, y, c.psm, &nibble);
                if (a >= 8192) { ok = false; continue; }
                const u32 slot = bytes ? a / bytes : a * 2 + nibble;
                if (seen[slot]++) ok = false;
            }
        char what[64];
        std::snprintf(what, sizeof what, "GS layout 0x%02X covers one page exactly", unsigned(c.psm));
        Check(ok, what);
    }
    // The precomputed pages give what the manual's tables give, anywhere.
    const u32 psms[] = {PSMCT32, PSMCT24, PSMCT16, PSMCT16S, PSMT8, PSMT4, PSMT8H, PSMT4HL, PSMT4HH,
                        PSMZ32, PSMZ24, PSMZ16, PSMZ16S};
    u32 seed = 99, wrong = 0;
    for (u32 psm : psms)
        for (int i = 0; i < 20000; ++i) {
            seed = seed * 1664525u + 1013904223u;
            const u32 bp = seed >> 18, bw = 1 + ((seed >> 8) & 15);
            seed = seed * 1664525u + 1013904223u;
            const u32 x = (seed >> 5) & 2047, y = (seed >> 17) & 2047;
            u32 n1 = 0, n2 = 0;
            if (Memory::Address(bp, bw, x, y, psm, &n1) != Memory::AddressReference(bp, bw, x, y, psm, &n2) ||
                n1 != n2)
                ++wrong;
        }
    Check(wrong == 0, "GS addresses from precomputed pages match the step-by-step layout");
    // A row's run (image transfers) stores what one pixel at a time stores.
    auto one = std::make_unique<Memory>(), run = std::make_unique<Memory>();
    for (u32 psm : psms)
        for (int i = 0; i < 300; ++i) {
            seed = seed * 1664525u + 1013904223u;
            const u32 bp = seed >> 18, bw = 1 + ((seed >> 8) & 15);
            seed = seed * 1664525u + 1013904223u;
            const u32 x = (seed >> 5) & 2047, y = (seed >> 17) & 2047, count = std::min(1 + (seed & 255), 2048 - x);
            std::vector<u32> values(count);
            for (u32& v : values) v = seed = seed * 1664525u + 1013904223u;
            for (u32 k = 0; k < count; ++k) one->Write(bp, bw, x + k, y, psm, values[k]);
            run->WriteRun(bp, bw, x, y, psm, values.data(), count);
        }
    Check(std::memcmp(one->data(), run->data(), 4u << 20) == 0, "GS row runs store as single pixels do");
}

void GsDrawsASprite() {
    using namespace ps2host::gs;
    Gs gs;
    gs.WriteRegister(0x4C, (u64(0) << 0) | (u64(1) << 16) | (u64(PSMCT32) << 24));   // FRAME: fbp 0, 64 wide
    gs.WriteRegister(0x40, u64(63) << 16 | u64(63) << 48);                          // SCISSOR 0..63
    gs.WriteRegister(0x00, 6);                                                      // PRIM: sprite
    gs.WriteRegister(0x01, 0x80402010ull);                                          // RGBAQ
    gs.WriteRegister(0x05, (u64(4) << 4) | (u64(4) << 4) << 16);                    // XYZ2 (4,4)
    gs.WriteRegister(0x05, (u64(8) << 4) | (u64(6) << 4) << 16);                    // XYZ2 (8,6)
    gs.Finish();                                                                    // the queued draws
    Check(gs.memory.Read(0, 1, 5, 5, PSMCT32) == 0x80402010u, "a sprite's pixels take its colour");
    Check(gs.memory.Read(0, 1, 8, 5, PSMCT32) == 0 && gs.memory.Read(0, 1, 5, 6, PSMCT32) == 0,
          "a sprite stops short of its far edges");
    Check(gs.primitives() == 1 && gs.pixels_drawn() == 8, "one sprite of 4x2 pixels");
}

// PATH3 held by MSKPATH3: an unmask lets one EOP-ended group of packets go,
// as Burnout 3's texture uploads need (one group per window, five a picture).
void GsLoadsACsm2Clut() {
    using namespace ps2host::gs;
    Gs gs;
    // CSM2: 16 CT16 entries in a row, at (32..47, 3) of a 64-wide buffer at
    // block 0x40 - TEXCLUT CBW 1, COU 2, COV 3 - loaded in order, CSA aside.
    for (u32 i = 0; i < 16; ++i) gs.memory.Write(0x40, 1, 32 + i, 3, PSMCT16, 0x8000u | (i * 0x0421u));
    gs.WriteRegister(0x1C, 1 | (2u << 6) | (3u << 12));
    gs.WriteRegister(0x06, (u64(PSMT4) << 20) | (u64(0x40) << 37) | (u64(PSMCT16) << 51) | (u64(1) << 55) |
                               (u64(5) << 56) | (u64(1) << 61));
    bool ok = true;
    for (u32 i = 0; i < 16; ++i) ok = ok && (gs.Clut()[i] & 0xFFFFFF) == (i << 3) * 0x010101u;
    Check(ok, "a CSM2 CLUT loads from TEXCLUT's row, in order");
}

void Path3GoesOneGroupPerWindow() {
    ps2host::Path3Queue queue;
    auto tag = [&](u32 nloop, bool eop, u32 flg, u32 nreg) {
        const u64 low = u64(nloop) | u64(eop) << 15 | u64(flg) << 58 | u64(nreg) << 60;
        queue.Push(u32(low));
        queue.Push(u32(low >> 32));
        queue.Push(0);
        queue.Push(0);
    };
    auto data = [&](u32 qwords) { for (u32 i = 0; i < 4 * qwords; ++i) queue.Push(0xD0D0D0D0u); };
    tag(1, false, 0, 4); data(4);                   // PACKED: A+D setup, no EOP
    tag(3, true, 2, 0); data(3);                    // IMAGE, EOP: the first group ends here
    tag(2, true, 1, 3); data(3);                    // REGLIST 2 x 3 registers: 3 quadwords
    std::vector<u32> sent;
    queue.Release(false, [&](u32 word) { sent.push_back(word); });
    Check(sent.size() == 4 * (1 + 4 + 1 + 3), "an unmask lets the packets up to the first EOP go");
    sent.clear();
    queue.Release(false, [&](u32 word) { sent.push_back(word); });
    Check(sent.size() == 4 * (1 + 3) && queue.empty(), "the next unmask the next group (REGLIST sized by halves)");
    tag(1, false, 2, 0); data(1);
    queue.Release(false, [&](u32 word) { sent.push_back(word); });
    Check(queue.empty(), "a group without its EOP yet goes whole");
}

// Fog (PRIM.FGE): C = (F * C + (255 - F) * FOGCOL) >> 8, F from XYZF2 or the
// FOG register for XYZ2.
void GsFogsTowardsFogcol() {
    using namespace ps2host::gs;
    Gs gs;
    gs.WriteRegister(0x4C, (u64(0) << 0) | (u64(1) << 16) | (u64(PSMCT32) << 24));   // FRAME: fbp 0, 64 wide
    gs.WriteRegister(0x40, u64(63) << 16 | u64(63) << 48);                          // SCISSOR 0..63
    gs.WriteRegister(0x3D, 0x0000FFull);                                            // FOGCOL: red
    gs.WriteRegister(0x00, 6 | 1 << 5);                                             // PRIM: sprite, FGE
    gs.WriteRegister(0x01, 0x80FFFFFFull);                                          // RGBAQ: white
    gs.WriteRegister(0x04, (u64(0) << 4) | (u64(0) << 4) << 16 | u64(0x80) << 56);  // XYZF2 (0,0) F 0x80
    gs.WriteRegister(0x04, (u64(4) << 4) | (u64(4) << 4) << 16 | u64(0x80) << 56);  // XYZF2 (4,4)
    gs.Finish();
    Check(gs.memory.Read(0, 1, 1, 1, PSMCT32) == 0x807F7FFEu, "fog halfway to red: (128 * 255 + 127 * FOGCOL) >> 8");
    gs.WriteRegister(0x0A, u64(0) << 56);                                           // FOG: F 0 for XYZ2
    gs.WriteRegister(0x05, (u64(8) << 4) | (u64(0) << 4) << 16);                    // XYZ2 (8,0)
    gs.WriteRegister(0x05, (u64(12) << 4) | (u64(4) << 4) << 16);                   // XYZ2 (12,4)
    gs.Finish();
    Check(gs.memory.Read(0, 1, 9, 1, PSMCT32) == 0x800000FEu, "F 0 from the FOG register: FOGCOL (* 255 / 256)");
}

// A flat depth stays exact across a triangle: Burnout 3's loading bar is drawn at
// z 0xFFFFFF with GEQUAL over a buffer cleared to 0xFFFFFF, and float
// interpolation once lost the last bit and carved it into stripes.
void GsKeepsAFlatDepthExact() {
    using namespace ps2host::gs;
    Gs gs;
    gs.WriteRegister(0x4C, (u64(0) << 0) | (u64(1) << 16) | (u64(PSMCT32) << 24));   // FRAME: fbp 0, 64 wide
    gs.WriteRegister(0x4E, u64(8) | (u64(PSMZ24 & 0xF) << 24));                      // ZBUF: zbp 8 (blocks 256)
    gs.WriteRegister(0x40, u64(63) << 16 | u64(63) << 48);                          // SCISSOR 0..63
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x) gs.memory.Write(256, 1, x, y, PSMZ24, 0xFFFFFF);
    gs.WriteRegister(0x47, 1ull << 16 | 2ull << 17);                                // TEST: Z GEQUAL
    gs.WriteRegister(0x00, 3);                                                      // PRIM: triangle
    gs.WriteRegister(0x01, 0x80FFFFFFull);
    const u64 z = u64(0xFFFFFF) << 32;
    gs.WriteRegister(0x05, (u64(3) << 4) | (u64(1) << 4) << 16 | z);
    gs.WriteRegister(0x05, (u64(61) << 4) | (u64(1) << 4) << 16 | z);
    gs.WriteRegister(0x05, (u64(3) << 4) | (u64(37) << 4) << 16 | z);
    gs.Finish();
    u32 missing = 0;
    for (u32 y = 2; y < 18; ++y)
        for (u32 x = 4; x < 20; ++x)
            if (gs.memory.Read(0, 1, x, y, PSMCT32) != 0x80FFFFFFu) ++missing;
    Check(missing == 0, "a flat depth passes GEQUAL against itself everywhere");
}

// Triangles fill top-left: a pixel on an edge is drawn only when the edge is a
// left or top one, so two triangles sharing an edge draw it once, and a right
// or bottom edge on a pixel leaves it alone. Burnout 3's text-entry keys are
// An IMAGE packet taken a run of words at a time (Gs::Words: its quadwords
// to the transfer together) stores what it stores a word at a time, for
// every format, 24-bit pixels across quadwords and a transfer that ends
// inside the packet included.
/* PCSX2's names for a texture pack: a 64x64 PSMT4 texture's blocks hashed as
   PCSX2 hashes them (the reference from xxhash's own XXH3 over the same
   blocks, in PCSX2's order), its CLUT's 16 entries from CSA's; and file
   names of every form read back. */
void TexturePackNamesAsPcsx2() {
    namespace texture_pack = ps2host::texture_pack;
    ps2host::gs::Memory memory;
    for (std::size_t i = 0; i < (4u << 20); ++i) memory.data()[i] = std::uint8_t((i * 7) % 251);
    std::uint32_t clut[256];
    for (std::uint32_t i = 0; i < 256; ++i) clut[i] = i * 0x01020304u + 5;
    const std::uint64_t tex0 = 0x100 | 2ull << 14 | 0x14ull << 20 | 6ull << 26 | 6ull << 30 | 3ull << 56;
    const texture_pack::Name name = texture_pack::NameOf(memory, tex0, 0, 0, clut);
    Check(name.Text() == "c7671cbbc91ae787-c31b5682050eba5e-00001994", "a 4-bit texture's PCSX2 name");
    // A pack's files by name, in a folder of their own: the forms PCSX2 reads.
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "ps2host_texture_pack_test";
    fs::remove_all(dir);
    fs::create_directories(dir / "sub");
    for (const char* file : {"c7671cbbc91ae787-c31b5682050eba5e-00005994.dds", "sub/1a2b-00001980.png",
                             "3c4d-r64x32-00001980.png", "5e6f-7a8b-r0x16-00001993.dds", "not-a-texture.png",
                             "1234-5678.txt"}) {
        std::FILE* f = std::fopen((dir / file).string().c_str(), "wb");
        if (f) std::fclose(f);
    }
    Check(texture_pack::Open(dir.string()) == 4, "a pack's files of every form, in folders too");
    bool pending = false;
    texture_pack::Find(name, &pending);
    Check(pending, "a texture the pack has (TCC's bit ignored): its file read in the background");
    fs::remove_all(dir);
    texture_pack::Open("");
}

void GsTakesImagesInBlocks() {
    using namespace ps2host::gs;
    const u32 psms[] = {PSMCT32, PSMCT24, PSMCT16, PSMCT16S, PSMT8, PSMT4, PSMT8H, PSMT4HL, PSMT4HH,
                        PSMZ32, PSMZ24, PSMZ16, PSMZ16S};
    u32 seed = 7, wrong = 0;
    for (u32 psm : psms) {
        std::vector<u32> stream;
        auto qword = [&](u64 low, u64 high) {
            stream.insert(stream.end(), {u32(low), u32(low >> 32), u32(high), u32(high >> 32)});
        };
        const u32 w = 37, h = 9, qwords = 60;   // 60 quadwords: more than 37x9 pixels of most formats
        qword(4 | u64(1) << 60, 0xE);           // A+D, four registers
        qword(u64(0x40) << 32 | u64(2) << 48 | u64(psm) << 56, 0x50);  // BITBLTBUF: DBP 0x40, DBW 2
        qword(u64(5) << 32 | u64(3) << 48, 0x51);                      // TRXPOS: 5, 3
        qword(w | u64(h) << 32, 0x52);                                 // TRXREG
        qword(0, 0x53);                                                // TRXDIR: host to local
        qword(qwords | u64(2) << 58, 0);        // IMAGE
        for (u32 i = 0; i < 4 * qwords; ++i) stream.push_back(seed = seed * 1664525u + 1013904223u);
        auto one = std::make_unique<Gs>(), blocks = std::make_unique<Gs>();
        for (u32 word : stream) one->Word(3, word);
        for (std::size_t at = 0, step = 1; at < stream.size(); at += step, step = step * 3 % 23 + 1)
            blocks->Words(3, stream.data() + at, u32(std::min(step, stream.size() - at)));
        if (std::memcmp(one->memory.data(), blocks->memory.data(), 4u << 20) != 0 || !one->uploads() ||
            !blocks->uploads())
            ++wrong;
    }
    Check(wrong == 0, "GS image packets taken in blocks store as a word at a time");
}

// cells whose right edge lies on a whole pixel, translucent: drawn on both
// sides they doubled a column.
void GsFillsTopLeft() {
    using namespace ps2host::gs;
    Gs gs;
    gs.WriteRegister(0x4C, (u64(0) << 0) | (u64(1) << 16) | (u64(PSMCT32) << 24));   // FRAME: fbp 0, 64 wide
    gs.WriteRegister(0x40, u64(63) << 16 | u64(63) << 48);                          // SCISSOR 0..63
    gs.WriteRegister(0x42, 0ull | 2ull << 2 | 2ull << 4 | 1ull << 6 | 0x80ull << 32); // ALPHA: Cs * FIX 1.0 + Cd
    gs.WriteRegister(0x00, 3 | 1 << 6);                                             // PRIM: triangle, ABE
    gs.WriteRegister(0x01, 0x80101010ull);
    const auto xy = [](u32 x, u32 y) { return (u64(x) << 4) | (u64(y) << 4) << 16; };
    gs.WriteRegister(0x05, xy(2, 2));                                               // the square (2,2)-(10,10)
    gs.WriteRegister(0x05, xy(10, 2));                                              // in two halves on its diagonal
    gs.WriteRegister(0x05, xy(10, 10));
    gs.WriteRegister(0x05, xy(2, 2));
    gs.WriteRegister(0x05, xy(10, 10));
    gs.WriteRegister(0x05, xy(2, 10));
    gs.Finish();
    u32 once = 0, twice = 0, outside = 0;
    for (u32 y = 0; y < 12; ++y)
        for (u32 x = 0; x < 12; ++x) {
            const u32 rgb = gs.memory.Read(0, 1, x, y, PSMCT32) & 0xFFFFFF;
            const bool inside = x >= 2 && x < 10 && y >= 2 && y < 10;
            if (inside && rgb == 0x101010) ++once;
            if (rgb == 0x202020) ++twice;
            if (!inside && rgb != 0) ++outside;
        }
    Check(once == 64, "every pixel of the square drawn once");
    Check(twice == 0, "the shared diagonal drawn once, not by both halves");
    Check(outside == 0, "the right and bottom edges, on whole pixels, not drawn");
}

#if PS2HOST_VULKAN
// GS local memory on the GPU (--gpu): targets and depth buffers swizzled into it
// and out of it by a compute shader must land where the GS model puts them,
// byte for byte, in every format a target or depth buffer takes.
void GpuMemorySwizzlesAsTheGs() {
    using namespace ps2host::gs;
    std::string error;
    auto gpu = ps2gpu::CreateGpu(&error);
    if (!gpu) {
        std::printf("  (no Vulkan device, GPU local memory unchecked: %s)\n", error.c_str());
        return;
    }
    auto renderer = ps2gpu::CreateGsRenderer(*gpu, &error);
    Check(renderer != nullptr, "the GPU renderer starts");
    if (!renderer) return;
    const u32 w = 128, h = 64, fbw = 2;
    u32 seed = 12345;
    auto next = [&] { seed = seed * 1664525u + 1013904223u; return seed; };
    auto to16 = [](u32 c) {
        return ((c >> 3) & 31) | (((c >> 11) & 31) << 5) | (((c >> 19) & 31) << 10) | ((c >> 31) << 15);
    };
    auto from16 = [](u32 c) {
        return ((c & 31) << 3) | (((c >> 5) & 31) << 11) | (((c >> 10) & 31) << 19) | ((c & 0x8000) ? 0x80000000u : 0);
    };
    Memory reference;                               // what the GPU's copy must hold, all along
    std::vector<u8> gpu_bytes(4u << 20);
    auto same_memory = [&] {
        renderer->MemoryRead(0, 16384, gpu_bytes.data());
        return std::memcmp(gpu_bytes.data(), reference.data(), gpu_bytes.size()) == 0;
    };
    struct Format { u32 psm; bool depth; const char* name; };
    const Format formats[] = {{PSMCT32, false, "CT32"}, {PSMCT24, false, "CT24"}, {PSMCT16, false, "CT16"},
                              {PSMCT16S, false, "CT16S"}, {PSMZ32, true, "Z32"}, {PSMZ24, true, "Z24"},
                              {PSMZ16, true, "Z16"}, {PSMZ16S, true, "Z16S"}};
    u32 base = 0x40;                                // FBP, in 2048-word units; a new place each time
    for (const Format& f : formats) {
        const bool sixteen = f.psm == PSMCT16 || f.psm == PSMCT16S || f.psm == PSMZ16 || f.psm == PSMZ16S;
        // Image to memory.
        std::vector<u32> pixels(w * h);
        for (u32& p : pixels) p = f.depth ? next() & (sixteen ? 0xFFFFu : 0xFFFFFFu) : next();
        if (f.depth) {
            renderer->UploadDepth(base, w, h, pixels.data());
            Check(renderer->DepthToMemory(base, w, h, fbw, f.psm, w, h), "a depth buffer goes to GPU memory");
        } else {
            renderer->UploadTarget(base, fbw, f.psm, pixels.data(), w, h);
            Check(renderer->TargetToMemory(base, fbw, f.psm, w, h), "a target goes to GPU memory");
        }
        for (u32 y = 0; y < h; ++y)
            for (u32 x = 0; x < w; ++x) {
                const u32 c = pixels[y * w + x];
                reference.Write(base * 32, fbw, x, y, f.psm, !f.depth && sixteen ? to16(c) : c);
            }
        std::string what = std::string(f.name) + ": an image swizzled on the GPU as the GS lays it out";
        Check(same_memory(), what.c_str());
        // Memory to image: other bytes there, from the CPU.
        const u32 blocks = fbw * (sixteen ? 1 : 2) * 32;   // the 64 rows' pages
        for (u32 i = base * 32 * 256; i < (base * 32 + blocks) * 256; ++i) reference.data()[i] = u8(next() >> 24);
        renderer->MemoryWrite(base * 32, base * 32 + blocks, reference.data() + base * 32 * 256);
        std::vector<u32> back;
        u32 bw_ = 0, bh = 0;
        if (f.depth) {
            renderer->DepthFromMemory(base, w, h, fbw, f.psm);
            Check(renderer->ReadDepth(base, w, h, &back), "a depth buffer reads back");
        } else {
            renderer->TargetFromMemory(base, fbw, f.psm, w, h);
            Check(renderer->ReadTarget(base, fbw, f.psm, &back, &bw_, &bh) && bw_ == w, "a target reads back");
        }
        u32 wrong = 0;
        for (u32 y = 0; y < h && back.size() == w * h; ++y)
            for (u32 x = 0; x < w; ++x) {
                const u32 raw = reference.Read(base * 32, fbw, x, y, f.psm);
                // Depth goes through a float (Z / 2^32): exact up to 24 bits.
                const u32 want = f.depth ? (raw >= 0xFFFFFF80u ? 0xFFFFFFFFu : u32(float(raw)))
                                         : sixteen ? from16(raw) : raw;
                if (back[y * w + x] != want) ++wrong;
            }
        what = std::string(f.name) + ": memory unswizzled on the GPU as the GS reads it";
        Check(back.size() == w * h && wrong == 0, what.c_str());
        base += 8;
    }
}

// Textures decoded on the GPU from its copy of local memory (--gpu, when the
// GPU holds their blocks) must come out texel for texel as the GS model
// decodes them: every format, CLUTs (CSA for 4-bit ones), TEXA, region modes.
void GpuDecodesTexturesAsTheGs() {
    using namespace ps2host::gs;
    std::string error;
    auto gpu = ps2gpu::CreateGpu(&error);
    if (!gpu) return;                               // reported by GpuMemorySwizzlesAsTheGs
    auto renderer = ps2gpu::CreateGsRenderer(*gpu, &error);
    if (!renderer) return;
    Gs gs;
    gs.rasterize = false;
    u32 seed = 777;
    for (u32 i = 0; i < (4u << 20); ++i) {
        seed = seed * 1664525u + 1013904223u;
        gs.memory.data()[i] = u8(seed >> 24);
    }
    renderer->MemoryWrite(0, 16384, gs.memory.data());
    std::vector<u32> decoded;
    gs.on_draw = [&](const Gs::DrawCall&) {
        u32 w = 0, h = 0;
        gs.DecodeTexture(&decoded, &w, &h);
    };
    struct Case { u32 psm, cpsm, csa; u64 clamp; const char* name; };
    const u64 region_clamp = 2 | 2 << 2 | u64(5) << 4 | u64(40) << 14 | u64(3) << 24 | u64(20) << 34;
    const u64 region_repeat = 3 | 3 << 2 | u64(15) << 4 | u64(32) << 14 | u64(7) << 24 | u64(8) << 34;
    const Case cases[] = {
        {PSMCT32, 0, 0, 0, "CT32"}, {PSMCT24, 0, 0, 0, "CT24"}, {PSMCT16, 0, 0, 0, "CT16"},
        {PSMCT16S, 0, 0, region_clamp, "CT16S, region clamp"}, {PSMT8, 0, 0, 0, "T8, CT32 CLUT"},
        {PSMT8, PSMCT16, 0, region_repeat, "T8, CT16 CLUT, region repeat"}, {PSMT4, 0, 3, 0, "T4, CSA 3"},
        {PSMT8H, 0, 0, 0, "T8H"}, {PSMT4HL, 0, 5, 0, "T4HL"}, {PSMT4HH, 0, 1, 0, "T4HH"},
        {PSMZ32, 0, 0, 0, "Z32"}, {PSMZ24, 0, 0, 0, "Z24"}, {PSMZ16, 0, 0, 0, "Z16"}, {PSMZ16S, 0, 0, 0, "Z16S"}};
    u64 key = 1;
    for (const Case& k : cases) {
        const u32 tbp = 0x480, tbw = 2, lw = 6, lh = 6;             // 64 x 64
        const u64 texa = 0x40 | 1ull << 15 | 0x90ull << 32;           // TA0 0x40, AEM, TA1 0x90
        const u64 tex0 = u64(tbp) | u64(tbw) << 14 | u64(k.psm) << 20 | u64(lw) << 26 | u64(lh) << 30 |
                         u64(0x2400) << 37 | u64(k.cpsm) << 51 | u64(k.csa) << 56 | 1ull << 61;   // CLD 1
        gs.WriteRegister(0x3B, texa);
        gs.WriteRegister(0x08, k.clamp);
        gs.WriteRegister(0x06, tex0);
        gs.WriteRegister(0x00, 6 | 1 << 4);                           // a textured sprite
        gs.WriteRegister(0x05, 0);
        gs.WriteRegister(0x05, (u64(16) << 4) | (u64(16) << 4) << 16);
        ps2gpu::GsTextureDecode d;
        d.tbp = tbp;
        d.tbw = tbw;
        d.psm = k.psm;
        d.tw = 1u << lw;
        d.th = 1u << lh;
        d.wms = u32(k.clamp & 3);
        d.wmt = u32(k.clamp >> 2 & 3);
        d.minu = u32(k.clamp >> 4 & 1023);
        d.maxu = u32(k.clamp >> 14 & 1023);
        d.minv = u32(k.clamp >> 24 & 1023);
        d.maxv = u32(k.clamp >> 34 & 1023);
        d.texa_lo = u32(texa & 0x80FF);
        d.ta1 = u32(texa >> 32 & 0xFF);
        d.csa = k.csa;
        d.clut = gs.Clut().data();
        renderer->TextureFromMemory(key, d);
        std::vector<u32> back;
        u32 w = 0, h = 0;
        const bool read = renderer->ReadTexture(key, &back, &w, &h);
        u32 wrong = 0;
        for (std::size_t i = 0; i < back.size() && i < decoded.size(); ++i) wrong += back[i] != decoded[i];
        const std::string what = std::string(k.name) + ": decoded on the GPU as the GS model decodes it";
        Check(read && back.size() == decoded.size() && decoded.size() == 64 * 64 && wrong == 0, what.c_str());
        ++key;
        if (k.psm != PSMT8 && k.psm != PSMT4 && k.psm != PSMT8H && k.psm != PSMT4HL && k.psm != PSMT4HH) continue;
        // The same through a CLUT the GPU loads from its own copy of memory
        // (--gpu, a CLUT drawn on the GPU): its other rows garbage first, so
        // a texture reading any row the load did not set fails.
        std::vector<u32> garbage(256);
        for (u32 i = 0; i < 256; ++i) garbage[i] = i * 0x9E3779B9u;
        const bool eight = k.psm == PSMT8 || k.psm == PSMT8H;
        renderer->ClutRows(garbage.data(), 0xFFFF);
        if (!eight) renderer->ClutRows(gs.Clut().data(), ~(1u << (k.csa % 16)));
        ps2gpu::GsClutLoad load;
        load.cbp = 0x2400;
        load.cpsm = k.cpsm;
        load.csa = k.csa;
        load.eight = eight;
        load.texa_lo = d.texa_lo;
        load.ta1 = d.ta1;
        renderer->ClutFromMemory(load);
        d.clut = nullptr;
        renderer->TextureFromMemory(key, d);
        back.clear();
        const bool again = renderer->ReadTexture(key, &back, &w, &h);
        wrong = 0;
        for (std::size_t i = 0; i < back.size() && i < decoded.size(); ++i) wrong += back[i] != decoded[i];
        const std::string loaded = std::string(k.name) + ": through a CLUT loaded on the GPU";
        Check(again && back.size() == decoded.size() && wrong == 0, loaded.c_str());
        ++key;
    }
}
#endif

// VIF1 into a VU: a microprogram loaded with MPG and run with MSCAL, UNPACK's
// formats, mask and skipping writes, and XGKICK.
void VifRunsMicroprogramsOnAVu() {
    ps2host::Vu vu(1);
    ps2host::VifStream vif;
    vif.vu = &vu;
    vif.double_buffered = true;
    std::vector<std::uint64_t> kicked;
    vu.on_kick = [&](std::uint64_t low, std::uint64_t) { kicked.push_back(low); };
    auto f = [](float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; };
    const std::uint32_t kNopU = 0x000002FF, kNopL = 0x8000033C, kEnd = 0x40000000;
    const std::uint32_t program[] = {
        // 0: lq.xyzw vf1, 0(vi0)              1: add.xyzw vf2, vf1, vf1
        0x01E10000, kNopU, kNopL, 0x01E108A8,
        // 2: sq.xyzw vf2, 1(vi0) [E]          3: (the pair after the end)
        0x03E01001, kNopU | kEnd, kNopL, kNopU,
        // 4: iaddiu vi1, vi0, 2               5: xgkick vi1 [E]          6: -
        0x10010002, kNopU, 0x80000EFC, kNopU | kEnd, kNopL, kNopU};
    vif.Word(0x4A000000u | (7u << 16));                     // MPG 7 pairs at 0
    for (std::uint32_t w : program) vif.Word(w);
    vif.Word(0x01000101u);                                  // STCYCL CL 1 WL 1
    vif.Word(0x6C010000u);                                  // UNPACK V4-32, 1 at 0
    for (float x : {1.0f, 2.0f, 3.0f, 4.0f}) vif.Word(f(x));
    vif.Word(0x14000000u);                                  // MSCAL 0
    const std::uint32_t* q = vu.Quad(1);
    Check(q[0] == f(2) && q[1] == f(4) && q[2] == f(6) && q[3] == f(8), "VU: lq, add, sq");
    Check(vu.resume == 4 * 8, "VU: a program ends after the pair following its E bit");

    // V3-16 masked: w from the row register.
    vif.Word(0x20000000u);                                  // STMASK
    vif.Word(0x40u);                                        // cycle 0, w: row
    vif.Word(0x30000000u);                                  // STROW
    for (std::uint32_t w : {0u, 0u, 0u, f(1)}) vif.Word(w);
    vif.Word(0x79010003u);                                  // UNPACK V3-16 masked, 1 at 3
    vif.Word(0x0002FFFFu);
    vif.Word(0x00000003u);
    q = vu.Quad(3);
    Check(q[0] == 0xFFFFFFFFu && q[1] == 2 && q[2] == 3 && q[3] == f(1), "UNPACK V3-16, sign-extended, masked");

    // CL 2, WL 1: every other quadword.
    vif.Word(0x01000102u);                                  // STCYCL CL 2 WL 1
    vif.Word(0x6002000Au);                                  // UNPACK S-32, 2 at 10
    vif.Word(5);
    vif.Word(6);
    Check(vu.Quad(10)[3] == 5 && vu.Quad(11)[0] == 0 && vu.Quad(12)[1] == 6, "UNPACK skipping writes");

    // A GIF packet at quadword 2: one A+D register write, then EOP.
    std::uint32_t* tag = vu.Quad(2);
    tag[0] = 1u | 1u << 15;                                 // NLOOP 1, EOP
    tag[1] = 1u << 28;                                      // FLG PACKED, NREG 1
    tag[2] = 0xE;                                           // A+D
    tag[3] = 0;
    vif.Word(0x01000101u);
    vif.Word(0x14000004u);                                  // MSCAL 4 (byte 32)
    Check(kicked.size() == 2 && (kicked[0] & 0x7FFF) == 1, "XGKICK sends the tag and its data");
}

// The VU's visible timing: the flags reach FMAND four cycles after the FMAC
// operation, Q seven after DIV (WAITQ waits for it), and an upper beside a LOI
// reads the I from before - as Burnout 3's microcode depends on.
// The FMAC's four lanes at once (single precision, SSE) give what the
// arithmetic model gives a lane at a time in double: every operation, operand
// form, dest mask and destination, on bit patterns rich in zeros, denormals,
// infinities, NaNs and values near overflow and underflow.
void VuLanesAreTheModel() {
    auto context = std::make_unique<lifted::Context>(), scalar = std::make_unique<lifted::Context>();
    u32 seed = 2024, wrong = 0, runs = 0;
    auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
    auto value = [&]() -> u32 {
        const u32 r = next(), sign = r & 0x80000000u;
        switch ((r >> 8) % 8) {
            case 0: return sign;                                            // zero
            case 1: return sign | (r & 0x007FFFFFu);                        // denormal
            case 2: return sign | 0x7F800000u | ((r >> 4) & 1);             // infinity, NaN
            case 3: return sign | 0x7F000000u | (next() & 0x007FFFFFu);     // near the top
            case 4: return sign | 0x00800000u | (next() & 0x00FFFFFFu);     // near the bottom
            case 5: return sign | 0x3F800000u | (next() & 0x007FFFFFu);     // about one
            default: return next();
        }
    };
    for (int i = 0; i < 200000; ++i) {
        for (auto& v : context->vf) for (u32& lane : v) lane = value();
        for (u32& lane : context->vacc) lane = value();
        context->vc[20 + 1] = value();                                       // I
        context->vc[20 + 2] = value();                                       // Q
        lifted::FmacOp f;
        f.op = int(next() % 7);
        f.operand = int(next() % 7);
        f.to_acc = (next() & 3) == 0;
        f.fd = next() % 32;
        f.fs = next() % 32;
        f.ft = next() % 32;
        f.dest = next() % 16;
        *scalar = *context;
        lifted::RunFmac(*context, f);
        lifted::RunFmacScalar(*scalar, f);
        ++runs;
        if (std::memcmp(context->vf, scalar->vf, sizeof context->vf) != 0 ||
            std::memcmp(context->vacc, scalar->vacc, sizeof context->vacc) != 0)
            ++wrong;
    }
    char what[96];
    std::snprintf(what, sizeof what, "VU FMAC lanes at once match the model (%u of %u differ)", wrong, runs);
    Check(wrong == 0, what);
}

// An IOP image's library tags (run.h): each module's release, found by its
// space-padded name; file I/O's version check answers its module's release
// (Stuntman's library refuses every open otherwise).
void IopImageGivesModuleReleases() {
    const std::string text = std::string("xx fileio  ab12 ioman   2430 ") + "fileio  2430" +
                             std::string(3, '\0') + "loadfile2430";
    const std::vector<u8> image(text.begin(), text.end());
    char release[4] = {};
    Check(ps2host::ModuleRelease(image, "fileio", release) && std::memcmp(release, "2430", 4) == 0,
          "fileio's release, past a tag without digits");
    Check(ps2host::ModuleRelease(image, "loadfile", release) && std::memcmp(release, "2430", 4) == 0,
          "loadfile's release");
    Check(!ps2host::ModuleRelease(image, "mcman", release), "a module the image does not hold");
    ps2host::SetFileIoRelease("2430");
    const std::vector<u8> answer = ps2host::FileIoVersionAnswer();
    const u8 expected[8] = {'2', '4', '3', '0', 0, 0, 0, 0};
    Check(answer.size() == 8 && std::memcmp(answer.data(), expected, 8) == 0,
          "file I/O's version check answers the release, then four zeros");
    ps2host::SetFileIoRelease("2800");
}

// The watchdog's stop (run.h): the budgets spent again and again until the run
// has stopped - one write can be lost to a loop's own decrement - and given up
// at the limit when it never does.
void StopIsWrittenUntilTheRunStops() {
    using std::chrono::milliseconds;
    int spends = 0, waits = 0;
    const bool stopped = ps2host::SpendUntilStopped(
        [&] { ++spends; }, [&](milliseconds) { return ++waits == 3; }, milliseconds(1), milliseconds(10000));
    Check(stopped && spends == 3, "the stop is written again until the run stops");
    spends = 0;
    const auto from = std::chrono::steady_clock::now();
    const bool never = ps2host::SpendUntilStopped(
        [&] { ++spends; },
        [](milliseconds every) {
            std::this_thread::sleep_for(every);
            return false;
        },
        milliseconds(5), milliseconds(50));
    const auto took = std::chrono::steady_clock::now() - from;
    Check(!never && spends >= 2 && took >= milliseconds(50) && took < milliseconds(5000),
          "a run that never stops is given up at the limit");
}

// COP0 for translated code (run.h): Count is the EE's cycles on the host's
// clock since StartCop0 - the virtual clock here, so exact - sign-extended
// like any mfc0; every other register reads as 0 (Stuntman's busy-wait on
// Count at 0x001A3208 never ended with 0).
void Cop0CountFollowsTheClock() {
    ps2host::SetVirtualClock(true);
    ps2host::StartCop0();
    Check(lifted::ReadCop0(9) == 0, "Count starts at 0");
    ps2host::AdvanceClock(1000000);                          // a millisecond
    Check(lifted::ReadCop0(9) == 294912, "Count advances 294,912,000 a second");
    ps2host::AdvanceClock(8000000000ull);                    // past 2^31 cycles
    const u64 cycles = 2359296000ull + 294912;
    Check(lifted::ReadCop0(9) == u64(s64(s32(u32(cycles)))), "Count is sign-extended, as mfc0 does");
    Check(lifted::ReadCop0(12) == 0, "Status reads as 0");
    lifted::read_cop0_hook = nullptr;
    Check(lifted::ReadCop0(9) == 0, "without a host, Count reads as 0 (verification)");
    ps2host::SetVirtualClock(false);
}

// The EE sends a SIF command packet as the SDK does: the packet in EE memory,
// one SifSetDma transfer naming it (run.h's SifDma) - after another with the
// call's arguments, if any.
void SendSifPacket(std::vector<u32> words, const std::vector<u32>& arguments = {}) {
    ps2host::Memory& m = ps2host::GuestMemory();
    const u32 packet = 0x01E11000u, args = 0x01E12000u, transfers = 0x01E10000u;
    words[0] = u32(words.size() * 4);                        // the header's own size
    for (std::size_t i = 0; i < words.size(); ++i) m.Write(packet + u32(4 * i), words[i], 4);
    for (std::size_t i = 0; i < arguments.size(); ++i) m.Write(args + u32(4 * i), arguments[i], 4);
    u32 t = transfers, count = 0;
    auto transfer = [&](u32 src, u32 size, u32 dest) {
        m.Write(t, src, 4);
        m.Write(t + 4, dest, 4);
        m.Write(t + 8, size, 4);
        m.Write(t + 12, 0, 4);
        t += 16;
        ++count;
    };
    if (!arguments.empty()) transfer(args, u32(arguments.size() * 4), 0x00190000u);  // a host server's
    transfer(packet, u32(words.size() * 4), 0);
    ps2host::SifDma(transfers, count);
}

// Bind `sid` and return the server address the IOP answered with.
u32 BindSifServer(u32 sid, u32 receive) {
    std::vector<u32> bind(16, 0);
    bind[2] = 0x80000009u;
    bind[8] = sid;
    SendSifPacket(bind);
    ps2host::DeliverIopReply();
    return u32(ps2host::GuestMemory().Read(receive + 36, 4));
}

// Call `procedure` of the server at `server` with `arguments`.
void CallSifServer(u32 server, u32 procedure, const std::vector<u32>& arguments) {
    std::vector<u32> call(16, 0);
    call[2] = 0x8000000Au;
    call[8] = procedure;
    call[13] = server;
    SendSifPacket(call, arguments);
}

// fileio's asynchronous requests (fileio.cpp): one with its semaphore negated
// ends with the completion record where the library's initialisation said, and
// command 0x80000011, after the RPC end; a synchronous one sends none.
void FileIoCompletesAsynchronousRequests() {
    ps2host::Memory& m = ps2host::GuestMemory();
    const u32 receive = 0x01E00000u, record = 0x01E03000u, result = 0x01E04000u;
    ps2host::InstallFileIo();
    const u32 server = BindSifServer(0x80000001u, receive);
    CallSifServer(server, 0xFF, {record});
    ps2host::DeliverIopReply();                              // its RPC end
    m.Write(record + 4, 0xFFFFFFFFu, 4);
    // read(descriptor 77, nowhere, 16 bytes), asynchronous: semaphore -5.
    CallSifServer(server, 2, {u32(-5), result, 4, 77, 0, 16});
    Check(ps2host::DeliverIopReply() && m.Read(receive + 8, 4) == 0x80000008u, "the read's RPC end first");
    Check(ps2host::DeliverIopReply() && m.Read(receive + 8, 4) == 0x80000011u &&
              m.Read(receive + 4, 4) == record && (m.Read(receive, 4) >> 8) == 0xA4u,
          "then fileio's completion command, its record at the library's address");
    Check(m.Read(record, 4) == u32(-5) && m.Read(record + 4, 4) == 2u && m.Read(record + 8, 4) == result &&
              m.Read(record + 12, 4) == 4u && m.Read(record + 16, 4) == u32(-9) &&
              m.Read(record + 0x14, 4) == 0u,
          "the record: semaphore as sent, procedure, result's address and size, result, nothing unaligned");
    CallSifServer(server, 2, {0, result, 4, 77, 0, 16});     // synchronous (no semaphore to signal)
    Check(ps2host::DeliverIopReply() && m.Read(receive + 8, 4) == 0x80000008u && !ps2host::DeliverIopReply(),
          "a synchronous request: its RPC end only");
}

// SendCmdToEe (run.h): a host server's command to the EE - its data where it
// said, its header as sceSifSendCmd's - and, sent while the server answers a
// call, delivered after that call's RPC end (Stuntman's fileio completes an
// asynchronous read with command 0x80000011 after answering it).
void HostServersSendCommandsAfterTheirAnswer() {
    ps2host::Memory& m = ps2host::GuestMemory();
    const u32 receive = 0x01E00000u, dest = 0x01E02000u;
    ps2host::RegisterRpcServer(0x7E570000u, [&](u32, const std::vector<u8>&) {
        ps2host::SendCmdToEe(0x80000011u, {0xAA, 0xBB, 0xCC, 0xDD}, dest, {1, 2, 3, 4, 5});
        return std::vector<u8>(4, 0);
    });
    std::vector<u32> saddr(8, 0);
    saddr[2] = 0x80000000u;
    saddr[4] = receive;
    SendSifPacket(saddr);
    const u32 server = BindSifServer(0x7E570000u, receive);
    Check(m.Read(receive + 8, 4) == 0x80000008u, "the bind's RPC end");
    CallSifServer(server, 1, {});
    Check(m.Read(dest, 4) == 0x04030201u && m.Read(dest + 4, 1) == 5u, "the command's data, where it said");
    Check(ps2host::DeliverIopReply() && m.Read(receive + 8, 4) == 0x80000008u &&
              m.Read(receive + 32, 4) == 0x8000000Au,
          "the call's RPC end first");
    Check(ps2host::DeliverIopReply() && m.Read(receive + 8, 4) == 0x80000011u &&
              m.Read(receive, 4) == (20u | (5u << 8)) && m.Read(receive + 4, 4) == dest &&
              m.Read(receive + 16, 4) == 0xDDCCBBAAu,
          "then the command: its header and fields");
    Check(!ps2host::DeliverIopReply(), "nothing more");
}

// padman's scePadSetActAlign (command 8, dbc.cpp): done, where libpad 2.3
// (+0xC) and 2.4 (+0x14) read it - Stuntman's controller set-up waits on +0x14.
void PadmanSetActAlignIsDone() {
    ps2host::Memory& m = ps2host::GuestMemory();
    const u32 receive = 0x01E00000u, answer = 0x01E05000u;
    ps2host::InstallDbc();
    const u32 server = BindSifServer(0x80000100u, receive);
    for (u32 i = 0; i < 0x80; i += 4) m.Write(answer + i, 0xEEEEEEEEu, 4);
    std::vector<u32> call(16, 0);
    call[2] = 0x8000000Au;
    call[8] = 1;                                             // procedure
    call[10] = answer;                                       // receive
    call[11] = 0x80;                                         // receive size
    call[13] = server;
    std::vector<u32> request(0x20, 0);
    request[0] = 8;                                          // scePadSetActAlign
    request[3] = 0x00000100u;                                // align: motor 0, then 1
    request[4] = 0xFFFFFFFFu;
    SendSifPacket(call, request);
    Check(m.Read(answer + 0xC, 4) == 1 && m.Read(answer + 0x14, 4) == 1,
          "scePadSetActAlign answered done at +0xC (libpad 2.3) and +0x14 (2.4)");
    ps2host::DeliverIopReply();                              // its RPC end

    // A controller's frame tells of its actuators' combination (scePadInfoComb):
    // one, of the two motors.
    const u32 area = 0x01E06000u;
    for (u32 i = 0; i < 0x100; i += 4) m.Write(area + i, 0, 4);
    std::vector<u32> open(0x20, 0);
    open[0] = 0x01;                                          // scePadPortOpen
    open[4] = area;                                          // +0x10: the EE pad area
    SendSifPacket(call, open);
    ps2host::DeliverIopReply();
    ps2host::DbcFrame();
    const u32 f = m.Read(area + 0x58, 4) > m.Read(area + 0xD8, 4) ? area : area + 0x80;
    Check(m.Read(f + 0x6A, 1) == 2 && m.Read(f + 0x6B, 1) == 1 && m.Read(f + 0x40, 1) == 2 &&
              m.Read(f + 0x41, 1) == 0 && m.Read(f + 0x42, 1) == 1,
          "the pad's frame: two actuators, one combination of both");
    Check(m.Read(f + 0x66, 1) == 2 && (m.Read(f + 0x79, 4) & 0xFFFFFFFFu) == 0x0003FFFFu,
          "the pad's frame: a DualShock 2, its 18 buttons' pressure readable");
    std::vector<u32> press(0x20, 0);
    press[0] = 0x0A;                                         // scePadEnterPressMode
    press[3] = 0xFFF;
    m.Write(answer + 0x10, 0, 4);
    SendSifPacket(call, press);
    ps2host::DeliverIopReply();
    Check(m.Read(answer + 0xC, 4) == 1 && m.Read(answer + 0x10, 4) == 1,
          "scePadEnterPressMode answered done at +0x10 (libpad 2.4)");
    std::vector<u32> close(0x20, 0);
    close[0] = 0x0E;                                         // scePadPortClose, port 0 slot 0
    SendSifPacket(call, close);
    ps2host::DeliverIopReply();
}

// libmc's sceMcGetEntSpace (mcserv procedure 0x12, mcserv.cpp): a directory's
// free entries, two to a cluster with "." and ".." counted; -4 for none.
void McservCountsFreeEntries() {
    namespace fs = std::filesystem;
    ps2host::Memory& m = ps2host::GuestMemory();
    const fs::path card = fs::temp_directory_path() / "ps2host_tests_mcserv";
    std::error_code error;
    fs::remove_all(card, error);
    fs::create_directories(card / "ONEFILE", error);
    fs::create_directories(card / "EMPTY", error);
    { std::ofstream(card / "ONEFILE" / "data") << "x"; }
    const std::string saved = ps2host::mc2::CardDirectory();
    ps2host::mc2::SetCardDirectory(card.u8string());
    const u32 receive = 0x01E00000u, answer = 0x01E05000u;
    ps2host::InstallMcserv();
    const u32 server = BindSifServer(0x80000400u, receive);
    auto space = [&](const char* directory) {
        std::vector<u32> request(0x30 / 4 + 16, 0);         // port 0, slot 0; the path at +0x14
        char* path = reinterpret_cast<char*>(request.data()) + 0x14;
        std::strcpy(path, directory);
        std::vector<u32> call(16, 0);
        call[2] = 0x8000000Au;
        call[8] = 0x12;
        call[10] = answer;
        call[11] = 4;
        call[13] = server;
        m.Write(answer, 0xEEEEEEEEu, 4);
        SendSifPacket(call, request);
        ps2host::DeliverIopReply();
        return s32(m.Read(answer, 4));
    };
    Check(space("ONEFILE") == 1, "sceMcGetEntSpace: one entry free beside . .. and a file");
    Check(space("EMPTY") == 0, "sceMcGetEntSpace: none free in an empty directory's cluster");
    Check(space("MISSING") == -4, "sceMcGetEntSpace: no such directory");
    ps2host::mc2::SetCardDirectory(saved);
    fs::remove_all(card, error);
}

// libmc2's handlers over a directory (mc2.h): a save made and read back, the
// listing the game's "open" reads sizes from, the answers it maps to errors.
void MemoryCardKeepsSaves() {
    namespace mc2 = ps2host::mc2;
    namespace fs = std::filesystem;
    const fs::path card = fs::temp_directory_path() / "ps2host_tests_memcard";
    std::error_code error;
    fs::remove_all(card, error);
    mc2::SetCardDirectory(card.u8string());
    auto context = std::make_unique<lifted::Context>();
    lifted::Context& c = *context;
    constexpr u32 kName = 0x00100100, kBuffer = 0x00100200, kEntries = 0x00100400, kCount = 0x00100300;
    auto name = [&](const char* text) {
        for (u32 i = 0;; ++i) {
            generated::Write8(kName + i, u8(text[i]));
            if (!text[i]) break;
        }
        c.r[5] = kName;
    };
    auto call = [&](void (*handler)(lifted::Context&)) { handler(c); return u32(c.r[2]); };
    c.r[4] = 0;
    name("SAVE");
    c.r[6] = 0x8427;
    Check(call(mc2::Create) == 0, "memory card: a directory made");
    Check(call(mc2::Create) == 0x81010011, "memory card: making it again is EEXIST");
    name("SAVE/data");
    c.r[6] = 0x8417;
    Check(call(mc2::Create) == 0, "memory card: a file made");
    for (u32 i = 0; i < 1500; ++i) generated::Write8(kBuffer + i, u8(i * 7));
    c.r[6] = kBuffer; c.r[7] = 0; c.r[8] = 1500;
    Check(call(mc2::WriteFile) == 1500, "memory card: written");
    for (u32 i = 0; i < 1500; ++i) generated::Write8(kBuffer + i, 0);
    c.r[6] = kBuffer; c.r[7] = 100; c.r[8] = 0;
    bool same = call(mc2::ReadFile) == 1400;
    for (u32 i = 0; i < 1400 && same; ++i) same = generated::ReadU8(kBuffer + i) == u8((i + 100) * 7);
    Check(same, "memory card: read back from an offset to the end");
    name("SAVE/data");
    c.r[6] = 0; c.r[7] = 1; c.r[8] = kEntries; c.r[9] = kCount;
    Check(call(mc2::GetDir) == 0 && generated::ReadU32(kCount) == 1 && generated::ReadU32(kEntries + 0x10) == 1500 &&
              (generated::ReadU32(kEntries + 0x14) & 0xFFFF) == 0x8497,
          "memory card: the listing gives the size and a written file's mode");
    name("SAVE/missing");
    Check(call(mc2::GetDir) == 0 && generated::ReadU32(kCount) == 0, "memory card: listing nothing is no error");
    name("NONE/data");
    Check(call(mc2::GetDir) == 0x81010002, "memory card: listing in no directory is ENOENT");
    name("SAVE/missing");
    c.r[6] = kBuffer; c.r[7] = 0; c.r[8] = 0;
    Check(call(mc2::ReadFile) == 0x81010002, "memory card: reading what is not there is ENOENT");
    name("../outside");
    c.r[6] = 0x8417;
    Check(call(mc2::Create) == 0x81010016 && !fs::exists(card.parent_path() / "outside"),
          "memory card: a path out of the card is refused");
    name("SAVE");
    c.r[6] = kEntries;
    Check(call(mc2::Lookup) == 0x81010011 && (generated::ReadU32(kEntries) & 0x20) != 0,
          "memory card: the lookup finds a directory");
    c.r[5] = kBuffer;
    Check(call(mc2::GetInfo) == 0 && generated::ReadU32(kBuffer) == 2 && generated::ReadU32(kBuffer + 4) == 1 &&
              generated::ReadU32(kBuffer + 8) < 8000,
          "memory card: a formatted PS2 card, its free space less the save");
    mc2::SetCardDirectory("");
    fs::remove_all(card, error);
}

// Random microprograms (vu_corpus.cpp, from engine/tools/vurecomp.py
// --corpus), compiled ahead of time, run as the interpreter runs them: from
// random registers and memory, every register, flag, pipeline, cycle and
// byte of data memory the same at the end - and the pairs counted.
/* A program of the test corpus in a VU's instruction memory (NOPs around
   it), and random registers, flags and data memory from `seed`, which goes
   on: two VUs given the same seed start the same. */
void PrepareVuRun(ps2host::Vu& vu, const ps2host::VuCompiled& program, u32& seed) {
    auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
    auto number = [&]() -> u32 {
        const u32 r = next();
        switch (r % 6) {
            case 0: return 0;
            case 1: return 0x3F800000u ^ (r & 0x80000000u);                    // +-1
            case 2: return (r & 0x807FFFFFu) | 0x3F000000u;                     // about one
            case 3: return (r & 0x807FFFFFu) | 0x43000000u;                     // about 128-256
            default: return next();
        }
    };
    for (std::size_t at = 0; at < vu.micro.size(); at += 8) {
        const u32 nop[2] = {0x8000033Cu, 0x000002FFu};
        std::memcpy(vu.micro.data() + at, nop, 8);
    }
    const u32* code = program.code;
    const u32 runs_in = *code++;
    for (u32 r = 0; r < runs_in; ++r) {
        std::memcpy(vu.micro.data() + code[0], code + 2, 4 * code[1]);
        code += 2 + code[1];
    }
    for (int r = 1; r < 32; ++r)
        for (u32& lane : vu.regs.vf[r]) lane = number();
    for (u32& lane : vu.regs.vacc) lane = number();
    for (int r = 1; r < 16; ++r) vu.regs.vc[r] = next() % ((next() & 1) ? 16 : 0x10000);
    vu.regs.vc[20] = 0x3F800000u | (next() & 0x7FFFFF);
    vu.regs.vc[21] = number();
    vu.regs.vc[22] = number();
    for (std::size_t b = 0; b < vu.data.size(); b += 4) {
        const u32 value = number();
        std::memcpy(vu.data.data() + b, &value, 4);
    }
    vu.mac = next() & 0xFFFF;
    vu.status = next() & 0xFFF;
    vu.clip = next() & 0xFFFFFF;
    vu.p = number();
}

void VuCompiledIsTheInterpreter() {
    std::size_t count = 0;
    const ps2host::VuCompiled* programs = VuGeneratedPrograms(&count);
    u32 seed = 77;
    u32 wrong = 0, runs = 0, entered = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const ps2host::VuCompiled& program = programs[i];
        if (!program.block_count) continue;
        for (int trial = 0; trial < 6; ++trial) {
            auto interpreted = std::make_unique<ps2host::Vu>(program.unit);
            auto ahead = std::make_unique<ps2host::Vu>(program.unit);
            const u32 start = seed;
            for (ps2host::Vu* vu : {interpreted.get(), ahead.get()}) {
                seed = start;
                PrepareVuRun(*vu, program, seed);
            }
            interpreted->use_compiled = false;
            const u64 limit = 3000;
            const u64 a = interpreted->Run(program.entry, limit), b = ahead->Run(program.entry, limit);
            ++runs;
            if (ahead->compiled_pairs()) ++entered;
            if (a != b || !interpreted->SameState(*ahead)) ++wrong;
        }
    }
    char what[128];
    std::snprintf(what, sizeof what, "VU programs compiled ahead run as interpreted (%u of %u runs differ)", wrong, runs);
    Check(wrong == 0 && runs > 0, what);
    // (A jump to a register's address mostly leaves for the NOPs around: the
    // interpreter goes on there.)
    std::snprintf(what, sizeof what, "VU compiled programs taken where they are (%u of %u runs)", entered, runs);
    Check(entered == runs, what);
}

// --vu-fast: the corpus's VU1 programs that never read MAC or STATUS
// (vurecomp's fast_eligible) run their flag-free blocks, and end as the
// exact blocks do in everything else - registers, memory, CLIP, Q and P,
// every pipeline and cycle - and in the pairs counted.
void VuFastIsTheExactButFlags() {
    std::size_t count = 0;
    const ps2host::VuCompiled* programs = VuGeneratedPrograms(&count);
    u32 seed = 991;
    u32 wrong = 0, runs = 0, flags_differ = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const ps2host::VuCompiled& program = programs[i];
        if (!program.block_count || !program.fast_blocks) continue;
        for (int trial = 0; trial < 20; ++trial) {
            auto exact = std::make_unique<ps2host::Vu>(program.unit);
            auto fast = std::make_unique<ps2host::Vu>(program.unit);
            const u32 start = seed;
            for (ps2host::Vu* vu : {exact.get(), fast.get()}) {
                seed = start;
                PrepareVuRun(*vu, program, seed);
            }
            fast->fast = true;
            const u64 limit = 3000;
            const u64 a = exact->Run(program.entry, limit), b = fast->Run(program.entry, limit);
            ++runs;
            if (a != b || !exact->SameState(*fast, false) || !fast->compiled_pairs()) {
                ++wrong;
                std::fprintf(stderr, "[fast] program %zu entry %X trial %d: pairs %llu %llu, compiled %llu\n", i,
                             program.entry, trial, (unsigned long long)a, (unsigned long long)b,
                             (unsigned long long)fast->compiled_pairs());
                for (int r = 0; r < 32; ++r)
                    for (int l = 0; l < 4; ++l)
                        if (exact->regs.vf[r][l] != fast->regs.vf[r][l]) std::fprintf(stderr, "  vf%d.%d\n", r, l);
                for (int r = 0; r < 32; ++r)
                    if (exact->regs.vc[r] != fast->regs.vc[r])
                        std::fprintf(stderr, "  vc%d %X %X\n", r, exact->regs.vc[r], fast->regs.vc[r]);
                if (exact->clip != fast->clip) std::fprintf(stderr, "  clip\n");
                if (std::memcmp(exact->data.data(), fast->data.data(), exact->data.size())) std::fprintf(stderr, "  data\n");
            }
            if (!exact->SameState(*fast)) ++flags_differ;
        }
    }
    char what[160];
    std::snprintf(what, sizeof what, "VU fast blocks run as the exact ones but for MAC and STATUS (%u of %u runs differ)",
                  wrong, runs);
    Check(wrong == 0 && runs >= 100, what);
    // The fast blocks do leave the flags alone (else they were not taken).
    std::snprintf(what, sizeof what, "VU fast blocks taken: the flags differ in %u of %u runs", flags_differ, runs);
    Check(flags_differ > runs / 4, what);
}

void VuTimingIsVisible() {
    ps2host::Vu vu(0);
    auto f = [](float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; };
    const std::uint32_t kNopU = 0x000002FF, kNopL = 0x8000033C, kI = 0x80000000, kEnd = 0x40000000;
    const std::uint32_t program[] = {
        0x100200FF, 0x0100006C,         // iaddiu vi2, vi0, 0xFF    sub.x vf1, vf0, vf0 (zero: MAC Zx)
        0x34011000, kNopU,              // fmand vi1, vi2: the flags from before
        0x81E003BC, kNopU,              // div Q, vf0w, vf0w
        kNopL, 0x002000DC,              // mulq.w vf3, vf0, Q: the Q from before
        0x34031000, kNopU,              // fmand vi3, vi2: four cycles on, the sub's
        0x800003BF, kNopU,              // waitq
        kNopL, 0x0020011C,              // mulq.w vf4, vf0, Q
        f(3), kI | kNopU,               // loi 3
        f(5), kI | 0x0020015E,          // loi 5                    muli.w vf5, vf0, I
        kNopL, kEnd | 0x0020019E,       //                          muli.w vf6, vf0, I [E]
        kNopL, kNopU};
    std::memcpy(vu.micro.data(), program, sizeof program);
    vu.Run(0);
    Check(vu.regs.vc[1] == 0 && vu.regs.vc[3] == 8, "VU: FMAND sees the MAC flags four cycles late");
    Check(vu.regs.vf[3][3] == 0 && vu.regs.vf[4][3] == f(1), "VU: Q arrives after DIV's latency; WAITQ waits");
    Check(vu.regs.vf[5][3] == f(3) && vu.regs.vf[6][3] == f(5), "VU: an upper beside LOI reads the old I");
}

// An upper operation into VF0 writes nowhere, but its flags are its result's:
// Burnout 3 tests a box against planes with `msubz.xyzw vf00, ...` and FMAND,
// and with VF0's 0, 0, 0, 1 read back instead culled and clipped nothing.
void VuFlagsOfAResultIntoVf0() {
    ps2host::Vu vu(0);
    auto f = [](float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return u; };
    const std::uint32_t kNopU = 0x000002FF, kNopL = 0x8000033C, kEnd = 0x40000000;
    const std::uint32_t program[] = {
        0x100200FF, 0x01E1002C,         // iaddiu vi2, vi0, 0xFF    sub.xyzw vf0, vf0, vf1
        kNopL, kNopU,
        kNopL, kNopU,
        kNopL, kNopU,
        0x34011000, kNopU,              // fmand vi1, vi2: the sub's flags
        kNopL, kEnd | kNopU,
        kNopL, kNopU};
    std::memcpy(vu.micro.data(), program, sizeof program);
    const std::uint32_t vf1[4] = {f(1), f(-2), f(0), f(3)};
    std::memcpy(vu.regs.vf[1], vf1, sizeof vf1);
    vu.Run(0);
    // 0 - (1, -2, 0, 3): x and w negative (MAC 0x80, 0x10), z zero (0x02).
    Check(vu.regs.vc[1] == 0x92, "VU: an upper into VF0 sets the flags of its result");
    Check(vu.regs.vf[0][0] == 0 && vu.regs.vf[0][2] == 0 && vu.regs.vf[0][3] == f(1), "VU: VF0 stays 0, 0, 0, 1");
}

// The IPU: FDEC and VDEC on known bits, and BDEC on two intra macroblocks of
// flat blocks (DC only), the second split over the input FIFO's quadwords
// so that it runs out of data after its first block's DC - retried, the
// DC predictor must not have moved (a retry once added the luma DC again
// each time, and a movie's slice went white from there).
void IpuDecodesAndRetriesCleanly() {
    ps2host::InstallIpu();
    ps2host::Memory& m = ps2host::GuestMemory();
    auto command = [&](u32 word) { m.Write(0x10002000u, word, 4); };
    auto busy = [&] { return (m.Read(0x10002000u, 8) >> 63) & 1; };
    std::vector<u8> bytes;
    auto pack = [&](const std::string& bits) {      // MSB first, as the stream is read
        bytes.assign((bits.size() + 7) / 8, 0);
        for (std::size_t i = 0; i < bits.size(); ++i)
            if (bits[i] == '1') bytes[i / 8] |= u8(0x80 >> (i % 8));
        bytes.resize((bytes.size() + 15) / 16 * 16, 0);
    };
    auto push = [&](std::size_t quadword) {
        for (u32 w = 0; w < 4; ++w) {
            u32 word = 0;
            std::memcpy(&word, &bytes[quadword * 16 + w * 4], 4);
            m.Write(0x10007010u + 4 * w, word, 4);
        }
    };
    auto first_luma = [&] {                         // the macroblock's Y[0][0], the rest drained
        const u64 q = m.Read(0x10007000u, 8);
        m.Read(0x10007008u, 8);
        for (int i = 1; i < 48; ++i) { m.Read(0x10007000u, 8); m.Read(0x10007008u, 8); }
        return s16(q & 0xFFFF);
    };
    m.Write(0x10002010u, 1u << 30, 4);              // reset; MPEG-2, IDP 0, zig-zag, B.14
    // FDEC and VDEC: 0x000001B3, then a 1-bit address increment of 1.
    pack("00000000000000000000000110110011" "1");
    command(0x00000000u);                           // BCLR, BP 0
    push(0);
    command(0x40000000u);                           // FDEC
    Check(!busy() && u32(m.Read(0x10002000u, 4)) == 0x000001B3u, "IPU: FDEC reads 32 bits");
    command(0x40000000u | 32);                      // FDEC past them
    command(0x30000000u);                           // VDEC, address increment
    Check(u32(m.Read(0x10002000u, 4)) == 0x00010001u, "IPU: VDEC gives the increment and its length");
    // Two macroblocks: luma blocks "100" (DC size 0) "10" (end of block),
    // chroma "00" "10"; the second's first luma DC "00" "1" (+1).
    const std::string flat = std::string("10010") + "10010" + "10010" + "10010" + "0010" + "0010";
    const std::string plus = std::string("00110") + "10010" + "10010" + "10010" + "0010" + "0010";
    // The first at bit 210, the second at 238: its first DC in the second
    // quadword, the end-of-block code after it running into the third.
    // Then the slice's stuffing to the byte and a start code (SCD).
    pack(std::string(210, '0') + flat + plus + "000000" + "00000000" "00000000" "00000001" "10110011");
    command(0x0000007Fu);                           // BCLR, BP 127
    push(0);
    push(1);
    command(0x40000000u | 63);                      // FDEC past 63 more: bit 190
    command(0x2C000000u | 20);                      // BDEC past 20: intra, DC reset, QSC 0
    Check(!busy() && first_luma() == 128, "IPU: BDEC of a flat intra block");
    command(0x28000000u);                           // BDEC: intra
    for (int i = 0; i < 5; ++i) busy();             // tried, and tried again, short of data
    Check(busy(), "IPU: BDEC waits for data");
    push(2);
    Check(!busy() && first_luma() == 129, "IPU: a retried BDEC keeps its DC predictor");
    Check(((m.Read(0x10002010u, 4) >> 15) & 1) && u32(m.Read(0x10002030u, 4)) == 0x000001B3u,
          "IPU: BDEC goes over a slice's stuffing to the start code (SCD)");
}

// MAX and MINI hand back an operand as it is: Burnout 3's car microcode
// builds its GIF tag's REGS (0x412, a denormal as a float) with LOI and MAXI;
// cleaned to 0, every vertex went to PRIM and no car was drawn.
// The MMI lane operations and PEXTL* in SSE2 (lifted_support.cpp) give
// what the lane-by-lane model gives, over random values and lanes at their
// limits (0, 1, the signed extremes, all ones).
void MmiLanesAreScalar() {
    u64 seed = 0x9E3779B97F4A7C15ull;
    auto next = [&] {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    static const u64 kEdges[] = {0, 1, 0x7F7F7F7F7F7F7F7Full, 0x8080808080808080ull, 0x7FFF7FFF7FFF7FFFull,
                                 0x8000800080008000ull, 0x7FFFFFFF7FFFFFFFull, 0x8000000080000000ull, ~0ull,
                                 0xFF00FF00FF00FF00ull};
    auto value = [&]() -> u64 {
        const u64 r = next();
        if (r % 3 == 0) return kEdges[(r >> 8) % (sizeof kEdges / sizeof kEdges[0])];
        if (r % 3 == 1) return next() & next();     // more zero lanes
        return next();
    };
    u32 wrong = 0, checked = 0;
    for (int op = lifted::kPmaxw; op <= lifted::kPminh; ++op)
        for (int trial = 0; trial < 4000; ++trial) {
            const u64 alo = value(), ahi = value(), blo = trial & 1 ? alo : value(), bhi = value();
            u64 lo1, hi1, lo2, hi2;
            lifted::Lanes(op, lo1, hi1, alo, ahi, blo, bhi);
            lifted::LanesScalar(op, lo2, hi2, alo, ahi, blo, bhi);
            ++checked;
            if (lo1 != lo2 || hi1 != hi2) ++wrong;
        }
    for (u32 width : {8u, 16u, 32u})
        for (int trial = 0; trial < 4000; ++trial) {
            const u64 a = value(), b = value();
            u64 lo1, hi1, lo2, hi2;
            lifted::Interleave(lo1, hi1, a, b, width);
            lifted::InterleaveScalar(lo2, hi2, a, b, width);
            ++checked;
            if (lo1 != lo2 || hi1 != hi2) ++wrong;
        }
    char what[128];
    std::snprintf(what, sizeof what, "MMI lanes in SSE2 are the model's (%u of %u differ)", wrong, checked);
    Check(wrong == 0, what);
}

void VuMaxKeepsADenormal() {
    ps2host::Vu vu(0);
    const std::uint32_t kNopU = 0x000002FF, kNopL = 0x8000033C, kI = 0x80000000, kEnd = 0x40000000;
    const std::uint32_t program[] = {
        0x00000412, kI | kNopU,         // loi 0x412
        kNopL, 0x0040005D,              //                          maxi.z vf1, vf0, I
        0x00000412, kI | kNopU,         // loi 0x412
        kNopL, 0x0040009F,              //                          minii.z vf2, vf0, I
        kNopL, kEnd | kNopU,
        kNopL, kNopU};
    std::memcpy(vu.micro.data(), program, sizeof program);
    vu.Run(0);
    Check(vu.regs.vf[1][2] == 0x412 && vu.regs.vf[2][2] == 0, "VU: MAXI and MINII compare and keep the bits");
}

// The IOP interpreter against emu.py, the verified R5900 interpreter: one MIPS I
// instruction per vector (engine/tools/iop_vectors.py).
struct CpuVector {
    const char* name;
    u32 word;
    u32 regs[32];
    u8 data[64];
    u32 out[32];
    u32 hi, lo;
    u8 data_out[64];
};

const CpuVector kCpuVectors[] = {
#include "iop_cpu_vectors.inc"
};

// The IOP library functions Stuntman's XAVSTRM.IRX needs beyond the rest
// (iop_libs.cpp): thfpool's fixed-size pools - blocks handed out, a thread
// waiting for one as WaitSema waits and woken with the block FreeFpl frees,
// the poll's KE_NO_MEMORY - sysclib's strstr and strtoul, cdvdman's sceCdMmode.
void IopPoolsAndLibraryCalls() {
    using namespace ps2host::iop;
    Vm& vm = TheVm();
    auto call = [&](const char* library, u32 index, std::vector<u32> args, Outcome* outcome = nullptr) {
        Context c;
        for (std::size_t i = 0; i < args.size() && i < 4; ++i) c.r[4 + i] = args[i];
        const Outcome o = FindLibrary(library, index)(vm, c);
        if (outcome) *outcome = o;
        return c.r[2];
    };
    const u32 param = 0x001F0000u, text = 0x001F0100u, end = 0x001F0200u;
    vm.Write(param, 0, 4);                                   // attr
    vm.Write(param + 4, 0, 4);                               // option
    vm.Write(param + 8, 30, 4);                              // block size: 32 once aligned
    vm.Write(param + 12, 2, 4);                              // blocks
    const u32 pool = call("thfpool", 4, {param});
    Check(s32(pool) > 0, "CreateFpl gives a pool");
    const u32 first = call("thfpool", 6, {pool}), second = call("thfpool", 6, {pool});
    Check(first && second == first + 32, "AllocateFpl gives word-aligned blocks in turn");
    Check(s32(call("thfpool", 7, {pool})) == -400, "pAllocateFpl with none free: KE_NO_MEMORY");
    // A thread waits for a block; another frees one, which goes to the waiter.
    vm.threads[900].id = 900;
    vm.threads[901].id = 901;
    const u32 saved_current = vm.current;
    vm.current = 900;
    Outcome waited = Outcome::Return;
    call("thfpool", 6, {pool}, &waited);
    Check(waited == Outcome::Block && vm.threads[900].wait == Wait::Fpl, "AllocateFpl with none free waits");
    vm.current = 901;
    Check(call("thfpool", 9, {pool, second}) == 0, "FreeFpl");
    Check(vm.threads[900].ready && vm.threads[900].c.r[2] == second, "the waiter wakes with the freed block");
    Check(s32(call("thfpool", 9, {pool, first + 4})) == -426, "FreeFpl of no block: KE_ILLEGAL_MEMBLOCK");
    Check(call("thfpool", 9, {pool, first}) == 0 && call("thfpool", 7, {pool}) == first,
          "a block freed with nobody waiting is given again");
    Check(call("thfpool", 5, {pool}) == 0 && s32(call("thfpool", 6, {pool})) == -412,
          "DeleteFpl, then the pool is unknown");
    vm.current = saved_current;
    vm.threads.erase(900);
    vm.threads.erase(901);
    vm.reschedule = false;

    const char* hay = "cdrom0:\\XA\\STREAM.XA;1";
    for (u32 i = 0; i <= std::strlen(hay); ++i) vm.Write(text + i, u8(hay[i]), 1);
    const char* needle = "STREAM";
    for (u32 i = 0; i <= std::strlen(needle); ++i) vm.Write(text + 0x40 + i, u8(needle[i]), 1);
    Check(call("sysclib", 34, {text, text + 0x40}) == text + 11, "strstr finds the substring");
    vm.Write(text + 0x40, 'Q', 1);
    Check(call("sysclib", 34, {text, text + 0x40}) == 0, "strstr: none");
    const char* number = "  0xFFFFFFF0z";
    for (u32 i = 0; i <= std::strlen(number); ++i) vm.Write(text + i, u8(number[i]), 1);
    Check(call("sysclib", 38, {text, end, 16}) == 0xFFFFFFF0u && vm.Read(end, 4) == text + 12,
          "strtoul: base 16 and the end it stopped at");
    const char* minus = "-1";
    for (u32 i = 0; i <= std::strlen(minus); ++i) vm.Write(text + i, u8(minus[i]), 1);
    Check(call("sysclib", 38, {text, 0, 10}) == 0xFFFFFFFFu, "strtoul of -1 wraps, as C's");
    Check(call("cdvdman", 75, {1}) == 1, "sceCdMmode succeeds");

    // CheckThreadStack: what is left of the caller's stack below $sp.
    vm.threads[902].id = 902;
    vm.threads[902].stack = 0x00180000u;
    vm.threads[902].stack_size = 0x1000u;
    vm.current = 902;
    Context t;
    t.r[29] = 0x00180C00u;
    FindLibrary("thbase", 21)(vm, t);
    Check(t.r[2] == 0xC00u, "CheckThreadStack: the bytes left below $sp");
    vm.current = 0;
    Context none;
    FindLibrary("thbase", 21)(vm, none);
    Check(s32(none.r[2]) == -100, "CheckThreadStack outside a thread: KE_ILLEGAL_CONTEXT");
    vm.current = saved_current;
    vm.threads.erase(902);
}

void IopInstructionsMatchTheR5900() {
    using namespace ps2host::iop;
    Vm& vm = TheVm();
    int failures = 0;
    for (const CpuVector& v : kCpuVectors) {
        vm.Write(0x00100000u, v.word, 4);
        for (u32 i = 0; i < 64; ++i) vm.Write(0x00100100u + i, v.data[i], 1);
        Context c;
        for (u32 i = 1; i < 32; ++i) c.r[i] = v.regs[i];
        c.hi = v.regs[0];
        c.lo = v.regs[0] ^ 0x5A5A5A5Au;
        c.pc = 0x00100000u;
        c.npc = 0x00100004u;
        vm.Step(c, 1);
        bool ok = c.hi == v.hi && c.lo == v.lo;
        for (u32 i = 1; i < 32; ++i) ok = ok && c.r[i] == v.out[i];
        for (u32 i = 0; i < 64; ++i) ok = ok && vm.Read(0x00100100u + i, 1) == v.data_out[i];
        if (!ok && failures++ < 5) {
            char what[96];
            std::snprintf(what, sizeof what, "IOP %s (0x%08X) matches the R5900", v.name, unsigned(v.word));
            Check(false, what);
        }
    }
    Check(failures == 0, "every IOP instruction vector matches");
}

}  // namespace

void SpuDecodesAdpcmAndPlaysAVoice() {
    using ps2host::iop::Spu2;
    // Shift 0, filter 0: a nibble is the sample's top four bits.
    Spu2::u8 block[16] = {0x00, 0x00, 0x81};
    Spu2::s32 out[28], older = 0, old = 0;
    Spu2::DecodeBlock(block, out, &older, &old);
    Check(out[0] == 4096 && out[1] == -32768 && out[2] == 0, "ADPCM: shift 0, filter 0");
    // Filter 1 predicts 60/64 of the previous sample.
    Spu2::u8 predicted[16] = {0x1C};
    older = 0;
    old = 64;
    Spu2::DecodeBlock(predicted, out, &older, &old);
    Check(out[0] == 60, "ADPCM: filter 1 predicts from the previous sample");

    // A voice looping one block of 4096s, at full volume once its attack is done.
    Spu2 spu;
    Spu2::u8 loop[16] = {0x00, 0x07};
    for (int i = 2; i < 16; ++i) loop[i] = 0x11;
    spu.Write(0x5000, loop, 16);
    std::vector<Spu2::s16> heard;
    spu.SetOutput([&](const Spu2::s16* frames, Spu2::u32 count) { heard.assign(frames, frames + 2 * count); });
    spu.SetAddr(0x2000, 0x5000);                    // SSA, voice 0 of core 0
    spu.SetParam(0x0000, 0x3FFF);                   // VOLL
    spu.SetParam(0x0100, 0x3FFF);                   // VOLR
    spu.SetParam(0x0200, 0x1000);                   // pitch: 48 kHz
    spu.SetParam(0x0300, 0x000F);                   // ADSR1: fastest attack, full sustain
    spu.SetParam(0x0400, 0x1FC0);                   // ADSR2: sustain holds
    spu.SetSwitch(0x1800, 1);                       // VMIXL
    spu.SetSwitch(0x1A00, 1);                       // VMIXR
    spu.SetSwitch(0x1500, 1);                       // KON
    auto none = [](Spu2::u32) {};
    spu.Advance(1000, none);
    spu.Advance(2000, none);                        // 48 samples
    Check(heard.size() == 96, "SPU2: a millisecond is 48 stereo frames");
    Check(heard.size() == 96 && heard[90] > 3500 && heard[90] < 4200 && heard[91] == heard[90],
          "SPU2: a voice sounds at its volume");
    Check(spu.GetSwitch(0x1700) & 1, "SPU2: ENDX after the loop's end");
    Check(spu.GetParam(0x0500) == 0x7FFF, "SPU2: ENVX after the attack");
    // A sweep: VOLL 0xA020 (linear, decreasing, rate 0x20 - a step of 64 a
    // sample) fades the voice from where it was to nothing in about 512
    // samples; VOLR stays. Burnout 3 quiets the voices it leaves running so:
    // played at full volume instead, a race's twenty of them clipped.
    spu.SetParam(0x0000, 0xA020);
    spu.Advance(3000, none);                        // 48 samples on
    const int fading = heard[90];
    spu.Advance(15000, none);                       // 576 more
    Check(fading > 2500 && fading < heard[91], "SPU2: a decreasing sweep lowers the volume from where it was");
    Check(heard[2 * 575] == 0 && heard[2 * 575 + 1] > 3500, "SPU2: the sweep reaches silence, the other side plays on");
    Check(spu.GetParam(0x0600) == 0, "SPU2: VOLXL, the current volume, is the sweep's");

    // The reverb: one loud sample sent only to it (VMIXEL/VMIXER, not the dry
    // mix) comes out later, and keeps coming out after the voice has ended.
    Spu2 wet;
    Spu2::u8 burst[16] = {0x00, 0x01};              // loop end, no repeat: one block, then silence
    for (int i = 2; i < 16; ++i) burst[i] = 0x77;   // 28 loud samples (the envelope rises through them)
    wet.Write(0x5000, burst, 16);
    std::vector<Spu2::s16> tail;
    wet.SetOutput([&](const Spu2::s16* frames, Spu2::u32 count) { tail.assign(frames, frames + 2 * count); });
    wet.SetAddr(0x2000, 0x5000);
    wet.SetParam(0x0000, 0x3FFF);
    wet.SetParam(0x0100, 0x3FFF);
    wet.SetParam(0x0200, 0x1000);
    wet.SetParam(0x0300, 0x000F);
    wet.SetParam(0x0400, 0x1FC0);
    wet.SetSwitch(0x1900, 1);                       // VMIXEL
    wet.SetSwitch(0x1B00, 1);                       // VMIXER
    Check(wet.SetEffectAttr(0, 0x101, 0x3FFF, 0x3FFF, 0, 0) == 0, "SPU2: the Room preset, its work area cleared");
    wet.SetCoreAttr(0x2, 1);                        // SD_CORE_EFFECT_ENABLE
    wet.SetSwitch(0x1500, 1);
    wet.Advance(1000, none);
    wet.Advance(201000, none);                      // 200 ms
    int loud_late = 0;
    for (std::size_t i = 2 * 4800; i < tail.size(); ++i) loud_late += std::abs(int(tail[i])) > 8;
    Check(tail.size() == 2 * 9600 && tail[0] == 0, "SPU2: nothing dry");
    Check(loud_late > 100, "SPU2: the reverb's tail, long after the sound ended");
    Check(wet.SetEffectAttr(0, 10, 0, 0, 0, 0) == -1, "SPU2: no eleventh preset");
}

int main() {
    MirrorsReachTheSameRam();
    LoadsExtendAsTheEeDoes();
    OutsideMemoryFaults();
    ScratchpadAndHooks();
    BiosIsReadOnly();
    Kseg1ReachesDeviceRegisters();
    DiscFindsFilesByPath();
    WideWritesReachEveryWordsHook();
    VifSkipsDataAndAppliesMaskPath3();
    DmaChainsAreWalked();
    VifInterruptBitStopsAfterItsCommand();
    DmaStallsAndGoesOn();
    WatchedReadsAreLogged();
    GsLayoutsArePermutations();
    GsDrawsASprite();
    GsFogsTowardsFogcol();
    GsLoadsACsm2Clut();
    Path3GoesOneGroupPerWindow();
    GsKeepsAFlatDepthExact();
    GsFillsTopLeft();
    GsTakesImagesInBlocks();
    TexturePackNamesAsPcsx2();
#if PS2HOST_VULKAN
    GpuMemorySwizzlesAsTheGs();
    GpuDecodesTexturesAsTheGs();
#endif
    VifRunsMicroprogramsOnAVu();
    MemoryCardKeepsSaves();
    IopImageGivesModuleReleases();
    StopIsWrittenUntilTheRunStops();
    Cop0CountFollowsTheClock();
    HostServersSendCommandsAfterTheirAnswer();
    FileIoCompletesAsynchronousRequests();
    PadmanSetActAlignIsDone();
    McservCountsFreeEntries();
    VuLanesAreTheModel();
    VuCompiledIsTheInterpreter();
    VuFastIsTheExactButFlags();
    VuTimingIsVisible();
    VuFlagsOfAResultIntoVf0();
    VuMaxKeepsADenormal();
    MmiLanesAreScalar();
    IpuDecodesAndRetriesCleanly();
    IopInstructionsMatchTheR5900();
    IopPoolsAndLibraryCalls();
    SpuDecodesAdpcmAndPlaysAVoice();
    std::printf(g_failures ? "host tests: %d failed\n" : "host tests: all passed\n", g_failures);
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
