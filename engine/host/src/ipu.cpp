/*
 * The IPU (ps2host/ipu.h): registers, FIFOs, its two DMA channels and the
 * commands, after the EE User's Manual's chapter 8 and ISO/IEC 13818-2.
 *
 *   IPU_CMD  (0x10002000) write: a command, CODE in bits 28-31; read: the
 *            last VDEC's or FDEC's result (bits 0-31), BUSY in bit 63
 *   IPU_CTRL (0x10002010) IFC 0-3, OFC 4-7, CBP 8-13, ECD 14, SCD 15, IDP
 *            16-17, AS 20, IVF 21, QST 22, MP1 23, PCT 24-26, RST 30 (write:
 *            reset), BUSY 31
 *   IPU_BP   (0x10002020) BP 0-6 (the bit position), IFC 8-11, FP 16-17
 *   IPU_TOP  (0x10002030) the next 32 bits of the bitstream, BUSY in bit 63
 *   0x10007000 the output FIFO (read), 0x10007010 the input FIFO (write)
 *
 * The bitstream is the input FIFO's 8 quadwords behind a 2-quadword buffer
 * (FP of them loaded, BP bits into the first used); DMA channel 4 keeps the
 * FIFO full. A command runs in steps - a header field, a macroblock - each
 * all or nothing: it reads ahead as far as it needs, through the FIFO and
 * on into what channel 4 has still to transfer (its source chain is walked
 * without moving it), and only a step that finishes takes the bits - the
 * channel's MADR, QWC and TADR moving by exactly the quadwords taken. A step
 * that runs out of data waits, BUSY, until more arrives. Results go to the
 * output FIFO, which channel 3 drains; a step waits while 8 quadwords sit
 * there undrained, as the 8-quadword FIFO would stall the decoder.
 *
 * Commands: BCLR, BDEC, VDEC, FDEC, SETIQ, SETVQ, CSC, PACK, SETTH. IDEC
 * (a whole intra slice to RGB) is not modelled: it is reported and ends at
 * once.
 */

#include "ps2host/ipu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

using Qword = std::array<u32, 4>;

// ---------------------------------------------------------------- tables

struct Vlc {
    u16 code;
    u8 length;
    s32 value;
};
constexpr s32 kEscape = -1, kEndOfBlock = -2;

#include "ipu_tables.inc"

// A VLC table as a lookup on the next 16 bits: the length (0: no such
// code) and the value.
struct Lookup {
    std::vector<u8> length;
    std::vector<s32> value;
};

template <std::size_t N>
Lookup Build(const Vlc (&codes)[N]) {
    Lookup t{std::vector<u8>(65536, 0), std::vector<s32>(65536, 0)};
    for (const Vlc& v : codes) {
        const u32 first = u32(v.code) << (16 - v.length), count = 1u << (16 - v.length);
        for (u32 i = 0; i < count; ++i) {
            t.length[first + i] = v.length;
            t.value[first + i] = v.value;
        }
    }
    return t;
}

const Lookup& DcLuminance() { static const Lookup t = Build(kDcLuminance); return t; }
const Lookup& DcChrominance() { static const Lookup t = Build(kDcChrominance); return t; }
const Lookup& AddressIncrement() { static const Lookup t = Build(kAddressIncrement); return t; }
const Lookup& BlockPattern() { static const Lookup t = Build(kBlockPattern); return t; }
const Lookup& MotionCode() { static const Lookup t = Build(kMotionCode); return t; }
const Lookup& CoefficientsB14() { static const Lookup t = Build(kCoefficientsB14); return t; }
const Lookup& CoefficientsB15() { static const Lookup t = Build(kCoefficientsB15); return t; }

// B.2-B.4 macroblock_type, by picture type (1 I, 2 P, 3 B, 4 D): value =
// quant 0x10 | motion_forward 8 | motion_backward 4 | pattern 2 | intra 1.
const Vlc kTypeI[] = {{0x1, 1, 0x01}, {0x1, 2, 0x11}};
const Vlc kTypeP[] = {{0x1, 1, 0x0A}, {0x1, 2, 0x02}, {0x1, 3, 0x08}, {0x3, 5, 0x01},
                      {0x2, 5, 0x1A}, {0x1, 5, 0x12}, {0x1, 6, 0x11}};
const Vlc kTypeB[] = {{0x2, 2, 0x0C}, {0x3, 2, 0x0E}, {0x2, 3, 0x04}, {0x3, 3, 0x06},
                      {0x2, 4, 0x08}, {0x3, 4, 0x0A}, {0x3, 5, 0x01}, {0x2, 5, 0x1E},
                      {0x3, 6, 0x1A}, {0x2, 6, 0x16}, {0x1, 6, 0x11}};
const Vlc kTypeD[] = {{0x1, 1, 0x01}};
const Lookup& MacroblockType(u32 picture) {
    static const Lookup i = Build(kTypeI), p = Build(kTypeP), b = Build(kTypeB), d = Build(kTypeD);
    return picture == 2 ? p : picture == 3 ? b : picture == 4 ? d : i;
}

// Coefficient order: the zig-zag scan and MPEG-2's alternate scan, as the
// position in the block (row * 8 + column) of each coefficient in turn.
constexpr u8 kZigZag[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                            12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                            35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
constexpr u8 kAlternate[64] = {0,  8,  16, 24, 1,  9,  2,  10, 17, 25, 32, 40, 48, 56, 57, 49,
                               41, 33, 26, 18, 3,  11, 4,  12, 19, 27, 34, 42, 50, 58, 35, 43,
                               51, 59, 20, 28, 5,  13, 6,  14, 21, 29, 36, 44, 52, 60, 37, 45,
                               53, 61, 22, 30, 7,  15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63};
// quantiser_scale_code to quantiser_scale, q_scale_type 1 (table 7-6).
constexpr u8 kNonLinearScale[32] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  10, 12, 14, 16, 18,  20,  22,
                                    24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112};

// The IDCT, separable, in double precision and rounded once (IEEE 1180's
// reference): out[y][x] = sum C(u) C(v) / 4 F[v][u] cos((2x+1)u pi/16) cos((2y+1)v pi/16).
void Idct(const s32 in[64], s32 out[64]) {
    static const auto basis = [] {
        std::array<std::array<double, 8>, 8> c{};
        const double pi = 3.14159265358979323846;
        for (int x = 0; x < 8; ++x)
            for (int u = 0; u < 8; ++u)
                c[x][u] = 0.5 * (u == 0 ? 1.0 / std::sqrt(2.0) : 1.0) * std::cos((2 * x + 1) * u * pi / 16.0);
        return c;
    }();
    // Most coefficients are zero, and a zero term adds nothing to a sum (a
    // signed zero at most, which no later sum or the rounding sees): only the
    // others are summed, in the same order - the same result, bit for bit.
    double rows[64];
    int terms[8], live[8], live_rows = 0;
    for (int v = 0; v < 8; ++v) {
        int n = 0;
        for (int u = 0; u < 8; ++u)
            if (in[v * 8 + u]) terms[n++] = u;
        if (n) live[live_rows++] = v;
        for (int x = 0; x < 8; ++x) {
            double sum = 0;
            for (int k = 0; k < n; ++k) sum += basis[x][terms[k]] * in[v * 8 + terms[k]];
            rows[v * 8 + x] = sum;
        }
    }
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            double sum = 0;
            for (int k = 0; k < live_rows; ++k) sum += basis[y][live[k]] * rows[live[k] * 8 + x];
            out[y * 8 + x] = std::clamp(s32(std::floor(sum + 0.5)), -256, 255);
        }
}

// ---------------------------------------------------------------- DMA

constexpr u32 kFromIpu = 0x1000B000u, kToIpu = 0x1000B400u;
constexpr u32 kChcr = 0, kMadr = 0x10, kQwc = 0x20, kTadr = 0x30, kAsr0 = 0x40, kAsr1 = 0x50;

/* A DMA address: bit 31 (SPR) is the scratchpad's, the rest main memory. */
u32 Physical(u32 address) {
    return (address >> 31) ? (0x70000000u | (address & 0x3FF0u)) : (address & 0x01FFFFF0u);
}

u32 Reg(u32 address) { return u32(GuestMemory().Read(address, 4)); }
void SetReg(u32 address, u32 value) {
    *reinterpret_cast<u32*>(GuestMemory().Translate(address, 4)) = value;   // no hooks: the channel's own state
}

/* Channel 4's source: normal mode (MADR, QWC) or a source chain (TADR's
   tags), a quadword at a time. A copy walks ahead without moving the
   channel. */
struct Source {
    u32 madr = 0, qwc = 0, tadr = 0, asr[2] = {0, 0}, asp = 0;
    u32 tag_bits = 0;       // the last tag's bits 16-31, as CHCR's TAG field keeps them
    bool chain = false, last = true, tie = false;
    u32 tag_at = 0;         // where the last tag read was (Physical), and the tag
    u64 tag_read = 0;
    bool tag_seen = false;

    bool Next(Qword* q) {
        Memory& m = GuestMemory();
        while (qwc == 0) {
            if (!chain || last) return false;
            tag_at = Physical(tadr);
            const u64 tag = m.Read(tag_at, 8);
            tag_read = tag;
            tag_seen = true;
            const u32 count = u32(tag & 0xFFFF), id = u32(tag >> 28) & 7;
            const u32 addr = u32(tag >> 32) & 0x7FFFFFF0u;
            const u32 at = (tag >> 63) ? (0x80000000u | addr) : addr;     // SPR: the scratchpad (Physical)
            switch (id) {
                case 0: madr = at; last = true; tadr += 16; break;                      // refe
                case 1: madr = tadr + 16; tadr = madr + count * 16; break;              // cnt
                case 2: madr = tadr + 16; tadr = at; break;                             // next
                case 3: case 4: madr = at; tadr += 16; break;                           // ref, refs
                case 5:                                                                  // call
                    madr = tadr + 16;
                    if (asp < 2) asr[asp++] = madr + count * 16;
                    tadr = at;
                    break;
                case 6:                                                                  // ret
                    madr = tadr + 16;
                    if (asp > 0) tadr = asr[--asp];
                    else last = true;
                    break;
                default: madr = tadr + 16; last = true; break;                          // end
            }
            qwc = count;
            tag_bits = u32(tag) >> 16;
            if (tie && (tag >> 31) & 1) last = true;                                    // IRQ, with TIE
        }
        const u8* bytes = m.Translate(Physical(madr), 16);
        if (!bytes) throw MemoryFault{Physical(madr), false};
        std::memcpy(q->data(), bytes, 16);
        madr += 16;
        --qwc;
        return true;
    }
};

// ---------------------------------------------------------------- the unit

struct NeedData {};       // a step ran out of bitstream: it waits

class Ipu {
public:
    // Registers.
    u32 cmd_data = 0;
    u32 ctrl = 0;                       // the writable fields and CBP, ECD, SCD
    u32 th0 = 0, th1 = 0;
    u8 iq[2][64] = {};                  // intra, non-intra (as transmitted: scan order)
    u16 vqclut[16] = {};

    // The bitstream: the buffer (FP quadwords, BP bits into the first) and the FIFO.
    std::deque<Qword> buffer, fifo;
    u32 bp = 0;
    std::deque<Qword> out;              // the output FIFO (more, while a channel is not draining)

    // The command.
    u32 command = 0;
    bool busy = false;
    u32 left = 0;                       // CSC and PACK: macroblocks still to do

    // The channels.
    Source to;                          // channel 4, while it runs
    bool to_running = false;
    bool from_running = false;
    s32 dc_pred[3] = {128, 128, 128};

    /* A step that ran out of bitstream ran out where the buffer, the FIFO
       and channel 4 (a chain to the tag that ended it) end; libmpeg polls
       IPU_CTRL and IPU_CMD meanwhile, and running the step again from the
       start each time was most of a movie's time. So it is tried again only
       when one of those has changed: the command, IPU_CTRL, the buffer's and FIFO's
       lengths, the channel's registers, or that tag in memory. */
    struct Stalled {
        u64 commands;
        u32 ctrl;
        std::size_t buffer, fifo;
        u32 bp, madr, qwc, tadr, asp;
        bool running, last;
        bool tag_seen;
        u32 tag_at;
        u64 tag;
    };
    u64 commands = 0;
    bool stalled = false;
    Stalled stall{};
    Stalled Now(const Source& walked) const {
        return Stalled{commands, ctrl, buffer.size(), fifo.size(), bp, to.madr, to.qwc, to.tadr, to.asp, to_running,
                       to.last, walked.tag_seen, walked.tag_at, walked.tag_read};
    }
    bool StillStalled() const {
        if (!stalled) return false;
        const Stalled& s = stall;
        if (s.commands != commands || s.ctrl != ctrl || s.buffer != buffer.size() || s.fifo != fifo.size() || s.bp != bp ||
            s.madr != to.madr || s.qwc != to.qwc || s.tadr != to.tadr || s.asp != to.asp ||
            s.running != to_running || s.last != to.last)
            return false;
        return !s.tag_seen || GuestMemory().Read(s.tag_at, 8) == s.tag;
    }

    void Reset() {
        buffer.clear();
        fifo.clear();
        out.clear();
        bp = 0;
        busy = false;
        ctrl &= ~((1u << 14) | (1u << 15) | (0x3Fu << 8));
        cmd_data = 0;
    }

    // ------------------------------------------------ reading the bitstream
    /* A step's view of the bitstream: from BP on, through the buffer, the
       FIFO and what channel 4 would transfer next. */
    struct Reader {
        Ipu& ipu;
        u64 bit;                        // from the start of the buffer's first quadword
        std::vector<Qword> ahead;       // beyond the FIFO, from a copy of channel 4
        Source source;
        bool source_on;
        bool ran_dry = false;           // it wanted more than channel 4 has left to send

        explicit Reader(Ipu& unit) : ipu(unit), bit(unit.bp), source(unit.to), source_on(unit.to_running) {}

        const Qword& At(std::size_t index) {
            if (index < ipu.buffer.size()) return ipu.buffer[index];
            index -= ipu.buffer.size();
            if (index < ipu.fifo.size()) return ipu.fifo[index];
            index -= ipu.fifo.size();
            while (ahead.size() <= index) {
                Qword q;
                if (!source_on) throw NeedData{};
                if (!source.Next(&q)) {
                    ran_dry = true;
                    throw NeedData{};
                }
                ahead.push_back(q);
            }
            return ahead[index];
        }
        u8 Byte(u64 at) {
            const Qword& q = At(std::size_t(at >> 4));
            return u8(q[(at >> 2) & 3] >> (8 * (at & 3)));
        }
        // The quadword `cached` and (when `cached_next`) the one after it, as
        // bytes in memory order, eight more to spare for the window's load.
        std::size_t cached = ~std::size_t(0);
        bool cached_next = false;
        u8 bytes_[40] = {};
        u32 Peek(u32 n) {               // n <= 32: the next n bits, the first the highest
            if (n == 0) return 0;
            const std::size_t index = std::size_t(bit >> 7);
            const u32 offset = u32(bit >> 3) & 15;
            const u32 skew = u32(bit & 7);
            const u32 count = (skew + n + 7) / 8;           // the bytes the bits are in
            if (index != cached) {
                std::memcpy(bytes_, At(index).data(), 16);
                cached = index;
                cached_next = false;
            }
            if (offset + count > 16 && !cached_next) {      // into the next quadword
                std::memcpy(bytes_ + 16, At(index + 1).data(), 16);
                cached_next = true;
            }
            u64 window;
            std::memcpy(&window, bytes_ + offset, 8);
#if defined(_MSC_VER)
            window = _byteswap_uint64(window);
#else
            window = __builtin_bswap64(window);
#endif
            return u32((window << skew) >> (64 - n));
        }
        void Skip(u32 n) { bit += n; }
        u32 Get(u32 n) {
            const u32 v = Peek(n);
            bit += n;
            return v;
        }
        s32 Signed(u32 n) { return s32(Get(n) << (32 - n)) >> (32 - n); }
        /* A VLC symbol: its value, and its length in `*length` (0: none such). */
        s32 Vlc(const Lookup& t, u32* length) {
            const u32 code = Peek(16);
            *length = t.length[code];
            if (*length) bit += *length;
            return t.value[code];
        }
    };

    /* What a step took: drop the bits, quadwords leaving the buffer and FIFO
       (and channel 4, for those read ahead), the buffer refilled. */
    void Take(const Reader& r) {
        u64 quadwords = r.bit >> 7;
        bp = u32(r.bit & 127);
        while (quadwords && !buffer.empty()) { buffer.pop_front(); --quadwords; }
        while (quadwords && !fifo.empty()) { fifo.pop_front(); --quadwords; }
        while (quadwords) {
            Qword q;
            if (!to.Next(&q)) break;
            --quadwords;
            SyncToRegisters();
        }
        Fill();
    }

    /* The buffer from the FIFO (2 quadwords), the FIFO from channel 4. */
    void Fill() {
        for (;;) {
            while (buffer.size() < 2 && !fifo.empty()) {
                buffer.push_back(fifo.front());
                fifo.pop_front();
            }
            if (!to_running || fifo.size() >= 8) break;
            Qword q;
            if (!to.Next(&q)) {
                EndTo();
                break;
            }
            fifo.push_back(q);
            SyncToRegisters();
        }
    }

    void SyncToRegisters() {
        if (to.chain) SetReg(kToIpu + kChcr, (Reg(kToIpu + kChcr) & 0xFFFFu) | (to.tag_bits << 16));
        SetReg(kToIpu + kMadr, to.madr);
        SetReg(kToIpu + kQwc, to.qwc);
        SetReg(kToIpu + kTadr, to.tadr);
    }
    void EndTo() {
        to_running = false;
        SetReg(kToIpu + kChcr, Reg(kToIpu + kChcr) & ~0x100u);
        DmaChannelEnded(4);
    }

    void StartTo(u32 chcr) {
        to.madr = Reg(kToIpu + kMadr);
        to.qwc = Reg(kToIpu + kQwc) & 0xFFFF;
        to.tadr = Reg(kToIpu + kTadr);
        to.asr[0] = Reg(kToIpu + kAsr0);
        to.asr[1] = Reg(kToIpu + kAsr1);
        to.asp = (chcr >> 4) & 3;
        to.chain = ((chcr >> 2) & 3) == 1;
        to.last = !to.chain;
        // Restarted in a block (QWC left): the tag it came from, in CHCR's
        // TAG field, says whether the chain ends after it - refe and end do,
        // and an IRQ tag with TIE. libmpeg stops and restarts the channel
        // around its other transfers; going on past a refe there ran the
        // stream on past where the data stops.
        to.tag_bits = chcr >> 16;
        if (to.chain && to.qwc) {
            const u32 id = (chcr >> 28) & 7;
            if (id == 0 || id == 7 || (((chcr >> 31) & 1) && ((chcr >> 7) & 1))) to.last = true;
        }
        to.tie = (chcr >> 7) & 1;
        to_running = true;
    }

    /* Channel 3: the output FIFO to memory, while QWC lasts. */
    void Drain() {
        if (!from_running) return;
        u32 madr = Reg(kFromIpu + kMadr), qwc = Reg(kFromIpu + kQwc) & 0xFFFF;
        Memory& m = GuestMemory();
        while (qwc && !out.empty()) {
            const Qword& q = out.front();
            u8* bytes = m.Translate(Physical(madr), 16);
            if (!bytes) throw MemoryFault{Physical(madr), true};
            std::memcpy(bytes, q.data(), 16);
            out.pop_front();
            madr += 16;
            --qwc;
        }
        SetReg(kFromIpu + kMadr, madr);
        SetReg(kFromIpu + kQwc, qwc);
        if (qwc == 0) {
            from_running = false;
            SetReg(kFromIpu + kChcr, Reg(kFromIpu + kChcr) & ~0x100u);
            DmaChannelEnded(3);
        }
    }

    // ------------------------------------------------ the commands
    void Command(u32 word) {
        ++commands;
        command = word;
        busy = true;
        ctrl &= ~(1u << 14);                            // ECD
        const u32 code = word >> 28;
        if (code == 0) {                                // BCLR: empty the FIFO, BP as given
            buffer.clear();
            fifo.clear();
            bp = word & 0x7F;
            busy = false;
            Fill();
            return;
        }
        if (code == 7 || code == 8) left = word & 0x7FF;
        if (code == 9) {                                // SETTH
            th0 = word & 0x1FF;
            th1 = (word >> 16) & 0x1FF;
            busy = false;
            return;
        }
        if (code == 1) {
            static bool reported = false;
            if (!reported) std::fprintf(stderr, "[ipu] IDEC is not modelled (the command ends at once)\n");
            reported = true;
            busy = false;
        }
    }

    /* One step of the command: true if it moved. */
    bool Step() {
        if (!busy) return false;
        if (out.size() >= 8) return false;              // the output FIFO is full
        if (StillStalled()) return false;
        stalled = false;
        Reader r(*this);
        const u32 code = command >> 28;
        // A step that runs out of data is tried again from the start when
        // more arrives, so what it changed on the way is put back: the DC
        // predictors above all (a macroblock retried while the stream
        // trickled in added its luma DC differentials each time, and the
        // slice went white from there).
        const u32 ctrl_before = ctrl;
        s32 dc_before[3];
        std::copy(dc_pred, dc_pred + 3, dc_before);
        try {
            switch (code) {
                case 2: Bdec(r); break;
                case 3: Vdec(r); break;
                case 4: Fdec(r); break;
                case 5: Setiq(r); break;
                case 6: Setvq(r); break;
                case 7: Csc(r); break;
                case 8: Pack(r); break;
                default: busy = false; return true;
            }
        } catch (const NeedData&) {
            ctrl = ctrl_before;
            std::copy(dc_before, dc_before + 3, dc_pred);
            if (r.ran_dry && to_running) {
                // The IPU takes its bits as it decodes, so a macroblock longer
                // than the FIFO keeps channel 4 going to the end of what it was
                // given, and the channel ends (its interrupt is when libmpeg's
                // caller hands it more). Decoded whole here, the macroblock
                // would hold a full FIFO and the channel for ever: the RACE
                // TRAINING movie stopped there. So the channel sends the rest
                // now, past the FIFO's eight quadwords.
                Qword q;
                while (to.Next(&q)) fifo.push_back(q);
                SyncToRegisters();
                EndTo();
                Fill();
            }
            stall = Now(r.source);
            stalled = true;
            return false;
        }
        Take(r);
        return true;
    }

    u32 Fb() const { return command & 0x3F; }

    void Fdec(Reader& r) {
        r.Skip(Fb());
        cmd_data = r.Peek(32);
        busy = false;
    }

    void Vdec(Reader& r) {
        r.Skip(Fb());
        const u32 table = (command >> 26) & 3;
        u32 length = 0;
        s32 value = 0;
        switch (table) {
            case 0: value = r.Vlc(AddressIncrement(), &length); break;
            case 1: value = r.Vlc(MacroblockType(Pct()), &length); break;
            case 2:
                value = r.Vlc(MotionCode(), &length);
                if (length && value) {
                    if (r.Get(1)) value = -value;
                    ++length;
                }
                break;
            default: {                                  // B.11 dmvector: 0, 10 (+1), 11 (-1)
                const u32 two = r.Peek(2);
                if (two < 2) { value = 0; length = 1; }
                else { value = two == 2 ? 1 : -1; length = 2; }
                r.Skip(length);
                break;
            }
        }
        if (!length) ctrl |= 1u << 14;                  // ECD: no such code
        cmd_data = (u32(value) & 0xFFFF) | (length << 16);
        busy = false;
    }

    void Setiq(Reader& r) {
        r.Skip(Fb());
        u8* matrix = iq[(command >> 27) & 1];
        for (int i = 0; i < 64; ++i) matrix[i] = u8(r.Get(8));
        busy = false;
    }

    void Setvq(Reader& r) {
        r.Skip(Fb());
        for (int i = 0; i < 16; ++i) vqclut[i] = u16(r.Get(16) >> 0);
        // As transmitted, each entry's two bytes in memory order.
        for (u16& e : vqclut) e = u16((e >> 8) | (e << 8));
        busy = false;
    }

    u32 Pct() const { return (ctrl >> 24) & 7; }
    bool Mpeg1() const { return (ctrl >> 23) & 1; }

    /* One block's coefficients, dequantised, in natural order. */
    void Block(Reader& r, bool intra, int component, u32 scale, s32 coefficients[64]) {
        std::fill(coefficients, coefficients + 64, 0);
        const u8* scan = ((ctrl >> 20) & 1) ? kAlternate : kZigZag;
        const u8* matrix = iq[intra ? 0 : 1];
        const bool mpeg1 = Mpeg1();
        int index = 0;
        if (intra) {
            u32 length = 0;
            const s32 size = r.Vlc(component == 0 ? DcLuminance() : DcChrominance(), &length);
            if (!length) { ctrl |= 1u << 14; return; }
            s32 differential = 0;
            if (size) {
                const u32 bits = r.Get(u32(size));
                differential = (bits >> (size - 1)) ? s32(bits) : s32(bits) - (1 << size) + 1;
            }
            dc_pred[component] += differential;
            coefficients[0] = dc_pred[component] << (3 - ((ctrl >> 16) & 3));
            index = 1;
        }
        const Lookup& table = (intra && ((ctrl >> 21) & 1)) ? CoefficientsB15() : CoefficientsB14();
        bool first = !intra;
        for (;;) {
            s32 run, level;
            if (first && r.Peek(1)) {                   // the first coefficient: "1s" is run 0, level 1
                r.Skip(1);
                run = 0;
                level = r.Get(1) ? -1 : 1;
            } else {
                u32 length = 0;
                const s32 value = r.Vlc(table, &length);
                if (!length) { ctrl |= 1u << 14; break; }
                if (value == kEndOfBlock) break;
                if (value == kEscape) {
                    run = s32(r.Get(6));
                    if (mpeg1) {
                        level = r.Signed(8);
                        if (level == -128) level = s32(r.Get(8)) - 256;
                        else if (level == 0) level = s32(r.Get(8));
                    } else {
                        level = r.Signed(12);
                    }
                } else {
                    run = value >> 8;
                    level = value & 0xFF;
                    if (r.Get(1)) level = -level;
                }
            }
            first = false;
            index += run;
            if (index > 63) { ctrl |= 1u << 14; break; }
            const int position = scan[index];
            const s32 magnitude = level < 0 ? -level : level;
            s32 value = intra ? (magnitude * s32(scale) * matrix[index]) >> 4
                              : ((2 * magnitude + 1) * s32(scale) * matrix[index]) >> 5;
            if (mpeg1 && value) value = (value - 1) | 1;                // oddification
            if (level < 0) value = -value;
            coefficients[position] = std::clamp(value, -2048, 2047);
            ++index;
        }
        if (!mpeg1) {                                    // mismatch control
            s32 sum = 0;
            for (int i = 0; i < 64; ++i) sum += coefficients[i];
            if ((sum & 1) == 0) coefficients[63] ^= 1;
        }
    }

    void Bdec(Reader& r) {
        r.Skip(Fb());
        const u32 code = (command >> 16) & 0x1F;
        const u32 scale = ((ctrl >> 22) & 1) ? kNonLinearScale[code] : code << 1;
        const bool field = (command >> 25) & 1, reset = (command >> 26) & 1, intra = (command >> 27) & 1;
        if (reset) dc_pred[0] = dc_pred[1] = dc_pred[2] = 128 << ((ctrl >> 16) & 3);
        u32 pattern = 63;
        if (!intra) {
            u32 length = 0;
            pattern = u32(r.Vlc(BlockPattern(), &length));
            if (!length) { ctrl |= 1u << 14; pattern = 0; }
            ctrl = (ctrl & ~(0x3Fu << 8)) | (pattern << 8);
        }
        s16 y[16][16] = {}, cb[64] = {}, cr[64] = {};
        for (int b = 0; b < 6; ++b) {
            if (!(pattern & (32u >> b))) continue;
            s32 coefficients[64], pixels[64];
            Block(r, intra, b < 4 ? 0 : b - 3, scale, coefficients);
            Idct(coefficients, pixels);
            if (intra)
                for (s32& p : pixels) p = std::clamp(p, 0, 255);
            if (b < 4) {
                const int x0 = (b & 1) * 8;
                for (int row = 0; row < 8; ++row) {
                    // Frame DCT: blocks are the quarters; field DCT: block
                    // 0 and 1 the even lines, 2 and 3 the odd.
                    const int line = field ? (row * 2 + (b >> 1)) : (row + (b >> 1) * 8);
                    for (int col = 0; col < 8; ++col) y[line][x0 + col] = s16(pixels[row * 8 + col]);
                }
            } else {
                s16* to = b == 4 ? cb : cr;
                for (int i = 0; i < 64; ++i) to[i] = s16(pixels[i]);
            }
        }
        // RAW16: Y (16 x 16), Cb (8 x 8), Cr (8 x 8), 384 halfwords.
        std::vector<s16> halfwords;
        halfwords.reserve(384);
        for (auto& row : y) halfwords.insert(halfwords.end(), row, row + 16);
        halfwords.insert(halfwords.end(), cb, cb + 64);
        halfwords.insert(halfwords.end(), cr, cr + 64);
        // Past the macroblock: zero bits are a slice's stuffing - the IPU
        // goes to the byte boundary and over the zero bytes to the next start
        // code and says so (SCD), or flags an error (ECD) at anything else.
        // libmpeg ends the slice there; without it the start-code search ran
        // on from inside the stuffing and could miss the next slice's header.
        ctrl &= ~(1u << 15);
        if (r.Peek(8) == 0) {
            r.bit = (r.bit + 7) & ~u64(7);
            for (;;) {
                const u32 next = r.Peek(24);
                if (next != 0) {
                    ctrl |= next == 1 ? (1u << 15) : (1u << 14);
                    break;
                }
                r.Skip(8);
            }
        }
        Emit(reinterpret_cast<const u8*>(halfwords.data()), 768);
        busy = false;
    }

    void Emit(const u8* bytes, std::size_t count) {
        for (std::size_t i = 0; i < count; i += 16) {
            Qword q;
            std::memcpy(q.data(), bytes + i, 16);
            out.push_back(q);
        }
    }

    /* The thresholds' alpha for a pixel: 0 below TH0, half below TH1. */
    u8 Alpha(int r, int g, int b) const {
        if (r < int(th0) && g < int(th0) && b < int(th0)) return 0;
        if (r < int(th1) && g < int(th1) && b < int(th1)) return 0x40;
        return 0x80;
    }

    void Csc(Reader& r) {
        if (left == 0) { busy = false; return; }
        u8 raw[384];
        for (int i = 0; i < 384; i += 4) {             // four bytes a read, the first the highest
            const u32 v = r.Get(32);
            raw[i] = u8(v >> 24);
            raw[i + 1] = u8(v >> 16);
            raw[i + 2] = u8(v >> 8);
            raw[i + 3] = u8(v);
        }
        const bool rgb16 = (command >> 27) & 1, dither = (command >> 26) & 1;
        static const int kDither[4][4] = {{-4, 0, -3, 1}, {2, -2, 3, -1}, {-3, 1, -4, 0}, {3, -1, 2, -2}};
        u32 rgba[256];
        u16 packed[256];
        static const auto kLuma = [] {
            std::array<int, 256> t{};
            for (int y = 0; y < 256; ++y) t[y] = (0x95 * std::max(0, y - 16)) >> 6;
            return t;
        }();
        int red_of[64], green_of[64], blue_of[64];     // each chroma sample's terms
        for (int i = 0; i < 64; ++i) {
            const int cb = int(raw[256 + i]) - 128, cr = int(raw[320 + i]) - 128;
            red_of[i] = ((0xCC * cr) >> 6) + 1;
            green_of[i] = ((-0x68 * cr) >> 6) + ((-0x32 * cb) >> 6) + 1;
            blue_of[i] = ((0x102 * cb) >> 6) + 1;
        }
        for (int py = 0; py < 16; ++py)
            for (int px = 0; px < 16; ++px) {
                const int luma = kLuma[raw[py * 16 + px]];
                const int c = (py >> 1) * 8 + (px >> 1);
                const int red = std::clamp((luma + red_of[c]) >> 1, 0, 255);
                const int green = std::clamp((luma + green_of[c]) >> 1, 0, 255);
                const int blue = std::clamp((luma + blue_of[c]) >> 1, 0, 255);
                const u8 alpha = Alpha(red, green, blue);
                rgba[py * 16 + px] = u32(red) | u32(green) << 8 | u32(blue) << 16 | u32(alpha) << 24;
                if (rgb16) {
                    const int d = dither ? kDither[py & 3][px & 3] : 0;
                    const int r5 = std::clamp(red + d, 0, 255) >> 3, g5 = std::clamp(green + d, 0, 255) >> 3,
                              b5 = std::clamp(blue + d, 0, 255) >> 3;
                    packed[py * 16 + px] = u16(r5 | g5 << 5 | b5 << 10 | (alpha ? 0x8000 : 0));
                }
            }
        if (rgb16) Emit(reinterpret_cast<const u8*>(packed), sizeof packed);
        else Emit(reinterpret_cast<const u8*>(rgba), sizeof rgba);
        if (--left == 0) busy = false;
    }

    void Pack(Reader& r) {
        if (left == 0) { busy = false; return; }
        u32 rgba[256];
        for (u32& p : rgba) {
            u32 v = 0;
            for (int i = 0; i < 4; ++i) v |= r.Get(8) << (8 * i);
            p = v;
        }
        const bool rgb16 = (command >> 27) & 1;
        if (rgb16) {
            u16 packed[256];
            for (int i = 0; i < 256; ++i) {
                const u32 p = rgba[i];
                packed[i] = u16(((p >> 3) & 0x1F) | ((p >> 11) & 0x1F) << 5 | ((p >> 19) & 0x1F) << 10 |
                                ((p >> 31) ? 0x8000 : ((p >> 24) ? 0x8000 : 0)));
            }
            Emit(reinterpret_cast<const u8*>(packed), sizeof packed);
        } else {
            // INDX4: the nearest of VQCLUT's 16 colours (RGB 5-5-5).
            u8 indices[128] = {};
            for (int i = 0; i < 256; ++i) {
                const u32 p = rgba[i];
                const int pr = (p >> 3) & 0x1F, pg = (p >> 11) & 0x1F, pb = (p >> 19) & 0x1F;
                int best = 0, distance = 1 << 30;
                for (int c = 0; c < 16; ++c) {
                    const int dr = pr - (vqclut[c] & 0x1F), dg = pg - ((vqclut[c] >> 5) & 0x1F),
                              db = pb - ((vqclut[c] >> 10) & 0x1F);
                    const int d = dr * dr + dg * dg + db * db;
                    if (d < distance) { distance = d; best = c; }
                }
                indices[i / 2] |= u8(best << ((i & 1) * 4));
            }
            Emit(indices, sizeof indices);
        }
        if (--left == 0) busy = false;
    }

    // ------------------------------------------------ running
    void Pump() {
        for (int guard = 0; guard < 1000000; ++guard) {
            Fill();
            Drain();
            if (!Step()) {
                Drain();
                if (!Step()) break;
            }
        }
        Fill();
        Drain();
    }

    u64 Top() {
        // Only what has arrived: the 32 bits from BP, in the bytes Peek reads.
        const u64 last_byte = (u64(bp) >> 3) + (u64(bp & 7) + 32 + 7) / 8 - 1;
        if ((last_byte >> 4) >= buffer.size() + fifo.size()) return 1ull << 63;
        Reader r(*this);
        r.source_on = false;
        return r.Peek(32);
    }
};

Ipu& Unit() {
    static Ipu ipu;
    return ipu;
}

}  // namespace

void InstallIpu() {
    Memory& m = GuestMemory();
    Ipu& ipu = Unit();
    // IPU_CMD.
    m.OnWrite(0x10002000u, [](u32, u64 value) {
        Ipu& u = Unit();
        u.Command(u32(value));
        u.Pump();
    });
    m.OnRead(0x10002000u, [](u32, u64) {
        Ipu& u = Unit();
        u.Pump();
        return u64(u.cmd_data) | (u64(u.busy) << 63);
    });
    m.OnRead(0x10002004u, [](u32, u64) { return u64(Unit().busy) << 31; });
    // IPU_CTRL.
    m.OnWrite(0x10002010u, [](u32, u64 value) {
        Ipu& u = Unit();
        const u32 written = u32(value);
        if (written & (1u << 30)) u.Reset();
        const u32 writable = (3u << 16) | (1u << 20) | (1u << 21) | (1u << 22) | (1u << 23) | (7u << 24);
        u.ctrl = (u.ctrl & ~writable) | (written & writable);
        u.Pump();
    });
    m.OnRead(0x10002010u, [](u32, u64) {
        Ipu& u = Unit();
        u.Pump();
        return u64(u.ctrl | u32(std::min<std::size_t>(u.fifo.size(), 8)) |
                   u32(std::min<std::size_t>(u.out.size(), 8)) << 4 | (u.busy ? 1u << 31 : 0));
    });
    // IPU_BP.
    m.OnRead(0x10002020u, [](u32, u64) {
        Ipu& u = Unit();
        u.Pump();
        return u64(u.bp | u32(std::min<std::size_t>(u.fifo.size(), 8)) << 8 | u32(u.buffer.size()) << 16);
    });
    // IPU_TOP.
    m.OnRead(0x10002030u, [](u32, u64) {
        Ipu& u = Unit();
        u.Pump();
        return u.Top() | (u64(u.busy) << 63);
    });
    m.OnRead(0x10002034u, [](u32, u64) { return u64(Unit().busy) << 31; });
    // The input FIFO, written a quadword at a time (the last word hands it over).
    m.OnWrite(0x1000701Cu, [](u32, u64) {
        Ipu& u = Unit();
        Qword q;
        for (u32 i = 0; i < 4; ++i) q[i] = u32(GuestMemory().Read(0x10007010u + 4 * i, 4));
        u.fifo.push_back(q);
        u.Pump();
    });
    // The output FIFO, read a quadword at a time (the first word takes it).
    m.OnRead(0x10007000u, [](u32, u64) {
        Ipu& u = Unit();
        u.Pump();
        Qword q = {0, 0, 0, 0};
        if (!u.out.empty()) {
            q = u.out.front();
            u.out.pop_front();
        }
        for (u32 i = 1; i < 4; ++i) SetReg(0x10007000u + 4 * i, q[i]);
        u.Pump();
        return u64(q[0]) | u64(q[1]) << 32;
    });
    // The channels.
    m.OnWrite(kToIpu + kChcr, [](u32, u64 value) {
        Ipu& u = Unit();
        if (value & 0x100) u.StartTo(u32(value));
        else u.to_running = false;
        u.Pump();
    });
    m.OnWrite(kFromIpu + kChcr, [](u32, u64 value) {
        Ipu& u = Unit();
        u.from_running = (value & 0x100) != 0;
        u.Pump();
    });
    (void)ipu;
}

}  // namespace ps2host
