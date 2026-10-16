/*
 * The SPU2 (spu2.h): voices decoded, enveloped and mixed a sample at a time,
 * loop flags, ENDX, and the IRQA interrupt.
 */

#include "spu2.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ps2host {
namespace iop {

namespace {

constexpr Spu2::u32 kRamMask = (2u << 20) - 1;
constexpr double kRate = 48000.0;
constexpr Spu2::u32 kSamplesPerBlock = 28, kBlockBytes = 16;
constexpr Spu2::u32 kBlockSpan = kSamplesPerBlock << 12;    // a block in counter units
constexpr Spu2::u32 kVmixl = 0x18 - 0x13, kVmixr = 0x1A - 0x13;
constexpr Spu2::u32 kVmixel = 0x19 - 0x13, kVmixer = 0x1B - 0x13;   // into the reverb

Spu2::s32 Clamp16(Spu2::s32 v) { return std::min(32767, std::max(-32768, v)); }

// A volume register's fixed value (bit 15 clear: 15 bits, signed). A sweep's
// (bit 15 set) is where it has brought the volume: Spu2::Level. Reverb's EVOL
// has no sweep.
Spu2::s32 Volume(Spu2::u32 v) {
    if (v & 0x8000) return 0x7FFF;
    return Spu2::s32(Spu2::s16(Spu2::u16(v << 1)));
}

struct Rate {
    Spu2::u32 shift, step;
    bool decreasing, exponential;
};

}  // namespace

void Spu2::Level::Set(u32 value) {
    reg = value & 0xFFFF;
    if (!(reg & 0x8000)) now = Volume(reg);         // fixed; a sweep goes on from where it is
    counter = 0;
}

/* A sample of a sweep, by the envelope's rules (Envelope): the rate's shift
   and step, slower near the top going up exponentially, in proportion going
   down; the phase bit makes the volume negative. Rate 0x7F holds it. */
void Spu2::Level::Step() {
    if (!(reg & 0x8000)) return;
    const u32 rate = reg & 0x7F;
    if (rate == 0x7F) return;
    const bool exponential = (reg & 0x4000) != 0, decreasing = (reg & 0x2000) != 0, negative = (reg & 0x1000) != 0;
    s32 magnitude = now < 0 ? -now : now;
    const u32 shift = rate >> 2;
    s32 step = decreasing ? -8 + s32(rate & 3) : 7 - s32(rate & 3);
    if (shift < 11) step <<= 11 - shift;
    u32 increment = shift > 11 ? (shift - 11 >= 16 ? 0 : 0x8000u >> (shift - 11)) : 0x8000u;
    if (exponential) {
        if (decreasing) {
            step = s32((std::int64_t(step) * magnitude) >> 15);
        } else if (magnitude >= 0x6000) {
            if (shift < 10) step >>= 2;
            else if (shift >= 11) increment >>= 2;
            else { step >>= 1; increment >>= 1; }
        }
    }
    counter += increment;
    if (counter < 0x8000) return;
    counter = 0;
    magnitude = std::min(0x7FFF, std::max(0, magnitude + step));
    now = negative ? -magnitude : magnitude;
}

void Spu2::DecodeBlock(const u8* block, s32 out[28], s32* older, s32* old) {
    static const s32 kF0[5] = {0, 60, 115, 98, 122}, kF1[5] = {0, 0, -52, -55, -60};
    u32 shift = block[0] & 15;
    if (shift > 12) shift = 9;                      // as the hardware treats 13-15
    const u32 filter = std::min<u32>((block[0] >> 4) & 7, 4);
    for (u32 i = 0; i < kSamplesPerBlock; ++i) {
        const u32 nibble = (block[2 + i / 2] >> ((i & 1) * 4)) & 15;
        s32 s = s32(s16(u16(nibble << 12))) >> shift;
        s += (*old * kF0[filter] + *older * kF1[filter] + 32) >> 6;
        s = Clamp16(s);
        *older = *old;
        *old = s;
        out[i] = s;
    }
}

void Spu2::SetParam(u32 entry, u32 value) {
    Core& c = cores_[CoreOf(entry)];
    Voice& v = c.voices[VoiceOf(entry)];
    switch (entry >> 8) {
        case 0x00: v.voll = value; v.left.Set(value); break;
        case 0x01: v.volr = value; v.right.Set(value); break;
        case 0x02: v.pitch = value & 0x3FFF; break;
        case 0x03: v.adsr1 = value; break;
        case 0x04: v.adsr2 = value; break;
        case 0x09: c.mvoll = value; c.main_left.Set(value); break;     // SD_P_MVOLL
        case 0x0A: c.mvolr = value; c.main_right.Set(value); break;    // SD_P_MVOLR
        case 0x0B: c.evoll = value; break;             // SD_P_EVOLL: the reverb's output
        case 0x0C: c.evolr = value; break;             // SD_P_EVOLR
        default: break;                         // external input: not yet
    }
}

Spu2::u32 Spu2::GetParam(u32 entry) const {
    const Core& c = cores_[CoreOf(entry)];
    const Voice& v = c.voices[VoiceOf(entry)];
    switch (entry >> 8) {
        case 0x00: return v.voll;
        case 0x01: return v.volr;
        case 0x02: return v.pitch;
        case 0x03: return v.adsr1;
        case 0x04: return v.adsr2;
        case 0x05: return u32(v.level);             // ENVX
        case 0x06: return u32(v.left.now) & 0xFFFF;    // VOLXL, VOLXR: the current volume
        case 0x07: return u32(v.right.now) & 0xFFFF;
        case 0x09: return c.mvoll;
        case 0x0A: return c.mvolr;
        case 0x0B: return c.evoll;
        case 0x0C: return c.evolr;
        default: return 0;
    }
}

void Spu2::Start(Voice& v) {
    v.nax = v.ssa & ~15u;
    v.counter = 0;
    v.older = v.old = v.previous = 0;
    v.playing = true;
    v.phase = Phase::Attack;
    v.level = 0;
    v.envelope_counter = 0;
    Decode(v);
}

void Spu2::Decode(Voice& v) {
    const u32 block = v.nax & kRamMask & ~15u;
    u8 bytes[kBlockBytes];
    for (u32 i = 0; i < kBlockBytes; ++i) bytes[i] = ram_[(block + i) & kRamMask];
    if (bytes[1] & 4) v.lsax = block;               // loop start: where a repeat returns
    DecodeBlock(bytes, v.samples, &v.older, &v.old);
}

void Spu2::SetSwitch(u32 entry, u32 value) {
    Core& c = cores_[CoreOf(entry)];
    const u32 which = (entry >> 8) - 0x13;
    if (which < 16) c.switches[which] = value & 0xFFFFFF;
    switch (entry >> 8) {
        case 0x15:                              // KON: start from SSA
            for (u32 i = 0; i < 24; ++i)
                if (value & (1u << i)) {
                    Start(c.voices[i]);
                    c.endx &= ~(1u << i);
                }
            break;
        case 0x16:                              // KOFF: release
            for (u32 i = 0; i < 24; ++i)
                if ((value & (1u << i)) && c.voices[i].phase != Phase::Off)
                    c.voices[i].phase = Phase::Release;
            break;
        case 0x17: c.endx = value & 0xFFFFFF; break;
        default: break;
    }
}

Spu2::u32 Spu2::GetSwitch(u32 entry) const {
    const Core& c = cores_[CoreOf(entry)];
    if ((entry >> 8) == 0x17) return c.endx;
    const u32 which = (entry >> 8) - 0x13;
    return which < 16 ? c.switches[which] : 0;
}

void Spu2::SetAddr(u32 entry, u32 value) {
    Core& c = cores_[CoreOf(entry)];
    Voice& v = c.voices[VoiceOf(entry)];
    value &= kRamMask;
    switch (entry >> 8) {
        case 0x1C: c.esa = value; break;
        case 0x1D: c.eea = value; break;
        case 0x1E: c.tsa = value; break;
        case 0x1F: c.irqa = value; break;
        case 0x20: v.ssa = value; break;
        case 0x21: v.lsax = value; break;
        case 0x22: v.nax = value; break;
        default: break;
    }
}

Spu2::u32 Spu2::GetAddr(u32 entry) const {
    const Core& c = cores_[CoreOf(entry)];
    const Voice& v = c.voices[VoiceOf(entry)];
    switch (entry >> 8) {
        case 0x1C: return c.esa;
        case 0x1D: return c.eea;
        case 0x1E: return c.tsa;
        case 0x1F: return c.irqa;
        case 0x20: return v.ssa;
        case 0x21: return v.lsax;
        case 0x22: return v.nax;
        default: return 0;
    }
}

void Spu2::SetCoreAttr(u32 entry, u32 value) {
    Core& c = cores_[CoreOf(entry)];
    switch (entry & ~1u) {
        case 0x4: c.irq_enabled = value != 0; break;    // SD_CORE_IRQ_ENABLE
        default: break;
    }
    c.attr = (c.attr & ~(1u << (entry & ~1u))) | ((value ? 1u : 0u) << (entry & ~1u));
}

Spu2::u32 Spu2::GetCoreAttr(u32 entry) const {
    const Core& c = cores_[CoreOf(entry)];
    if ((entry & ~1u) == 0x4) return c.irq_enabled ? 1 : 0;
    return (c.attr >> (entry & ~1u)) & 1;
}

// ---------------------------------------------------------------- reverb

namespace {

// libsd's reverb presets (SD_EFFECT_MODE_OFF ... PIPE), as it programs them:
// 32 halfwords each in its register order; and each one's work area, in
// 16-byte units, ending at EEA. (The same values as ps2sdk's libsd, which
// took them from Sony's; close to the PS1's.)
const Spu2::u32 kEffectSizes[10] = {0x1, 0x26C, 0x1F4, 0x484, 0x6FE, 0xADE, 0xF6C, 0x1804, 0x1804, 0x3C0};
const Spu2::u32 kEffectParams[10][16] = {
    {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00010001, 0x00000000, 0x00000000,
     0x00000000, 0x00010001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
    {0x005B007D, 0x54B86D80, 0x0000BED0, 0xBA800000, 0x53005800, 0x033304D6, 0x022703F0, 0x01EF0374,
     0x01B70336, 0x01B60335, 0x01B50334, 0x01B50334, 0x01B50334, 0x013601B4, 0x005C00B8, 0x80008000},
    {0x00250033, 0x4FA870F0, 0x4410BCE0, 0x9C00C0F0, 0x4EC05280, 0x031B03E4, 0x02AF03A4, 0x02660372,
     0x025D031C, 0x018E025C, 0x0135022F, 0x00B701D2, 0x00B5018F, 0x008000B4, 0x0026004C, 0x80008000},
    {0x007F00B1, 0x4FA870F0, 0x4510BCE0, 0xB4C0BEF0, 0x4EC05280, 0x076B0904, 0x065F0824, 0x061607A2,
     0x05ED076C, 0x042E05EC, 0x0305050F, 0x02B70462, 0x0265042F, 0x01B20264, 0x00800100, 0x80008000},
    {0x00A900E3, 0x4FA86F60, 0x4510BCE0, 0xA680BEF0, 0x52C05680, 0x0B580DFB, 0x0A3C0D09, 0x09730BD9,
     0x08DA0B59, 0x05E908D9, 0x04B007EC, 0x03D206EF, 0x031D05EA, 0x0238031C, 0x00AA0154, 0x80008000},
    {0x013901A5, 0x50006000, 0xB8004C00, 0xC000BC00, 0x5C006000, 0x11BB15BA, 0x10BD14C2, 0x0DC111BC,
     0x0DC311C0, 0x09C10DC0, 0x07C10BC4, 0x06CD0A00, 0x05C109C2, 0x041A05C0, 0x013A0274, 0x80008000},
    {0x0231033D, 0x50007E00, 0xB000B400, 0xB0004C00, 0x54006000, 0x1A311ED6, 0x183B1D14, 0x16B21BC2,
     0x15EF1A32, 0x105515EE, 0x0F2D1334, 0x0C5D11F6, 0x0AE11056, 0x07A20AE0, 0x02320464, 0x80008000},
    {0x00030003, 0x7FFF7FFF, 0x00000000, 0x81000000, 0x00000000, 0x0FFD1FFD, 0x00091009, 0x00000000,
     0x00091009, 0x1FFF1FFF, 0x1FFE1FFE, 0x1FFE1FFE, 0x1FFE1FFE, 0x10041008, 0x00040008, 0x80008000},
    {0x00030003, 0x7FFF7FFF, 0x00000000, 0x00000000, 0x00000000, 0x0FFD1FFD, 0x00091009, 0x00000000,
     0x00091009, 0x1FFF1FFF, 0x1FFE1FFE, 0x1FFE1FFE, 0x1FFE1FFE, 0x10041008, 0x00040008, 0x80008000},
    {0x00130017, 0x4FA870F0, 0x4510BCE0, 0x8500BEF0, 0x54C05F80, 0x02AF0371, 0x01DF02E5, 0x01D702B0,
     0x026A0358, 0x011E01D6, 0x00B1012D, 0x0059011F, 0x00E301A0, 0x00400058, 0x00140028, 0x80008000},
};

// Where the registers sit in Core::reverb.
enum : int { kDApf1 = 0, kDApf2 = 1, kVIir = 2, kVComb1 = 3, kVComb2 = 4, kVComb3 = 5, kVComb4 = 6, kVWall = 7,
             kVApf1 = 8, kVApf2 = 9, kMLSame = 10, kMRSame = 11, kMLComb1 = 12, kMRComb1 = 13, kMLComb2 = 14,
             kMRComb2 = 15, kDLSame = 16, kDRSame = 17, kMLDiff = 18, kMRDiff = 19, kMLComb3 = 20, kMRComb3 = 21,
             kMLComb4 = 22, kMRComb4 = 23, kDLDiff = 24, kDRDiff = 25, kMLApf1 = 26, kMRApf1 = 27, kMLApf2 = 28,
             kMRApf2 = 29, kVLIn = 30, kVRIn = 31 };

Spu2::s32 Mul(Spu2::s32 a, Spu2::u16 coefficient) {
    return Spu2::s32((std::int64_t(a) * Spu2::s16(coefficient)) >> 15);
}

}  // namespace

int Spu2::SetEffectAttr(u32 core, u32 mode, s32 depth_left, s32 depth_right, u32 delay, u32 feedback) {
    Core& c = cores_[core & 1];
    const bool clear = (mode & 0x100) != 0;
    mode &= ~0x100u;
    if (mode > 9) return -1;
    u16 data[32];
    std::memcpy(data, kEffectParams[mode], sizeof data);
    c.effect_mode = mode;
    c.effect_depth_left = depth_left;
    c.effect_depth_right = depth_right;
    c.effect_delay = mode == 7 || mode == 8 ? 0x80 : 0;          // as libsd keeps them
    c.effect_feedback = mode == 7 ? 0x80 : 0;
    if (mode == 7 || mode == 8) {                               // echo, delay: their own delay and feedback
        const u16 d = u16((delay + 1) << 5);
        data[7] = u16((((feedback << 7) + feedback) << 1));
        data[10] = u16((d << 1) - data[0]);
        data[11] = u16(d - data[1]);
        data[16] = u16(data[17] + d);
        data[12] = u16(data[13] + d);
        data[27] = u16(data[29] + d);
        data[26] = u16(data[28] + d);
    }
    std::memcpy(c.reverb, data, sizeof data);
    c.evoll = u32(depth_left) & 0xFFFF;
    c.evolr = u32(depth_right) & 0xFFFF;
    const u32 eea = c.eea ? c.eea : (core & 1 ? 0x1FFFFFu : 0x1DFFFFu);
    c.esa = (eea - ((kEffectSizes[mode] << 4) - 2)) & kRamMask;
    c.reverb_position = 0;
    if (clear)
        for (u32 a = c.esa; a <= eea && a < ram_.size(); ++a) ram_[a] = 0;
    return 0;
}

void Spu2::GetEffectAttr(u32 core, u32* mode, s32* depth_left, s32* depth_right, u32* delay, u32* feedback) const {
    const Core& c = cores_[core & 1];
    *mode = c.effect_mode;
    *depth_left = c.effect_depth_left;
    *depth_right = c.effect_depth_right;
    *delay = c.effect_delay;
    *feedback = c.effect_feedback;
}

// The work area's end (EEA), libsd's default where the game set none.
Spu2::u32 Spu2::EffectEnd(const Core& c) const {
    return c.eea ? c.eea : (&c == &cores_[1] ? 0x1FFFFFu : 0x1DFFFFu);
}

// One step of the reverb (24 kHz): psx-spx's "SPU Reverb Formula". Offsets
// are in 8-byte units; "-2" there is the sample before (two bytes).
void Spu2::Reverb(Core& c, s32 in_left, s32 in_right) {
    const u16* r = c.reverb;
    const u32 size = EffectEnd(c) + 1 - c.esa;
    if (!size || size > ram_.size()) return;
    // A sample of the work area `offset` bytes past where the reverb is now,
    // wrapping within [ESA, EEA].
    const u32 position = c.reverb_position, esa = c.esa;
    u8* ram = ram_.data();
    auto place = [&](u32 offset) {
        u32 a = position + offset;
        if (a >= size) a = a - size < size ? a - size : a % size;
        return (esa + a) & kRamMask & ~1u;
    };
    auto ReverbRead = [&](const Core&, u32 offset) {
        const u32 a = place(offset);
        return s32(s16(u16(ram[a] | (ram[a + 1] << 8))));
    };
    auto ReverbWrite = [&](const Core&, u32 offset, s32 value) {
        const u32 a = place(offset);
        const u16 v = u16(s16(Clamp16(value)));
        ram[a] = u8(v);
        ram[a + 1] = u8(v >> 8);
    };
    auto at = [&](int reg) { return (u32(r[reg]) * 8) % size; };
    auto before = [&](int reg) { return (at(reg) + size - 2) % size; };
    const s32 lin = Mul(in_left, r[kVLIn]), rin = Mul(in_right, r[kVRIn]);
    // Same side, then the other side's reflection.
    const s32 lsame = Mul(lin + Mul(ReverbRead(c, at(kDLSame)), r[kVWall]) - ReverbRead(c, before(kMLSame)),
                          r[kVIir]) + ReverbRead(c, before(kMLSame));
    const s32 rsame = Mul(rin + Mul(ReverbRead(c, at(kDRSame)), r[kVWall]) - ReverbRead(c, before(kMRSame)),
                          r[kVIir]) + ReverbRead(c, before(kMRSame));
    const s32 ldiff = Mul(lin + Mul(ReverbRead(c, at(kDRDiff)), r[kVWall]) - ReverbRead(c, before(kMLDiff)),
                          r[kVIir]) + ReverbRead(c, before(kMLDiff));
    const s32 rdiff = Mul(rin + Mul(ReverbRead(c, at(kDLDiff)), r[kVWall]) - ReverbRead(c, before(kMRDiff)),
                          r[kVIir]) + ReverbRead(c, before(kMRDiff));
    ReverbWrite(c, at(kMLSame), lsame);
    ReverbWrite(c, at(kMRSame), rsame);
    ReverbWrite(c, at(kMLDiff), ldiff);
    ReverbWrite(c, at(kMRDiff), rdiff);
    // Early echo: the combs.
    s32 lout = Mul(ReverbRead(c, at(kMLComb1)), r[kVComb1]) + Mul(ReverbRead(c, at(kMLComb2)), r[kVComb2]) +
               Mul(ReverbRead(c, at(kMLComb3)), r[kVComb3]) + Mul(ReverbRead(c, at(kMLComb4)), r[kVComb4]);
    s32 rout = Mul(ReverbRead(c, at(kMRComb1)), r[kVComb1]) + Mul(ReverbRead(c, at(kMRComb2)), r[kVComb2]) +
               Mul(ReverbRead(c, at(kMRComb3)), r[kVComb3]) + Mul(ReverbRead(c, at(kMRComb4)), r[kVComb4]);
    // Late reverb: two all-pass filters.
    const u32 apf1 = at(kDApf1), apf2 = at(kDApf2);
    auto delayed = [&](int reg, u32 by) { return (at(reg) + size - by % size) % size; };
    lout = Clamp16(lout - Mul(ReverbRead(c, delayed(kMLApf1, apf1)), r[kVApf1]));
    ReverbWrite(c, at(kMLApf1), lout);
    lout = Mul(lout, r[kVApf1]) + ReverbRead(c, delayed(kMLApf1, apf1));
    rout = Clamp16(rout - Mul(ReverbRead(c, delayed(kMRApf1, apf1)), r[kVApf1]));
    ReverbWrite(c, at(kMRApf1), rout);
    rout = Mul(rout, r[kVApf1]) + ReverbRead(c, delayed(kMRApf1, apf1));
    lout = Clamp16(lout - Mul(ReverbRead(c, delayed(kMLApf2, apf2)), r[kVApf2]));
    ReverbWrite(c, at(kMLApf2), lout);
    lout = Mul(lout, r[kVApf2]) + ReverbRead(c, delayed(kMLApf2, apf2));
    rout = Clamp16(rout - Mul(ReverbRead(c, delayed(kMRApf2, apf2)), r[kVApf2]));
    ReverbWrite(c, at(kMRApf2), rout);
    rout = Mul(rout, r[kVApf2]) + ReverbRead(c, delayed(kMRApf2, apf2));
    c.reverb_out_left = Clamp16(lout);
    c.reverb_out_right = Clamp16(rout);
    c.reverb_position = (c.reverb_position + 2) % size;
}

void Spu2::CheckIrq(u32 core_hint, u32 from, u32 to, const std::function<void(u32)>& irq) {
    // IRQA is shared memory: either core with interrupts on sees it.
    (void)core_hint;
    for (u32 core = 0; core < 2; ++core) {
        Core& c = cores_[core];
        if (!c.irq_enabled) continue;
        if (c.irqa >= from && c.irqa < to) {
            c.irq_enabled = false;              // fires once; the driver re-arms it
            irq(core);
        }
    }
}

void Spu2::Write(u32 address, const u8* data, u32 size) {
    for (u32 i = 0; i < size; ++i) ram_[(address + i) & kRamMask] = data[i];
    pending_irqs_.push_back(address & kRamMask);
    pending_irqs_.push_back((address + size) & kRamMask);
}

void Spu2::Read(u32 address, u8* data, u32 size) const {
    for (u32 i = 0; i < size; ++i) data[i] = ram_[(address + i) & kRamMask];
}

// The end of a voice's block: its flags decide where it goes next. A voice never
// stops fetching: key-off and an unrepeated end only silence it, and its
// address runs on through the loop - drivers time things by silent looping
// voices.
void Spu2::NextBlock(Core& c, u32 index, const std::function<void(u32)>& irq) {
    Voice& v = c.voices[index];
    const u32 block = v.nax & kRamMask & ~15u;
    const u8 flags = ram_[(block + 1) & kRamMask];
    CheckIrq(0, block, block + kBlockBytes, irq);
    if (flags & 1) {
        c.endx |= 1u << index;
        if (!(flags & 2)) {                         // an end without repeat: silence
            v.playing = false;
            v.phase = Phase::Off;
            v.level = 0;
        }
        v.nax = v.lsax;
    } else {
        v.nax = (block + kBlockBytes) & kRamMask;
    }
    Decode(v);
}

// One step of the ADSR envelope (the PS1/PS2 SPU's: a rate is a shift and a
// step; exponential attack slows above 0x6000, exponential decrease is
// proportional to the level).
void Spu2::Envelope(Voice& v) {
    if (v.phase == Phase::Off) return;
    if (v.phase == Phase::Decay && v.level <= std::min(0x7FFF, s32(((v.adsr1 & 15) + 1) * 0x800))) {
        v.phase = Phase::Sustain;                   // already at the sustain level
        return;
    }
    // The phase's rate: worked out again only when the phase or ADSR1/2 change.
    const u64 key = u64(v.phase) | u64(v.adsr1 & 0xFFFF) << 8 | u64(v.adsr2 & 0xFFFF) << 24;
    if (key != v.rate_key) {
        Rate r{};
        switch (v.phase) {
            case Phase::Attack: {
                const u32 rate = (v.adsr1 >> 8) & 0x7F;
                r = {rate >> 2, rate & 3, false, (v.adsr1 & 0x8000) != 0};
                break;
            }
            case Phase::Decay:
                r = {(v.adsr1 >> 4) & 15, 0, true, true};
                break;
            case Phase::Sustain: {
                const u32 rate = (v.adsr2 >> 6) & 0x7F;
                r = {rate >> 2, rate & 3, (v.adsr2 & 0x4000) != 0, (v.adsr2 & 0x8000) != 0};
                break;
            }
            case Phase::Release:
                r = {v.adsr2 & 0x1F, 0, true, (v.adsr2 & 0x20) != 0};
                break;
            default: break;
        }
        s32 step = r.decreasing ? -8 + s32(r.step) : 7 - s32(r.step);
        v.rate_step = r.shift < 11 ? step << (11 - r.shift) : step;
        v.rate_increment = r.shift > 11 ? (r.shift - 11 >= 16 ? 0 : 0x8000u >> (r.shift - 11)) : 0x8000u;
        v.rate_shift = r.shift;
        v.rate_decreasing = r.decreasing;
        v.rate_exponential = r.exponential;
        v.rate_key = key;
    }
    s32 step = v.rate_step;
    u32 increment = v.rate_increment;
    if (v.rate_exponential) {
        if (v.rate_decreasing) {
            step = s32((std::int64_t(step) * v.level) >> 15);
        } else if (v.level >= 0x6000) {
            if (v.rate_shift < 10) step >>= 2;
            else if (v.rate_shift >= 11) increment >>= 2;
            else { step >>= 1; increment >>= 1; }
        }
    }
    v.envelope_counter += increment;
    if (v.envelope_counter < 0x8000) return;
    v.envelope_counter = 0;
    v.level = std::min(0x7FFF, std::max(0, v.level + step));
    switch (v.phase) {
        case Phase::Attack:
            if (v.level >= 0x7FFF) v.phase = Phase::Decay;
            break;
        case Phase::Decay: {
            const s32 sustain = std::min(0x7FFF, s32(((v.adsr1 & 15) + 1) * 0x800));
            if (v.level <= sustain) v.phase = Phase::Sustain;
            break;
        }
        case Phase::Release:
            if (v.level == 0) { v.phase = Phase::Off; v.playing = false; }
            break;
        default: break;
    }
}

void Spu2::Advance(u64 now_us, const std::function<void(u32 core)>& irq) {
    // Transfers that touched IRQA.
    for (std::size_t i = 0; i + 1 < pending_irqs_.size(); i += 2)
        CheckIrq(0, pending_irqs_[i], pending_irqs_[i + 1] > pending_irqs_[i] ? pending_irqs_[i + 1] : pending_irqs_[i] + 1, irq);
    pending_irqs_.clear();
    if (!last_us_) last_us_ = now_us;
    const double samples = double(now_us - last_us_) * kRate / 1e6;
    last_us_ = now_us;
    if (samples <= 0) return;
    // The SPU2's memory streams: SPU RAM 0x0000-0x27FF is rings of 0x200 bytes
    // (256 16-bit samples) that the hardware writes (outputs) and reads (block
    // transfer input) one sample per 48 kHz tick, all at the same position. An
    // IRQA inside them fires each time that position passes it - every 256
    // samples, the clock drivers stream by.
    const u64 before = u64(clock_), after = u64(clock_ + samples);
    clock_ += samples;
    for (u32 core = 0; core < 2; ++core) {
        Core& c = cores_[core];
        if (!c.irq_enabled || c.irqa >= 0x2800) continue;
        const u64 offset = (c.irqa & 0x1FF) >> 1;
        // Is there a t in (before, after] with t % 256 == offset?
        const u64 next = before - before % 256 + offset + (before % 256 >= offset ? 256 : 0);
        if (next <= after) {
            c.irq_enabled = false;
            irq(core);
        }
    }
    // The voices, a sample at a time; at most a second's worth at once (a host
    // that stalled longer than that catches up the clock, not the sound).
    owed_ += samples;
    u32 count = u32(owed_);
    owed_ -= count;
    count = std::min<u32>(count, u32(kRate));
    // PS2HOST_SPU_LOG=1: once a second, the voices sounding and how their
    // volumes are set (fixed, or a sweep), and the cores' main volumes.
    static const bool log = std::getenv("PS2HOST_SPU_LOG") != nullptr;
    if (log && u64(clock_) / 48000 != u64(clock_ - samples) / 48000) {
        for (u32 core = 0; core < 2; ++core) {
            const Core& c = cores_[core];
            u32 sounding = 0, sweeping = 0;
            std::string sweeps;
            for (u32 i = 0; i < 24; ++i) {
                const Voice& v = c.voices[i];
                if (v.phase == Phase::Off || !v.pitch) continue;
                ++sounding;
                if ((v.voll | v.volr) & 0x8000) {
                    ++sweeping;
                    char one[48];
                    std::snprintf(one, sizeof one, " %u:%04X/%04X", unsigned(i), unsigned(v.voll), unsigned(v.volr));
                    sweeps += one;
                }
            }
            std::fprintf(stderr, "[spu2] core %u: %u voices sounding, %u sweeping%s; main %04X/%04X, effect %04X/%04X\n",
                         unsigned(core), unsigned(sounding), unsigned(sweeping), sweeps.c_str(), unsigned(c.mvoll),
                         unsigned(c.mvolr), unsigned(c.evoll), unsigned(c.evolr));
        }
    }
    mixed_.assign(std::size_t(count) * 2, 0);
    for (u32 n = 0; n < count; ++n) {
        s32 left = 0, right = 0;
        for (u32 core = 0; core < 2; ++core) {
            Core& c = cores_[core];
            s32 core_left = 0, core_right = 0, wet_left = 0, wet_right = 0;
            for (u32 i = 0; i < 24; ++i) {
                Voice& v = c.voices[i];
                v.left.Step();
                v.right.Step();
                if (!v.pitch) continue;
                if (v.phase != Phase::Off) {
                    const u32 index = v.counter >> 12;
                    const s32 current = v.samples[index];
                    const s32 sample = v.previous + ((current - v.previous) * s32(v.counter & 0xFFF) >> 12);
                    const s32 shaped = (sample * v.level) >> 15;
                    const s32 sent_left = (shaped * v.left.now) >> 15, sent_right = (shaped * v.right.now) >> 15;
                    if (c.switches[kVmixl] & (1u << i)) core_left += sent_left;
                    if (c.switches[kVmixr] & (1u << i)) core_right += sent_right;
                    if (c.switches[kVmixel] & (1u << i)) wet_left += sent_left;
                    if (c.switches[kVmixer] & (1u << i)) wet_right += sent_right;
                    Envelope(v);
                }
                const u32 before_index = v.counter >> 12;
                s32 last = v.samples[before_index];
                bool crossed = false;
                v.counter += v.pitch;
                while (v.counter >= kBlockSpan) {
                    v.counter -= kBlockSpan;
                    last = v.samples[kSamplesPerBlock - 1];
                    NextBlock(c, i, irq);
                    crossed = true;
                }
                const u32 index = v.counter >> 12;
                if (crossed || index != before_index)
                    v.previous = index ? v.samples[index - 1] : last;
            }
            // The reverb (SD_CORE_EFFECT_ENABLE, attr bit 2): a step every
            // other sample, on the pair's first input; its output held for
            // the pair.
            static const bool no_reverb = std::getenv("PS2HOST_SPU_NO_REVERB") != nullptr;   // to hear without it
            if ((c.attr & 4) && !no_reverb) {
                if ((u64(clock_) + n) & 1) Reverb(c, Clamp16(c.reverb_in_left), Clamp16(c.reverb_in_right));
                else { c.reverb_in_left = wet_left; c.reverb_in_right = wet_right; }
                core_left += (c.reverb_out_left * Volume(c.evoll)) >> 15;
                core_right += (c.reverb_out_right * Volume(c.evolr)) >> 15;
            }
            c.main_left.Step();
            c.main_right.Step();
            left += (core_left * c.main_left.now) >> 15;
            right += (core_right * c.main_right.now) >> 15;
        }
        mixed_[2 * n] = s16(Clamp16(left));
        mixed_[2 * n + 1] = s16(Clamp16(right));
    }
    if (output_ && count) output_(mixed_.data(), count);
}

}  // namespace iop
}  // namespace ps2host
