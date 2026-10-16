/*
 * The SPU2, the PS2's sound processor, as libsd drives it (iop_libs.cpp).
 *
 * Voices play through SPU RAM in real time, a sample at a time at 48 kHz:
 * their ADPCM is decoded, shaped by the ADSR envelope and mixed by their
 * volumes into the stereo output (an Output sink - the host's audio device -
 * takes it). Games synchronise with the same machinery: loop flags, ENDX, and
 * the SPU2 interrupt raised when a voice's playback (or a transfer) reaches the
 * interrupt address IRQA - the clock sound drivers stream by.
 *
 * Voice data is the SPU's ADPCM: 16-byte blocks of 28 samples. Byte 0 holds
 * the shift (bits 0-3) and the prediction filter (bits 4-6); byte 1 the flags:
 * bit 0 loop end, bit 1 repeat (go back to the loop start rather than stop),
 * bit 2 loop start. A voice plays at pitch/4096 of 48 kHz.
 *
 * Reverb (the effect unit): each core's voices with VMIXEL/VMIXER set feed it;
 * it runs at 24 kHz on its work area at the top of SPU RAM, with the
 * registers libsd's presets set (SetEffectAttr: sceSdSetEffectAttr), and its
 * output joins the core's at EVOL. The algorithm is the PS1 SPU's, which the
 * SPU2 kept (no-cash psx-spx, "SPU Reverb Formula").
 *
 * Not modelled yet: volume sweeps (a sweeping volume plays at its maximum),
 * noise, pitch modulation, and the cores' external and sound-data inputs
 * (core 0's output into core 1's reverb included).
 *
 * libsd's entries (ps2sdk libsd-common.h): a voice parameter or address is
 * (param << 8) | core | voice << 1; a switch or core address (n << 8) | core.
 */

#ifndef PS2HOST_SPU2_H
#define PS2HOST_SPU2_H

#include <cstdint>
#include <functional>
#include <vector>

namespace ps2host {
namespace iop {

class Spu2 {
public:
    using u8 = std::uint8_t;
    using u16 = std::uint16_t;
    using s16 = std::int16_t;
    using u32 = std::uint32_t;
    using s32 = std::int32_t;
    using u64 = std::uint64_t;

    Spu2() : ram_(2u << 20) {}

    void SetParam(u32 entry, u32 value);
    u32 GetParam(u32 entry) const;
    void SetSwitch(u32 entry, u32 value);
    u32 GetSwitch(u32 entry) const;
    void SetAddr(u32 entry, u32 value);
    u32 GetAddr(u32 entry) const;
    void SetCoreAttr(u32 entry, u32 value);
    u32 GetCoreAttr(u32 entry) const;
    /* sceSdSetEffectAttr: a reverb preset (SD_EFFECT_MODE_*, 0x100 to clear
       the work area), its depth, and for echo and delay their delay and
       feedback - as libsd programs the registers. -1 for a bad mode. */
    int SetEffectAttr(u32 core, u32 mode, s32 depth_left, s32 depth_right, u32 delay, u32 feedback);
    /* What the last SetEffectAttr was given (libsd keeps it). */
    void GetEffectAttr(u32 core, u32* mode, s32* depth_left, s32* depth_right, u32* delay, u32* feedback) const;

    /* Data into (or out of) SPU RAM at a byte address, as a transfer would. */
    void Write(u32 address, const u8* data, u32 size);
    void Read(u32 address, u8* data, u32 size) const;

    /* Advance playback to `now_us`; `irq(core)` for each interrupt raised. */
    void Advance(u64 now_us, const std::function<void(u32 core)>& irq);

    /* Where the mixed output goes: interleaved 16-bit stereo at 48 kHz. */
    using Output = std::function<void(const s16* frames, u32 count)>;
    void SetOutput(Output output) { output_ = std::move(output); }

    /* Exposed for tests: one ADPCM block decoded, with the filter history. */
    static void DecodeBlock(const u8* block, s32 out[28], s32* older, s32* old);

private:
    enum class Phase : u8 { Off, Attack, Decay, Sustain, Release };
    /* A volume as it is now (a voice's left or right, a core's main): set to
       a register's fixed value (bit 15 clear: 15 bits, signed), or moved a
       sample at a time by its sweep (bit 15 set: bit 14 exponential, 13
       decreasing, 12 the phase inverted, 0-6 the rate, as an envelope's)
       from where it was - the game fades a voice it leaves running so. */
    struct Level {
        u32 reg = 0;
        s32 now = 0;                    // -0x7FFF to 0x7FFF
        u32 counter = 0;
        explicit Level(u32 value = 0) { Set(value); }
        void Set(u32 value);
        void Step();
    };
    struct Voice {
        u32 pitch = 0, ssa = 0, lsax = 0, nax = 0;
        u32 adsr1 = 0, adsr2 = 0, voll = 0, volr = 0;
        Level left, right;              // VOLL, VOLR as they are now
        bool playing = false;           // sounding (a stopped voice still runs on)
        u32 counter = 0;                // 4.12 fixed point: samples into the block
        s32 samples[28] = {};           // the current block, decoded
        s32 older = 0, old = 0;         // the decoder's history
        s32 previous = 0;               // the sample before the current one
        Phase phase = Phase::Off;
        s32 level = 0;                  // the envelope, 0-0x7FFF
        u32 envelope_counter = 0;
        // The phase's rate as Envelope works it out from ADSR1/2, kept while
        // the phase and the registers stay (a voice's every sample asked).
        u64 rate_key = ~0ull;
        s32 rate_step = 0;
        u32 rate_increment = 0, rate_shift = 0;
        bool rate_decreasing = false, rate_exponential = false;
    };
    struct Core {
        Voice voices[24];
        u32 endx = 0, irqa = 0, attr = 0, esa = 0, eea = 0, tsa = 0;
        u32 switches[16] = {};          // by switch number - 0x13
        u32 mvoll = 0x3FFF, mvolr = 0x3FFF;
        Level main_left{0x3FFF}, main_right{0x3FFF};     // MVOLL, MVOLR as they are now
        bool irq_enabled = false;
        // Reverb: the registers (libsd's order: dAPF1, dAPF2, vIIR, vCOMB1-4,
        // vWALL, vAPF1, vAPF2, then the 20 buffer offsets in 8-byte units,
        // vLIN, vRIN), EVOL, where it is in its work area, what feeds it.
        u16 reverb[32] = {};
        u32 evoll = 0, evolr = 0;
        u32 effect_mode = 0, effect_delay = 0, effect_feedback = 0;
        s32 effect_depth_left = 0, effect_depth_right = 0;
        u32 reverb_position = 0;        // bytes into the work area
        s32 reverb_in_left = 0, reverb_in_right = 0;     // the first of each pair of samples
        s32 reverb_out_left = 0, reverb_out_right = 0;   // held for the pair
    };
    Core cores_[2];
    std::vector<u8> ram_;
    u64 last_us_ = 0;
    double clock_ = 0;               // samples since the SPU2 started
    double owed_ = 0;                // a fraction of a sample not yet produced
    Output output_;
    std::vector<s16> mixed_;

    static u32 CoreOf(u32 entry) { return entry & 1; }
    static u32 VoiceOf(u32 entry) { return ((entry >> 1) & 0x1F) % 24; }
    void CheckIrq(u32 core_hint, u32 from, u32 to, const std::function<void(u32)>& irq);
    void Start(Voice& v);
    void Decode(Voice& v);
    void NextBlock(Core& c, u32 index, const std::function<void(u32)>& irq);
    void Envelope(Voice& v);
    void Reverb(Core& c, s32 in_left, s32 in_right);
    u32 EffectEnd(const Core& c) const;
    std::vector<u32> pending_irqs_;
};

}  // namespace iop
}  // namespace ps2host

#endif  /* PS2HOST_SPU2_H */
