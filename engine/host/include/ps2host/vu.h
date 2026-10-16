/*
 * The vector units in micro mode: a VU running a microprogram out of its own
 * instruction memory over its own data memory, as VIF1 starts VU1 (MSCAL) and
 * the EE starts VU0 (vcallms).
 *
 * A microinstruction is a pair: an upper instruction for the FMAC units and a
 * lower one for everything else, issued together. The upper encodes exactly as
 * VU0 macro mode's COP2 operations do, so it runs through the runtime's
 * lifted::Cop2 - the arithmetic verified against vu0.py - as do the lower
 * operations macro mode shares (MOVE, MR32, DIV, SQRT, RSQRT, MTIR, MFIR, the
 * R register, and the integer ALU). Loads, stores, branches, the flags, the
 * elementary functions and XGKICK are here.
 *
 * Within a pair the lower instruction reads its registers before the upper
 * writes them, and where both write the same field the upper's result stands.
 * An E bit ends the program after the pair that follows it.
 *
 * Timing is modelled where code can see it, in cycles: a pair reading a vector
 * field still being computed stalls until it lands (4 cycles after the FMAC or
 * load issued); Q arrives 7 cycles after DIV and SQRT, 13 after RSQRT, and P
 * after the EFU operation's latency, the old value readable meanwhile; and the
 * MAC, status and clip flags reach the lower instructions (FMAND, FCAND...) 4
 * cycles after the upper operation that set them. Hand-pipelined microcode -
 * Burnout 3's clip tests, which read the flags of the vertex before - depends
 * on all of these (engine/tools/vuhazards.py lists where).
 */

#ifndef PS2HOST_VU_H
#define PS2HOST_VU_H

#include <cstdint>
#include <functional>
#include <map>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include "ps2rt/lifted.h"

namespace lifted { struct FmacOp; }

namespace ps2host {

struct VuRunState;

class Vu {
public:
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;

    /* VU0 (4 KB of each memory) or VU1 (16 KB). */
    explicit Vu(int index);
    Vu(const Vu&) = delete;             /* its memory spans point into itself */
    Vu& operator=(const Vu&) = delete;

    /* Instruction and data memory: the VU's own, or (Attach) the EE's view of
       it at 0x11000000, which code also reads and writes directly. */
    struct Span {
        std::uint8_t* p;
        std::size_t n;
        std::uint8_t* data() const { return p; }
        std::size_t size() const { return n; }
    };
    Span micro, data;
    void Attach(std::uint8_t* micro_memory, std::uint8_t* data_memory);
    lifted::Context regs{};         /* vf, vc (VI0-15, R 20, I 21, Q 22), vacc */
    /* The flags as the lower instructions read them, and P. */
    u32 mac = 0, status = 0, clip = 0, p = 0;
    u32 top = 0, itop = 0;          /* what XTOP and XITOP read (VU1) */
    u32 resume = 0;                 /* where the last run ended: MSCNT continues there */

    /* XGKICK: each quadword of the GIF packet, as PATH1 delivers it. */
    std::function<void(u64 low, u64 high)> on_kick;
    /* ...or, when set, a packet's quadwords a contiguous run of words at a
       time (as data memory holds them; a run ends where memory wraps). */
    std::function<void(const u32* words, u32 count)> on_kick_block;
    /* Instruction memory was written (MPG, or a store into it): compiled
       programs are compared with it again before they next run. */
    void MicroWritten() { ++micro_generation_; }

    /* Runs from instruction address `pc` (bytes) to the end of the program;
       returns the instruction pairs executed. Stops after `limit` regardless. */
    u64 Run(u32 pc, u64 limit = u64(1) << 24);

    u64 pairs() const { return pairs_; }
    /* Of those, the pairs run compiled ahead of time (vu_core.h). */
    u64 compiled_pairs() const { return compiled_pairs_; }
    /* Compiled programs entered, found with other code in memory than they
       were compiled from, and the bytes compared to tell. */
    u64 compiled_entries() const { return compiled_entries_; }
    u64 compiled_stale() const { return compiled_stale_; }
    u64 compiled_checked() const { return compiled_checked_; }
    u64 kicks() const { return kicks_; }
    /* Whether Run takes compiled programs where it has them (the tests run
       the interpreter alone to compare). */
    bool use_compiled = true;
    /* --vu-fast: compiled VU1 programs that never read MAC or STATUS run
       blocks that do not compute them (VuCompiled::fast_blocks): faster, and
       exact but for those flags - which a later run reading them before an
       FMAC of its own lands would see. */
    bool fast = false;
    /* For tests: everything a run can change, the same in both (`flags`
       false: but the MAC and STATUS flags, landed, latest and in flight). */
    bool SameState(const Vu& other, bool flags = true) const;
    /* --vu-record: each program a run starts, and the code reachable from
       where it starts, to `path` (engine/tools/vurecomp.py compiles them). */
    static void RecordPrograms(const std::string& path);
    /* --vu-snapshot: from frame `from` on, `count` VU1 runs as they start -
       where, the registers and flags, both memories - to `path`, for
       --vu-bench. SnapshotFrame: the frame now (devices.cpp). */
    static void Snapshots(const std::string& path, u64 from, u32 count);
    static void SnapshotFrame(u64 frame);
    /* --vu-bench: those runs again, on a VU1 of its own, compiled where the
       port has them, in `passes` passes; the fastest pass's time a pair, and a
       hash of every run's registers and memory after (what a change to the
       VU must leave alone). Returns 0, or 1 when the file cannot be read. */
    static int Bench(const std::string& path, int passes = 5, bool fast = false);
    /* How often each program address was started, for reports. */
    const std::map<u32, u64>& starts() const { return starts_; }

    /* For tests: one 128-bit quadword of data memory. */
    u32* Quad(u32 address);

private:
    friend struct VuCore;
    std::vector<std::uint8_t> own_micro_, own_data_;
    u32 data_mask_;                 /* in quadwords */
    u64 pairs_ = 0, kicks_ = 0, compiled_pairs_ = 0, compiled_entries_ = 0, compiled_stale_ = 0,
        compiled_checked_ = 0;
    std::map<u32, u64> starts_;
    // Instruction memory's writes so far, and for each compiled program the
    // count when it was last found in memory: unchanged since, it still is
    // (2.3 million entries a race compared 4.4 GB of code otherwise).
    u64 micro_generation_ = 1;
    std::unordered_map<const void*, u64> held_at_;

    // Compiled code: the vector registers that may hold fields the model's
    // Clean would change (bit r for VF r) - loaded so, or not known; the FMAC
    // cleans only those operands. Every bit is set as a run starts and when
    // the interpreter runs (it does not keep them); VF0 is always clean.
    u32 unclean_ = ~1u;
    bool acc_unclean_ = true;       // the same for ACC
    // An upper operation's result whose destination is VF0: the register
    // keeps 0, 0, 0, 1, but the MAC and status flags are this result's (a
    // test that only wants the signs - `msubz.xyzw vf00, ...` then FMAND -
    // writes nowhere).
    u32 discarded_[4] = {};

    /* Timing, in cycles (one per pair issued, plus stalls). */
    u64 cycle_ = 0;
    u64 vf_ready_[32][4] = {};      /* when each vector field's pending write lands */
    u32 q_next_ = 0, p_next_ = 0;   /* DIV's and the EFU's results in flight */
    u64 q_ready_ = 0, p_ready_ = 0;
    bool q_busy_ = false, p_busy_ = false;
    /* No earlier than the first of the busy Q and P lands (~0: neither is
       busy): what each pair looks at before landing them, one compare for
       four. LandNow sets it again. */
    u64 qp_due_ = ~u64(0);
    /* The FMAC sets the flags as it finishes: what upper instructions have
       produced so far, and what is still on its way to the lower's view. */
    u32 mac_latest_ = 0, status_latest_ = 0, clip_latest_ = 0;
    struct FlagWrite { u64 ready; u32 mac, status, clip; bool sets_mac, sets_clip; };
    /* In the order they were set, which is the order they land in (each
       lands 4 cycles after it was set, and time only goes forward): a ring,
       flag_head_ the oldest. */
    FlagWrite flag_queue_[8] = {};
    int flag_head_ = 0, flag_count_ = 0;
    void PushFlags(const FlagWrite& write);   /* (vu_core.h) */
    void Land();                    /* results whose time has come (a look, inline) */
    void LandNow();                 /* ...landing them */
    void LandAll();
    /* The CLIP judgements in flight (when each lands, its value): what
       SameState compares without the MAC and STATUS flags. */
    std::vector<std::pair<u64, u32>> ClipsInFlight() const;

    struct Pending;
    /* The interpreter, from where `state` stands. */
    void Interpret(VuRunState& state, u64 limit);
    /* Compiled programs from where `state` stands, as far as they go
       (vu_core.h); whether any ran. */
    bool RunCompiled(VuRunState& state, u64 limit);
    void Lower(u32 word, u32 pc, Pending& out, u32* branch);
    void Upper(u32 word, int* written_reg, u32* written_lanes, bool* set_flags,
               const lifted::FmacOp* fmac = nullptr);
    void Kick(u32 address);
};

/* VU1, which VIF1 loads and starts. */
Vu& Vu1();
/* VU0, which VIF0 loads and the EE starts (vcallms) or VIF0 does. */
Vu& Vu0();

}  // namespace ps2host

#endif  /* PS2HOST_VU_H */
