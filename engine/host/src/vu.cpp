/*
 * The vector units in micro mode (vu.h).
 *
 * Instruction fields (both halves): dest x y z w in bits 24-21, ft/it 20-16,
 * fs/is 15-11, fd/id 10-6. The upper's flags: I 31 (the lower word is a float
 * for the I register), E 30 (end), M 29, D 28, T 27. Instruction addresses are
 * in bytes (pairs of 8); data addresses in quadwords.
 */

#include "ps2host/vu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <xmmintrin.h>
#include <unordered_set>
#include <vector>

#include "ps2host/memory.h"
#include "ps2host/profile.h"
#include "ps2host/vu_core.h"
#include "ps2rt/generated.h"
#include "ps2rt/vu_math.h"


namespace ps2host {

namespace {

constexpr Vu::u32 kNone = ~0u;
constexpr int kRegR = 20, kRegI = 21, kRegQ = 22;
constexpr Vu::u32 kCop2 = 0x4A000000u;       // a macro-mode COP2 operation

Vu::u32 Field(Vu::u32 word, int low, int count) { return (word >> low) & ((1u << count) - 1); }
std::int32_t Imm11(Vu::u32 word) { return std::int32_t(word << 21) >> 21; }

// The vector fields a pair reads, for its stalls: (register, lanes as dest bits).
struct Reads {
    int n = 0;
    Vu::u32 reg[6], lanes[6];
    void Add(Vu::u32 r, Vu::u32 l) {
        if (r && l) { reg[n] = r; lanes[n] = l; ++n; }
    }
};

Vu::u32 LaneBit(Vu::u32 lane) { return 8u >> lane; }    // x 0 .. w 3, as dest bits

void UpperReads(Vu::u32 word, Reads& r) {
    const Vu::u32 funct = word & 0x3F, dest = Field(word, 21, 4);
    const Vu::u32 fs = Field(word, 11, 5), ft = Field(word, 16, 5);
    if (funct < 0x1C) { r.Add(fs, dest); r.Add(ft, LaneBit(funct & 3)); return; }   // the broadcasts
    if (funct < 0x28) { r.Add(fs, dest); return; }                                   // Q and I forms
    if (funct < 0x30) {
        const Vu::u32 lanes = funct == 0x2E ? 0xE : dest;                             // OPMSUB: xyz
        r.Add(fs, lanes);
        r.Add(ft, lanes);
        return;
    }
    if (funct < 0x3C) return;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    if (index == 0x2F) return;                                                       // NOP
    if (index == 0x1F) { r.Add(fs, 0xE); r.Add(ft, 0x1); return; }                   // CLIP
    if (index < 0x10 || (index >= 0x18 && index < 0x1C)) {                           // the broadcasts
        r.Add(fs, dest);
        r.Add(ft, LaneBit(index & 3));
        return;
    }
    if (index < 0x28) { r.Add(fs, dest); return; }      // ITOF FTOI ABS, Q and I forms
    const Vu::u32 lanes = index == 0x2E ? 0xE : dest;   // OPMULA: xyz
    r.Add(fs, lanes);
    r.Add(ft, lanes);
}

void LowerReads(Vu::u32 word, Reads& r) {
    const Vu::u32 op = word >> 25, dest = Field(word, 21, 4);
    const Vu::u32 is = Field(word, 11, 5), it = Field(word, 16, 5);
    const Vu::u32 fsf = LaneBit(Field(word, 21, 2)), ftf = LaneBit(Field(word, 23, 2));
    if (op == 0x01) { r.Add(is, dest); return; }        // SQ
    if (op != 0x40 || (word & 0x3F) < 0x3C) return;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    switch (index) {
        case 0x30: r.Add(is, dest); return;                                          // MOVE
        case 0x31: r.Add(is, ((dest << 3) | (dest >> 1)) & 0xF); return;             // MR32: x from y ..
        case 0x35: case 0x37: r.Add(is, dest); return;                               // SQI SQD
        case 0x38: case 0x3A: r.Add(is, fsf); r.Add(it, ftf); return;                // DIV RSQRT
        case 0x39: r.Add(it, ftf); return;                                           // SQRT
        case 0x3C: case 0x42: case 0x43: r.Add(is, fsf); return;                     // MTIR RINIT RXOR
        case 0x70: case 0x71: case 0x72: case 0x73: r.Add(is, 0xE); return;         // ESADD .. ERLENG
        case 0x74: r.Add(is, 0xC); return;                                           // EATANxy
        case 0x75: r.Add(is, 0xA); return;                                           // EATANxz
        case 0x76: r.Add(is, 0xF); return;                                           // ESUM
        case 0x78: case 0x79: case 0x7A: case 0x7C: case 0x7D: case 0x7E:
            r.Add(is, fsf);
            return;
        default: return;
    }
}

bool IsEfu(Vu::u32 word) {
    if ((word >> 25) != 0x40 || (word & 0x3F) < 0x3C) return false;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    return index >= 0x70 && index <= 0x7E && index != 0x77 && index != 0x7B;
}

bool IsDivide(Vu::u32 word) {
    if ((word >> 25) != 0x40 || (word & 0x3F) < 0x3C) return false;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    return index >= 0x38 && index <= 0x3A;
}

bool IsWait(Vu::u32 word, Vu::u32 which) {             // 0x3B WAITQ, 0x7B WAITP
    return (word >> 25) == 0x40 && (word & 0x3F) >= 0x3C && ((word & 3) | ((word >> 4) & 0x7C)) == which;
}

constexpr int kFmacLatency = 4;

// A pair, decoded for its stalls once: what it reads and whether it waits on
// the divider or the EFU. Kept per address and checked against the words
// there (an MPG may have replaced them).
struct DecodedPair {
    Vu::u32 lower = 0, upper = 0;
    bool valid = false;
    bool immediate = false, q_waits = false, p_waits = false;
    Reads reads;
    bool has_fmac = false;          // the upper is an FMAC operation, decoded
    lifted::FmacOp fmac;
};

DecodedPair& Decoded(std::vector<DecodedPair>& cache, Vu::u32 pc, Vu::u32 lower, Vu::u32 upper) {
    DecodedPair& d = cache[pc / 8];
    if (d.valid && d.lower == lower && d.upper == upper) return d;
    d = DecodedPair{};
    d.lower = lower;
    d.upper = upper;
    d.valid = true;
    d.immediate = (upper & 0x80000000u) != 0;          // I bit: the lower is I's value
    UpperReads(upper, d.reads);
    d.has_fmac = lifted::DecodeFmac(kCop2 | (upper & 0x01FFFFFF), &d.fmac);
    if (!d.immediate) {
        LowerReads(lower, d.reads);
        d.q_waits = IsDivide(lower) || IsWait(lower, 0x3B);
        d.p_waits = IsEfu(lower) || IsWait(lower, 0x7B);
    }
    return d;
}

// ---------------------------------------------------------------- compiled programs

std::unordered_map<std::uint64_t, std::vector<const VuCompiled*>>& Compiled() {
    static std::unordered_map<std::uint64_t, std::vector<const VuCompiled*>> programs;
    return programs;
}

// A program's blocks by pair address (null where none starts), made once.
const std::vector<VuBlock>& BlockTable(const VuCompiled* program, std::size_t pairs, bool fast) {
    // A thread's own (VU1 runs on VIF1's worker, VU0 on the EE's).
    thread_local std::unordered_map<const VuCompiled*, std::vector<VuBlock>> tables[2];
    const VuBlock* blocks = fast && program->fast_blocks ? program->fast_blocks : program->blocks;
    std::vector<VuBlock>& table = tables[blocks == program->blocks ? 0 : 1][program];
    if (table.empty()) {
        table.assign(pairs, nullptr);
        for (std::uint32_t i = 0; i < program->block_count; ++i)
            if (program->starts[i] / 8 < pairs) table[program->starts[i] / 8] = blocks[i];
    }
    return table;
}

// Whether instruction memory still holds the code a program was compiled from.
bool Holds(const Vu::Span& micro, const Vu::u32* code) {
    const Vu::u32 runs = *code++;
    for (Vu::u32 i = 0; i < runs; ++i) {
        const Vu::u32 base = code[0], words = code[1];
        if (base + 4 * words > micro.size() || std::memcmp(micro.data() + base, code + 2, 4 * words) != 0)
            return false;
        code += 2 + words;
    }
    return true;
}

// ---------------------------------------------------------------- recording (--vu-record)

std::FILE* g_record = nullptr;

std::unordered_set<std::size_t>& Recorded() {
    static std::unordered_set<std::size_t> seen;
    return seen;
}

bool IsBranch(Vu::u32 lower, bool* conditional, bool* known) {
    const Vu::u32 op = lower >> 25;
    *conditional = op == 0x28 || op == 0x29 || (op >= 0x2C && op <= 0x2F);
    *known = op == 0x20 || op == 0x21 || *conditional;
    return *known || op == 0x24 || op == 0x25;
}

// The pairs a run started at `entry` can reach: on from each, both ways at a
// branch (its delay slot run first), as far as the pair after an E bit; a
// jump to a register's address is followed no further (the run leaves what
// was compiled there, and the interpreter goes on).
std::vector<Vu::u32> Reachable(const Vu::Span& micro, Vu::u32 entry) {
    const Vu::u32 mask = Vu::u32(micro.size() - 1);
    std::vector<std::uint8_t> seen(micro.size() / 8, 0);     // 1: followed on, 2: included only
    std::vector<std::pair<Vu::u32, bool>> work{{entry, true}};
    auto words = [&](Vu::u32 pc, Vu::u32* lower, Vu::u32* upper) {
        std::memcpy(lower, micro.data() + pc, 4);
        std::memcpy(upper, micro.data() + pc + 4, 4);
    };
    while (!work.empty()) {
        const auto [pc, follow] = work.back();
        work.pop_back();
        std::uint8_t& mark = seen[pc / 8];
        if (mark & 1 || (!follow && mark)) continue;
        mark |= follow ? 1 : 2;
        if (!follow) continue;
        Vu::u32 lower, upper;
        words(pc, &lower, &upper);
        const Vu::u32 next = (pc + 8) & mask;
        bool conditional = false, known = false;
        if (upper & 0x40000000u) {
            work.push_back({next, false});
        } else if (!(upper & 0x80000000u) && IsBranch(lower, &conditional, &known)) {
            work.push_back({next, false});
            if (known) work.push_back({(pc + 8 + Vu::u32(Imm11(lower) * 8)) & mask, true});
            if (conditional) work.push_back({(pc + 16) & mask, true});
        } else {
            work.push_back({next, true});
        }
    }
    std::vector<Vu::u32> pcs;
    for (Vu::u32 i = 0; i < seen.size(); ++i)
        if (seen[i]) pcs.push_back(i * 8);
    return pcs;
}

void RecordProgram(const Vu::Span& micro, int unit, Vu::u32 entry) {
    const std::vector<Vu::u32> pcs = Reachable(micro, entry);
    std::string line = "vu" + std::to_string(unit);
    char text[32];
    std::snprintf(text, sizeof text, " %X ", unsigned(entry));
    line += text;
    for (std::size_t i = 0; i < pcs.size(); ++i) {
        const bool starts = i == 0 || pcs[i - 1] + 8 != pcs[i];
        if (starts) {
            std::snprintf(text, sizeof text, "%s%X:", i ? ";" : "", unsigned(pcs[i]));
            line += text;
        }
        Vu::u32 lower, upper;
        std::memcpy(&lower, micro.data() + pcs[i], 4);
        std::memcpy(&upper, micro.data() + pcs[i] + 4, 4);
        std::snprintf(text, sizeof text, "%s%X,%X", starts ? "" : ",", unsigned(lower), unsigned(upper));
        line += text;
    }
    if (!Recorded().insert(std::hash<std::string>{}(line)).second) return;
    std::fprintf(g_record, "%s\n", line.c_str());
    std::fflush(g_record);
}

}  // namespace

void RegisterVuProgram(const VuCompiled* program) {
    Compiled()[std::uint64_t(program->unit) << 32 | program->entry].push_back(program);
}

void Vu::RecordPrograms(const std::string& path) {
    g_record = std::fopen(path.c_str(), "a");
    if (!g_record) std::fprintf(stderr, "vu: cannot write %s\n", path.c_str());
}

bool Vu::SameState(const Vu& o, bool flags) const {
    auto same = [](const void* a, const void* b, std::size_t n) { return std::memcmp(a, b, n) == 0; };
    if (!same(regs.vf, o.regs.vf, sizeof regs.vf) || !same(regs.vc, o.regs.vc, sizeof regs.vc) ||
        !same(regs.vacc, o.regs.vacc, sizeof regs.vacc))
        return false;
    if ((flags && (mac != o.mac || status != o.status)) || clip != o.clip || p != o.p || resume != o.resume)
        return false;
    if (cycle_ != o.cycle_ || !same(vf_ready_, o.vf_ready_, sizeof vf_ready_) || q_next_ != o.q_next_ ||
        p_next_ != o.p_next_ || q_ready_ != o.q_ready_ || p_ready_ != o.p_ready_ || q_busy_ != o.q_busy_ ||
        p_busy_ != o.p_busy_)
        return false;
    if (!flags)     // CLIP's in flight still, without the MAC writes between
        return clip_latest_ == o.clip_latest_ && ClipsInFlight() == o.ClipsInFlight() && data.size() == o.data.size() &&
               same(data.data(), o.data.data(), data.size()) && kicks_ == o.kicks_;
    if (mac_latest_ != o.mac_latest_ || status_latest_ != o.status_latest_ || clip_latest_ != o.clip_latest_ ||
        flag_count_ != o.flag_count_)
        return false;
    for (int i = 0; i < flag_count_; ++i) {
        const FlagWrite &a = flag_queue_[(flag_head_ + i) & 7], &b = o.flag_queue_[(o.flag_head_ + i) & 7];
        if (a.ready != b.ready || a.mac != b.mac || a.status != b.status || a.clip != b.clip ||
            a.sets_mac != b.sets_mac || a.sets_clip != b.sets_clip)
            return false;
    }
    return data.size() == o.data.size() && same(data.data(), o.data.data(), data.size()) && kicks_ == o.kicks_;
}

Vu::Vu(int index)
    : own_micro_(index ? 16384 : 4096), own_data_(index ? 16384 : 4096), data_mask_(index ? 0x3FF : 0xFF) {
    micro = {own_micro_.data(), own_micro_.size()};
    data = {own_data_.data(), own_data_.size()};
    regs.vf[0][3] = 0x3F800000u;    // vf0 = (0, 0, 0, 1)
}

void Vu::Attach(std::uint8_t* micro_memory, std::uint8_t* data_memory) {
    if (micro_memory) micro.p = micro_memory;
    if (data_memory) data.p = data_memory;
}

Vu::u32* Vu::Quad(u32 address) {
    return reinterpret_cast<u32*>(data.data() + std::size_t(address & data_mask_) * 16);
}

void Vu::Kick(u32 address) {
    ++kicks_;
    for (int packets = 0; packets < 4096; ++packets) {
        const u32* tag = Quad(address);
        const u64 low = u64(tag[0]) | u64(tag[1]) << 32, high = u64(tag[2]) | u64(tag[3]) << 32;
        const u32 nloop = u32(low & 0x7FFF), eop = u32(low >> 15) & 1, flg = u32(low >> 58) & 3;
        const u32 nreg = u32(low >> 60) ? u32(low >> 60) : 16;
        const u32 qwords = flg == 0 ? nloop * nreg : flg == 1 ? (nloop * nreg + 1) / 2 : nloop;
        if (on_kick_block) {
            // The tag and its data, in runs up to where data memory wraps.
            for (u32 left = 1 + qwords; left;) {
                const u32 run = std::min(left, data_mask_ + 1 - address);
                on_kick_block(Quad(address), run * 4);
                address = (address + run) & data_mask_;
                left -= run;
            }
            if (eop) return;
            continue;
        }
        if (on_kick) on_kick(low, high);
        address = (address + 1) & data_mask_;
        for (u32 i = 0; i < qwords; ++i) {
            const u32* q = Quad(address);
            if (on_kick) on_kick(u64(q[0]) | u64(q[1]) << 32, u64(q[2]) | u64(q[3]) << 32);
            address = (address + 1) & data_mask_;
        }
        if (eop) return;
    }
}

// The upper instruction, run on the registers; says which vector register and
// fields it wrote (if any), for the lower's writes to give way to.
void Vu::Upper(u32 word, int* written_reg, u32* written_lanes, bool* set_flags, const lifted::FmacOp* fmac) {
    *written_reg = -1;
    *written_lanes = 0;
    *set_flags = false;
    const u32 funct = word & 0x3F, dest = Field(word, 21, 4);
    const u32 ft = Field(word, 16, 5), fd = Field(word, 6, 5);
    bool flags = false;
    int target = -1;
    if (funct < 0x3C) {
        if (funct >= 0x30) return;                          // not upper operations
        target = int(fd);
        // MAX and MINI leave the flags alone.
        flags = !((funct >= 0x10 && funct <= 0x17) || funct == 0x1D || funct == 0x1F ||
                  funct == 0x2B || funct == 0x2F);
    } else {
        const u32 index = (word & 3) | ((word >> 4) & 0x7C);
        if (index == 0x1F) {                                // CLIP: fs.xyz against |ft.w|
            const u32* v = regs.vf[Field(word, 11, 5)];
            const double w = std::fabs(lifted::Value(regs.vf[ft][3]));
            u32 judgement = 0;
            for (int lane = 0; lane < 3; ++lane) {
                const double x = lifted::Value(v[lane]);
                if (x > w) judgement |= 1u << (2 * lane);
                if (x < -w) judgement |= 2u << (2 * lane);
            }
            clip_latest_ = ((clip_latest_ << 6) | judgement) & 0xFFFFFF;
            *set_flags = true;
            return;
        }
        if (index == 0x2F) return;                          // NOP
        if ((index >= 0x10 && index <= 0x17) || index == 0x1D) target = int(ft);   // ITOF, FTOI, ABS
        else flags = true;                                  // the accumulator forms
    }
    if (fmac) lifted::RunFmac(regs, *fmac);
    else lifted::Cop2(regs, kCop2 | (word & 0x01FFFFFF));
    if (target > 0) {
        *written_reg = target;
        *written_lanes = dest;
    }
    if (flags && target == 0) {
        // Into VF0: the flags are the result's, which went nowhere.
        if (fmac) {
            lifted::FmacResult(regs, *fmac, discarded_);
        } else {
            lifted::FmacOp decoded;
            if (lifted::DecodeFmac(kCop2 | (word & 0x01FFFFFF), &decoded)) {
                lifted::FmacResult(regs, decoded, discarded_);
            } else if (funct == 0x2E) {                     // OPMSUB (it wrote nothing: its operands stand)
                const u32 *a = regs.vf[Field(word, 11, 5)], *b = regs.vf[ft], *acc = regs.vacc;
                const u32 out[3] = {lifted::Binary(lifted::kMsub, a[1], b[2], acc[0]),
                                    lifted::Binary(lifted::kMsub, a[2], b[0], acc[1]),
                                    lifted::Binary(lifted::kMsub, a[0], b[1], acc[2])};
                for (int lane = 0; lane < 3; ++lane) discarded_[lane] = (dest & (8u >> lane)) ? out[lane] : 0;
                discarded_[3] = 0;
            }
        }
    }
    if (flags) {
        // MAC: zero (bits 3-0) and sign (7-4) of each written field, x highest.
        const u32* out = target > 0 ? regs.vf[target] : target == 0 ? discarded_ : regs.vacc;
        u32 m = 0;
        for (int lane = 0; lane < 4; ++lane) {
            if (!(dest & (8u >> lane))) continue;
            const u32 bits = lifted::Clean(out[lane]);
            if ((bits & 0x7FFFFFFFu) == 0) m |= 8u >> lane;
            if (bits & 0x80000000u) m |= 0x80u >> lane;
        }
        mac_latest_ = m;
        status_latest_ = (status_latest_ & ~0x3u) | ((m & 0xF) ? 1u : 0u) | ((m & 0xF0) ? 2u : 0u);
        status_latest_ |= (status_latest_ & 3u) << 6;       // the sticky copies
        *set_flags = true;
    }
}

void Vu::Lower(u32 word, u32 pc, Pending& out, u32* branch) {
    const u32 op = word >> 25, dest = Field(word, 21, 4);
    const u32 it = Field(word, 16, 5), is = Field(word, 11, 5);
    u32* vi = regs.vc;
    auto setvi = [&](u32 r, u32 v) { if (r > 0 && r < 16) vi[r] = v & 0xFFFF; };
    auto load = [&](u32 address) {
        const u32* q = Quad(address);
        out.vf = int(it);
        out.lanes = dest;
        std::memcpy(out.value, q, 16);
    };
    auto store = [&](u32 address, const u32* v) {
        u32* q = Quad(address);
        for (int lane = 0; lane < 4; ++lane)
            if (dest & (8u >> lane)) q[lane] = v[lane];
    };
    auto through_cop2 = [&]() {
        // Run it as macro mode would, then hold its writes back until the upper
        // has read its operands.
        u32 before_vc[32], before_vf[4];
        std::memcpy(before_vc, regs.vc, sizeof before_vc);
        std::memcpy(before_vf, regs.vf[it], sizeof before_vf);
        lifted::Cop2(regs, kCop2 | (word & 0x01FFFFFF));
        if (std::memcmp(before_vf, regs.vf[it], sizeof before_vf) != 0) {
            out.vf = int(it);
            out.lanes = dest;
            std::memcpy(out.value, regs.vf[it], 16);
            std::memcpy(regs.vf[it], before_vf, sizeof before_vf);
        }
        if (std::memcmp(before_vc, regs.vc, sizeof before_vc) != 0) {
            out.vc = true;
            std::memcpy(out.vc_after, regs.vc, sizeof out.vc_after);
            std::memcpy(regs.vc, before_vc, sizeof before_vc);
        }
    };
    auto efu = [&](double value) {
        out.p = true;
        out.p_value = lifted::Result(value);
        static const int kLatency[16] = {11, 18, 18, 24, 54, 54, 12, 0, 12, 18, 12, 0, 29, 54, 44, 0};
        out.p_latency = kLatency[(((word & 3) | ((word >> 4) & 0x7C)) - 0x70) & 15];
    };
    const u32 target = (pc + 8 + u32(Imm11(word) * 8)) & u32(micro.size() - 1);
    const std::int16_t a = std::int16_t(vi[is]), b = std::int16_t(vi[it]);
    switch (op) {
        case 0x40: {
            const u32 funct = word & 0x3F;
            if (funct >= 0x30 && funct <= 0x35) {
                // IADD ISUB IADDI IAND IOR, as the macro-mode unit has them: they
                // write vi1-vi15, which no upper reads, so they need not wait.
                const u32 id = Field(word, 6, 5);
                if (funct == 0x32) {
                    const std::int32_t imm = (id & 0x10) ? std::int32_t(id) - 32 : std::int32_t(id);
                    setvi(it, u32(std::int32_t(vi[is]) + imm));
                    return;
                }
                const u32 x = vi[is], y = vi[it];
                setvi(id, funct == 0x30 ? x + y : funct == 0x31 ? x - y : funct == 0x34 ? (x & y) : (x | y));
                return;
            }
            if (funct < 0x3C) { through_cop2(); return; }
            const u32 index = (word & 3) | ((word >> 4) & 0x7C);
            const u32 fsf = Field(word, 21, 2);
            const u32* v = regs.vf[is];
            switch (index) {
                case 0x34: load(vi[is]); setvi(is, vi[is] + 1); return;                 // LQI
                case 0x35: store(vi[it], regs.vf[is]); setvi(it, vi[it] + 1); return;   // SQI
                case 0x36: setvi(is, vi[is] - 1); load(vi[is]); return;                 // LQD
                case 0x37: setvi(it, vi[it] - 1); store(vi[it], regs.vf[is]); return;   // SQD
                case 0x3E: {                                                            // ILWR
                    const u32* q = Quad(vi[is]);
                    for (int lane = 0; lane < 4; ++lane)
                        if (dest & (8u >> lane)) { setvi(it, q[lane]); break; }
                    return;
                }
                case 0x3F: {                                                            // ISWR
                    u32* q = Quad(vi[is]);
                    for (int lane = 0; lane < 4; ++lane)
                        if (dest & (8u >> lane)) q[lane] = vi[it] & 0xFFFF;
                    return;
                }
                case 0x64: {                                                            // MFP
                    out.vf = int(it);
                    out.lanes = dest;
                    for (u32& x : out.value) x = p;
                    return;
                }
                case 0x68: setvi(it, top); return;                                      // XTOP
                case 0x69: setvi(it, itop); return;                                     // XITOP
                case 0x6C: Kick(vi[is]); return;                                        // XGKICK
                case 0x70: case 0x71: case 0x72: case 0x73: {   // ESADD ERSADD ELENG ERLENG
                    const double x = lifted::Value(v[0]), y = lifted::Value(v[1]), z = lifted::Value(v[2]);
                    const double sum = x * x + y * y + z * z;
                    efu(index == 0x70 ? sum : index == 0x71 ? 1.0 / sum
                        : index == 0x72 ? std::sqrt(sum) : 1.0 / std::sqrt(sum));
                    return;
                }
                case 0x74: efu(std::atan2(lifted::Value(v[1]), lifted::Value(v[0]))); return;  // EATANxy
                case 0x75: efu(std::atan2(lifted::Value(v[2]), lifted::Value(v[0]))); return;  // EATANxz
                case 0x76: efu(lifted::Value(v[0]) + lifted::Value(v[1]) + lifted::Value(v[2]) +
                               lifted::Value(v[3])); return;                                   // ESUM
                case 0x78: efu(std::sqrt(std::fabs(lifted::Value(v[fsf])))); return;          // ESQRT
                case 0x79: efu(1.0 / std::sqrt(std::fabs(lifted::Value(v[fsf])))); return;    // ERSQRT
                case 0x7A: efu(1.0 / lifted::Value(v[fsf])); return;                          // ERCPR
                case 0x7B: return;                                                            // WAITP: a stall (Run)
                case 0x7C: efu(std::sin(lifted::Value(v[fsf]))); return;                      // ESIN
                case 0x7D: efu(std::atan(lifted::Value(v[fsf]))); return;                     // EATAN
                case 0x7E: efu(std::exp(-lifted::Value(v[fsf]))); return;                     // EEXP
                case 0x38: case 0x39: case 0x3A:                                            // DIV SQRT RSQRT
                    through_cop2();
                    out.q_latency = index == 0x3A ? 13 : 7;
                    return;
                case 0x3B: return;                                                            // WAITQ: a stall (Run)
                default: through_cop2(); return;    // MOVE MR32 MTIR MFIR R*
            }
        }
        case 0x00: load(vi[is] + u32(Imm11(word))); return;                     // LQ
        case 0x01: store(vi[it] + u32(Imm11(word)), regs.vf[is]); return;       // SQ
        case 0x04: {                                                             // ILW
            const u32* q = Quad(vi[is] + u32(Imm11(word)));
            for (int lane = 0; lane < 4; ++lane)
                if (dest & (8u >> lane)) { setvi(it, q[lane]); break; }
            return;
        }
        case 0x05: {                                                             // ISW
            u32* q = Quad(vi[is] + u32(Imm11(word)));
            for (int lane = 0; lane < 4; ++lane)
                if (dest & (8u >> lane)) q[lane] = vi[it] & 0xFFFF;
            return;
        }
        case 0x08: setvi(it, vi[is] + ((Field(word, 21, 4) << 11) | Field(word, 0, 11))); return;  // IADDIU
        case 0x09: setvi(it, vi[is] - ((Field(word, 21, 4) << 11) | Field(word, 0, 11))); return;  // ISUBIU
        case 0x10: setvi(1, (clip & 0xFFFFFF) == (word & 0xFFFFFF)); return;                        // FCEQ
        case 0x11: clip = clip_latest_ = word & 0xFFFFFF; return;                                   // FCSET
        case 0x12: setvi(1, (clip & word & 0xFFFFFF) != 0); return;                                 // FCAND
        case 0x13: setvi(1, ((clip | word) & 0xFFFFFF) == 0xFFFFFF); return;                        // FCOR
        case 0x14: case 0x15: case 0x16: case 0x17: {                           // FSEQ FSSET FSAND FSOR
            const u32 imm = (Field(word, 21, 1) << 11) | Field(word, 0, 11);
            if (op == 0x14) setvi(it, (status & 0xFFF) == imm);
            else if (op == 0x15) {
                status = (status & 0x3F) | (imm & 0xFC0);
                status_latest_ = (status_latest_ & 0x3F) | (imm & 0xFC0);
            }
            else if (op == 0x16) setvi(it, status & imm);
            else setvi(it, ((status | imm) & 0xFFF) == 0xFFF);
            return;
        }
        case 0x18: setvi(it, (vi[is] & 0xFFFF) == (mac & 0xFFFF)); return;     // FMEQ
        case 0x1A: setvi(it, vi[is] & mac); return;                              // FMAND
        case 0x1B: setvi(it, vi[is] | mac); return;                              // FMOR
        case 0x1C: setvi(it, clip & 0xFFF); return;                              // FCGET
        case 0x20: *branch = target; return;                                     // B
        case 0x21: setvi(it, (pc + 16) / 8); *branch = target; return;           // BAL
        case 0x24: *branch = (vi[is] * 8) & u32(micro.size() - 1); return;       // JR
        case 0x25: {                                                             // JALR
            const u32 to = (vi[is] * 8) & u32(micro.size() - 1);
            setvi(it, (pc + 16) / 8);
            *branch = to;
            return;
        }
        case 0x28: if (a == b) *branch = target; return;                         // IBEQ
        case 0x29: if (a != b) *branch = target; return;                         // IBNE
        case 0x2C: if (a < 0) *branch = target; return;                          // IBLTZ
        case 0x2D: if (a > 0) *branch = target; return;                          // IBGTZ
        case 0x2E: if (a <= 0) *branch = target; return;                         // IBLEZ
        case 0x2F: if (a >= 0) *branch = target; return;                         // IBGEZ
        default: return;
    }
}

std::vector<std::pair<Vu::u64, Vu::u32>> Vu::ClipsInFlight() const {
    std::vector<std::pair<u64, u32>> out;
    for (int i = 0; i < flag_count_; ++i) {
        const FlagWrite& f = flag_queue_[(flag_head_ + i) & 7];
        if (f.sets_clip) out.emplace_back(f.ready, f.clip);
    }
    return out;
}

void Vu::LandAll() {
    cycle_ += 64;                   // time passes after the program: what was in flight lands
    Land();
}

namespace {

// --vu-snapshot and --vu-bench (Vu::Snapshots, Vu::Bench): a run as it starts.
struct Snapshot {
    u32 pc;
    u32 flags[6];                       // mac, status, clip, p, top, itop
    u32 vf[32][4], vc[32], vacc[4];
};
std::FILE* g_snapshots = nullptr;
std::atomic<u64> g_snapshot_from{~u64(0)};
std::atomic<bool> g_snapshotting{false};
u32 g_snapshots_left = 0;

}  // namespace

void Vu::Snapshots(const std::string& path, u64 from, u32 count) {
    g_snapshots = std::fopen(path.c_str(), "wb");
    if (!g_snapshots) {
        std::fprintf(stderr, "vu: cannot write %s\n", path.c_str());
        return;
    }
    g_snapshot_from = from;
    g_snapshots_left = count;
}

void Vu::SnapshotFrame(u64 frame) {
    if (g_snapshots && frame >= g_snapshot_from.load()) g_snapshotting = true;
}

int Vu::Bench(const std::string& path, int passes, bool fast) {
    std::FILE* in = std::fopen(path.c_str(), "rb");
    if (!in) {
        std::fprintf(stderr, "vu: cannot read %s\n", path.c_str());
        return 1;
    }
    struct Run {
        Snapshot s;
        std::vector<std::uint8_t> micro, data;
    };
    std::vector<Run> runs;
    for (;;) {
        Run r;
        r.micro.resize(16384);
        r.data.resize(16384);
        if (std::fread(&r.s, sizeof r.s, 1, in) != 1 || std::fread(r.micro.data(), 1, 16384, in) != 16384 ||
            std::fread(r.data.data(), 1, 16384, in) != 16384)
            break;
        runs.push_back(std::move(r));
    }
    std::fclose(in);
    Vu vu(1);
    vu.fast = fast;
    vu.on_kick = [](u64, u64) {};
    vu.on_kick_block = [](const u32*, u32) {};
    // PS2HOST_VU_FIND=x,y,z,w (hex words): the run that leaves that quadword
    // in data memory, the pair that first stores it (the interpreter, run
    // again with a growing limit) and the registers before and after.
    if (const char* find = std::getenv("PS2HOST_VU_FIND")) {
        u32 want[4] = {};
        if (std::sscanf(find, "%x,%x,%x,%x", &want[0], &want[1], &want[2], &want[3]) != 4) return 64;
        vu.use_compiled = false;
        auto load = [&](const Run& r) {
            std::memcpy(vu.micro.data(), r.micro.data(), 16384);
            vu.MicroWritten();
            std::memcpy(vu.data.data(), r.data.data(), 16384);
            std::memcpy(vu.regs.vf, r.s.vf, sizeof r.s.vf);
            std::memcpy(vu.regs.vc, r.s.vc, sizeof r.s.vc);
            std::memcpy(vu.regs.vacc, r.s.vacc, sizeof r.s.vacc);
            vu.mac = r.s.flags[0];
            vu.status = r.s.flags[1];
            vu.clip = r.s.flags[2];
            vu.p = r.s.flags[3];
            vu.top = r.s.flags[4];
            vu.itop = r.s.flags[5];
        };
        auto holds = [&]() -> int {
            for (u32 q = 0; q < 1024; ++q) {
                u32 w[4];
                std::memcpy(w, vu.data.data() + q * 16, 16);
                if ((w[0] & 0xFF) == want[0] && (w[1] & 0xFF) == want[1] && (w[2] & 0xFF) == want[2] && (w[3] & 0xFF) == want[3]) return int(q);    // (PACKED RGBAQ: the low bytes)
                if (want[1] == 0xFFFFFFFFu && (w[0] == want[0] || w[1] == want[0] || w[2] == want[0] || w[3] == want[0])) return int(q);    // x,FFFFFFFF,0,0: one word anywhere
            }
            return -1;
        };
        auto dump = [&](const char* when) {
            std::printf("  %s (pc %04X):\n", when, vu.resume * 1);
            for (int i = 0; i < 32; ++i) {
                float f[4];
                std::memcpy(f, vu.regs.vf[i], 16);
                std::printf("    vf%02d %12g %12g %12g %12g   [%08X %08X %08X %08X]\n", i, f[0], f[1], f[2], f[3],
                            vu.regs.vf[i][0], vu.regs.vf[i][1], vu.regs.vf[i][2], vu.regs.vf[i][3]);
            }
            std::printf("    vi:");
            for (int i = 0; i < 16; ++i) std::printf(" %X", vu.regs.vc[i] & 0xFFFF);
            float q, acc[4];
            std::memcpy(&q, &vu.regs.vc[22], 4);
            std::memcpy(acc, vu.regs.vacc, 16);
            std::printf("\n    Q %g  I %08X  ACC %g %g %g %g\n", q, vu.regs.vc[21], acc[0], acc[1], acc[2], acc[3]);
        };
        // PS2HOST_VU_STEPS=run,first,last: that run's registers after each pair count.
        if (const char* steps = std::getenv("PS2HOST_VU_STEPS")) {
            unsigned run = 0; unsigned long long first = 0, last = 0;
            std::sscanf(steps, "%u,%llu,%llu", &run, &first, &last);
            for (unsigned long long k = first; k <= last && run < runs.size(); ++k) {
                load(runs[run]);
                vu.Run(runs[run].s.pc, k);
                float acc[4];
                std::memcpy(acc, vu.regs.vacc, 16);
                std::printf("%llu pc %04X ACC %g %g %g %g", k, vu.resume * 1, acc[0], acc[1], acc[2], acc[3]);
                for (int r : {18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}) {
                    float f[4];
                    std::memcpy(f, vu.regs.vf[r], 16);
                    std::printf(" | vf%d %g %g %g %g", r, f[0], f[1], f[2], f[3]);
                }
                std::printf("\n");
            }
            return 0;
        }
        for (std::size_t n = 0; n < runs.size(); ++n) {
            load(runs[n]);
            if (holds() >= 0) continue;                 // there before it ran
            const u64 total = vu.Run(runs[n].s.pc);
            const int q = holds();
            if (q < 0) continue;
            std::printf("run %zu (pc %04X, %llu pairs) stores it at quadword %d\n", n, runs[n].s.pc,
                        (unsigned long long)total, q);
            u64 lo = 0, hi = total;                     // after hi pairs it is there; after lo not
            while (hi - lo > 1) {
                const u64 mid = (lo + hi) / 2;
                load(runs[n]);
                vu.Run(runs[n].s.pc, mid);
                (holds() >= 0 ? hi : lo) = mid;
            }
            load(runs[n]);
            vu.Run(runs[n].s.pc, lo);
            std::printf("  the storing pair: after %llu pairs\n", (unsigned long long)lo);
            dump("before the storing pair");
            load(runs[n]);
            vu.Run(runs[n].s.pc, hi);
            dump("after it");
            return 0;
        }
        std::printf("no run stores it\n");
        return 1;
    }
    double best = 0;
    u64 pairs = 0, hash = 0;
    for (int pass = 0; pass < passes; ++pass) {
        double seconds = 0;
        u64 pass_pairs = 0, pass_hash = 1469598103934665603ull;
        std::uint8_t* micro_now = nullptr;
        for (Run& r : runs) {
            if (micro_now != r.micro.data() &&
                (!micro_now || std::memcmp(micro_now, r.micro.data(), 16384) != 0)) {
                std::memcpy(vu.micro.data(), r.micro.data(), 16384);
                vu.MicroWritten();
            }
            micro_now = r.micro.data();
            std::memcpy(vu.data.data(), r.data.data(), 16384);
            std::memcpy(vu.regs.vf, r.s.vf, sizeof r.s.vf);
            std::memcpy(vu.regs.vc, r.s.vc, sizeof r.s.vc);
            std::memcpy(vu.regs.vacc, r.s.vacc, sizeof r.s.vacc);
            vu.mac = r.s.flags[0];
            vu.status = r.s.flags[1];
            vu.clip = r.s.flags[2];
            vu.p = r.s.flags[3];
            vu.top = r.s.flags[4];
            vu.itop = r.s.flags[5];
            const auto began = std::chrono::steady_clock::now();
            pass_pairs += vu.Run(r.s.pc);
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            // FNV-1a, a word at a time, over what the run left: registers, flags, data memory.
            auto mix = [&pass_hash](const void* p, std::size_t n) {
                const std::uint8_t* b = static_cast<const std::uint8_t*>(p);
                for (std::size_t i = 0; i + 8 <= n; i += 8) {      // (every size here is a multiple of 8)
                    std::uint64_t word;
                    std::memcpy(&word, b + i, 8);
                    pass_hash = (pass_hash ^ word) * 1099511628211ull;
                }
            };
            mix(vu.regs.vf, sizeof vu.regs.vf);
            mix(vu.regs.vc, sizeof vu.regs.vc);
            mix(vu.regs.vacc, sizeof vu.regs.vacc);
            const u32 flags[4] = {vu.mac, vu.status, vu.clip, vu.p};
            mix(flags, sizeof flags);
            mix(vu.data.data(), 16384);
        }
        if (pass == 0 || seconds < best) best = seconds;
        pairs = pass_pairs;
        hash = pass_hash;
    }
    std::printf("vu-bench: %zu runs, %llu pairs, best pass %.1f ms, %.2f ns a pair; state hash %016llX\n",
                runs.size(), (unsigned long long)pairs, best * 1000, pairs ? best * 1e9 / double(pairs) : 0.0,
                (unsigned long long)hash);
    return 0;
}

namespace {
/* The VUs' arithmetic as the console's: rounded toward zero, denormals flushed
   (inputs and results) - the host's SSE control word, for a run. Burnout 3's
   particles reduce an angle by adding 2^23 and taking it away again: rounded
   to nearest it came out in [0, 1], not [-0.5, 0.5], and the sine polynomial
   after it gave corners twelve times too long - huge smoke puffs lit with
   wrapped colours (the race's white-outs and stretched textures). */
struct VuRounding {
    unsigned saved = _mm_getcsr();
    VuRounding() { _mm_setcsr((saved & ~0x6000u) | 0x6000u | 0x8000u | 0x0040u); }
    ~VuRounding() { _mm_setcsr(saved); }
};
}  // namespace

Vu::u64 Vu::Run(u32 pc, u64 limit) {
    Timed timed(data_mask_ == 0x3FF ? Section::Vu1 : Section::Vu0);
    VuRounding rounding;
    const int unit = data_mask_ == 0x3FF ? 1 : 0;
    pc &= u32(micro.size() - 1) & ~7u;
    ++starts_[pc];
    if (g_record) RecordProgram(micro, unit, pc);
    if (unit == 1 && g_snapshotting.load(std::memory_order_relaxed) && g_snapshots_left) {
        Snapshot s{};
        s.pc = pc;
        const u32 flags[6] = {mac, status, clip, p, top, itop};
        std::memcpy(s.flags, flags, sizeof flags);
        std::memcpy(s.vf, regs.vf, sizeof s.vf);
        std::memcpy(s.vc, regs.vc, sizeof s.vc);
        std::memcpy(s.vacc, regs.vacc, sizeof s.vacc);
        std::fwrite(&s, sizeof s, 1, g_snapshots);
        std::fwrite(micro.data(), 1, 16384, g_snapshots);
        std::fwrite(data.data(), 1, 16384, g_snapshots);
        if (--g_snapshots_left == 0) {
            std::fclose(g_snapshots);
            g_snapshots = nullptr;
            g_snapshotting = false;
        }
    }
    VuRunState state;
    state.pc = pc;
    unclean_ = ~1u;                     // nothing known about the registers yet
    acc_unclean_ = true;
    RunCompiled(state, limit);
    if (!state.done) Interpret(state, limit);
    LandAll();
    resume = state.pc;
    pairs_ += state.count;
    return state.count;
}

bool Vu::RunCompiled(VuRunState& state, u64 limit) {
    if (!use_compiled || Compiled().empty()) return false;
    const int unit = data_mask_ == 0x3FF ? 1 : 0;
    bool ran = false;
    // From program to program (a jump to a register's address may go to
    // another), as long as one starts here and its code is in memory.
    while (!state.done && state.count < limit && state.delayed == kNone && !state.stopping) {
        const auto found = Compiled().find(std::uint64_t(unit) << 32 | state.pc);
        if (found == Compiled().end()) {
            if (g_record && ran) RecordProgram(micro, unit, state.pc);
            break;
        }
        const VuCompiled* program = nullptr;
        for (const VuCompiled* candidate : found->second) {
            u64& held_at = held_at_[candidate];
            if (held_at == micro_generation_) { program = candidate; break; }
            compiled_checked_ += 4 * (candidate->code[2] + 2);
            if (Holds(micro, candidate->code)) {
                held_at = micro_generation_;
                program = candidate;
                break;
            }
        }
        if (!program) {
            ++compiled_stale_;
            break;
        }
        ++compiled_entries_;
        const std::vector<VuBlock>& table = BlockTable(program, micro.size() / 8, fast);
        const u64 before = state.count;
        const u32 started = state.pc;
        while (!state.done && state.count < limit) {
            const VuBlock block = table[(state.pc & u32(micro.size() - 1)) / 8];
            if (!block) break;
            const u64 at = state.count;
            block(*this, state, limit);
            if (state.count == at) break;   // it would not start here: the interpreter goes on
        }
        compiled_pairs_ += state.count - before;
        ran = true;
        if (state.pc == started && state.count == before) break;     // nothing taken
    }
    return ran;
}

void Vu::Interpret(VuRunState& state, u64 limit) {
    unclean_ = ~1u;                     // the interpreter does not keep track
    acc_unclean_ = true;
    u64 count = state.count;
    u32 delayed = state.delayed;
    bool stopping = state.stopping;
    u32 pc = state.pc;
    bool delayed_by_register = false;   // the branch delayed is JR's or JALR's
    const int unit = data_mask_ == 0x3FF ? 1 : 0;
    // One cache a unit size; an entry stands only while its words are there.
    static std::vector<DecodedPair> caches[2];
    std::vector<DecodedPair>& decoded_ = caches[data_mask_ == 0x3FF ? 1 : 0];
    if (decoded_.size() != micro.size() / 8) decoded_.assign(micro.size() / 8, DecodedPair{});
    while (count < limit) {
        u32 lower, upper;
        std::memcpy(&lower, micro.data() + pc, 4);
        std::memcpy(&upper, micro.data() + pc + 4, 4);
        const DecodedPair& d = Decoded(decoded_, pc, lower, upper);
        const bool immediate = d.immediate;

        // Stalls: operands still in the pipeline, a busy divider or EFU, the waits.
        const Reads& reads = d.reads;
        u64 issue = cycle_;
        for (int i = 0; i < reads.n; ++i)
            for (int lane = 0; lane < 4; ++lane)
                if (reads.lanes[i] & (8u >> lane)) issue = std::max(issue, vf_ready_[reads.reg[i]][lane]);
        if (q_busy_ && d.q_waits) issue = std::max(issue, q_ready_);
        if (p_busy_ && d.p_waits) issue = std::max(issue, p_ready_);
        cycle_ = issue;
        Land();

        Pending pending;
        u32 branch = kNone;
        if (!immediate) Lower(lower, pc, pending, &branch);
        int written = -1;
        u32 lanes = 0;
        bool set_flags = false;
        Upper(upper, &written, &lanes, &set_flags, d.has_fmac ? &d.fmac : nullptr);
        // LOI: the upper beside it read the I from before.
        if (immediate) regs.vc[kRegI] = lower;
        if (pending.vf > 0) {
            for (int lane = 0; lane < 4; ++lane) {
                const u32 bit = 8u >> lane;
                if ((pending.lanes & bit) && !(written == pending.vf && (lanes & bit))) {
                    regs.vf[pending.vf][lane] = pending.value[lane];
                    vf_ready_[pending.vf][lane] = cycle_ + kFmacLatency;
                }
            }
        }
        if (written > 0)
            for (int lane = 0; lane < 4; ++lane)
                if (lanes & (8u >> lane)) vf_ready_[written][lane] = cycle_ + kFmacLatency;
        if (set_flags)
            PushFlags({cycle_ + kFmacLatency, mac_latest_, status_latest_, clip_latest_, (upper & 0x7FF) != 0x1FF,
                       (upper & 0x7FF) == 0x1FF});
        if (pending.q_latency) {
            // (A quotient equal to the Q before changes no register: it is that Q.)
            q_next_ = pending.vc ? pending.vc_after[kRegQ] : regs.vc[kRegQ];
            q_ready_ = cycle_ + u64(pending.q_latency);
            q_busy_ = true;
            qp_due_ = std::min(qp_due_, q_ready_);
            if (pending.vc) pending.vc_after[kRegQ] = regs.vc[kRegQ];
        }
        if (pending.vc) std::memcpy(regs.vc, pending.vc_after, sizeof pending.vc_after);
        if (pending.p) {
            p_next_ = pending.p_value;
            p_ready_ = cycle_ + u64(pending.p_latency ? pending.p_latency : 1);
            p_busy_ = true;
            qp_due_ = std::min(qp_due_, p_ready_);
        }
        regs.vc[0] = 0;
        ++cycle_;
        ++count;
        if (stopping) {                      // the pair after an E bit runs, then the program ends
            pc = (pc + 8) & u32(micro.size() - 1);
            state.done = true;
            break;
        }
        if (upper & 0x40000000u) stopping = true;
        const bool jumped = delayed != kNone, by_register = delayed_by_register;
        const u32 next = jumped ? delayed : (pc + 8) & u32(micro.size() - 1);
        delayed = branch;
        delayed_by_register = branch != kNone && ((lower >> 25) == 0x24 || (lower >> 25) == 0x25);
        pc = next;
        if (jumped && delayed == kNone && !stopping) {
            // Where a jump to a register's address goes is a program of its own
            // (one recorded, and compiled, from there); compiled code, if any
            // starts here, goes on.
            if (g_record && by_register) RecordProgram(micro, unit, pc);
            state.pc = pc;
            state.count = count;
            state.delayed = delayed;
            state.stopping = stopping;
            if (RunCompiled(state, limit)) {
                if (state.done) return;
                pc = state.pc;
                count = state.count;
                delayed = state.delayed;
                stopping = state.stopping;
                delayed_by_register = false;
            }
        }
    }
    state.pc = pc;
    state.count = count;
    state.delayed = delayed;
    state.stopping = stopping;
}

Vu& Vu1() {
    static Vu vu(1);
    static const bool attached = [] {
        // The EE sees VU1's memories at 0x11008000 (instructions) and
        // 0x1100C000 (data).
        vu.Attach(GuestMemory().Translate(0x11008000u, 16384), GuestMemory().Translate(0x1100C000u, 16384));
        return true;
    }();
    (void)attached;
    return vu;
}

Vu& Vu0() {
    static Vu vu(0);
    static const bool attached = [] {
        // VU0's at 0x11000000 and 0x11004000.
        vu.Attach(GuestMemory().Translate(0x11000000u, 4096), GuestMemory().Translate(0x11004000u, 4096));
        return true;
    }();
    (void)attached;
    return vu;
}

}  // namespace ps2host
