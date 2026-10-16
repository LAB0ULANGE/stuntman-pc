/*
 * VIFcode decoding (EE User's Manual, chapter 6). A VIFcode is
 *   bits  0-15 IMMEDIATE, 16-23 NUM, 24-30 CMD, 31 interrupt.
 * Data lengths, in 32-bit words:
 *   STMASK 1; STROW, STCOL 4; MPG NUM*2 (NUM 0 = 256);
 *   DIRECT, DIRECTHL IMMEDIATE*4 (0 = 65536 quadwords);
 *   UNPACK (0x60-0x7F) by its element format, NUM and STCYCL's CL and WL;
 *   every other command none.
 *
 * UNPACK's CMD is 0x60 | m << 4 | vn << 2 | vl: m masks, vn + 1 components of
 * 32 >> vl bits (vn 3, vl 3 is V4-5, 16 bits of RGBA 5-5-5-1). IMMEDIATE holds
 * the address in quadwords (bits 0-9), USN (14: zero-extend, not sign-extend)
 * and FLG (15: relative to TOPS). With CL >= WL each CL quadwords take WL
 * elements and skip the rest; with WL > CL each WL take CL elements and the
 * rest are filled from the row and column registers by the mask.
 */

#include "ps2host/vif.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "ps2host/dma.h"
#include "ps2host/vu.h"

namespace ps2host {

std::uint32_t VifStream::UnpackWords(std::uint32_t format, std::uint32_t num) const {
    const std::uint32_t vl = format & 3, vn = (format >> 2) & 3;
    // Bits per element: components (vn + 1) of 32 >> vl bits; V4-5 packs 16.
    const std::uint32_t bits = (vn == 3 && vl == 3) ? 16 : (vn + 1) * (32u >> vl);
    // With WL > CL (filling writes) only CL of every WL elements come from data.
    std::uint32_t elements = num;
    if (write_length_ > cycle_length_ && write_length_ != 0) {
        const std::uint32_t partial = num % write_length_;
        elements = cycle_length_ * (num / write_length_) +
                   (partial < cycle_length_ ? partial : cycle_length_);
    }
    return (elements * bits + 31) / 32;
}

void VifStream::WriteElement(const std::uint32_t value[4], bool from_data) {
    Unpack& u = unpack_;
    if (u.written >= u.num) return;
    std::uint32_t cl = cycle_length_, wl = write_length_;
    if (wl == 0) { wl = 1; cl = std::max(cl, 1u); }
    const std::uint32_t k = u.written++;
    const std::uint32_t address = wl <= cl ? u.address + (k / wl) * cl + (k % wl) : u.address + k;
    if (!vu) return;
    std::uint32_t* q = vu->Quad(address);
    const bool three = ((u.format >> 2) & 3) == 2;
    if (from_data && !u.masked && mode == 0) {      // the usual case: the element as it is
        std::memcpy(q, value, three ? 12 : 16);
        return;
    }
    const std::uint32_t position = std::min(k % wl, 3u);
    for (std::uint32_t f = 0; f < 4; ++f) {
        std::uint32_t m = u.masked ? (mask >> (position * 8 + f * 2)) & 3 : 0;
        if (!from_data && m == 0) m = 1;            // a filling write: the row
        switch (m) {
            case 0: {
                if (three && f == 3) break;         // V3 leaves w as it was
                std::uint32_t v = value[f];
                if (mode == 1) v += row[f];
                else if (mode == 2) row[f] = v = v + row[f];
                q[f] = v;
                break;
            }
            case 1: q[f] = row[f]; break;
            case 2: q[f] = col[position]; break;
            default: break;                         // write-protected
        }
    }
}

void VifStream::UnpackFill() {
    Unpack& u = unpack_;
    if (write_length_ <= cycle_length_) return;
    const std::uint32_t zero[4] = {};
    while (u.written < u.num && (u.written % write_length_) >= cycle_length_) WriteElement(zero, false);
}

void VifStream::Element(const std::uint8_t* b, std::uint32_t c[4]) const {
    const Unpack& u = unpack_;
    const std::uint32_t vl = u.format & 3, vn = (u.format >> 2) & 3;
    c[0] = c[1] = c[2] = c[3] = 0;
    if (vn == 3 && vl == 3) {                       // V4-5
        const std::uint32_t v = std::uint32_t(b[0]) | std::uint32_t(b[1]) << 8;
        c[0] = (v & 0x1F) << 3;
        c[1] = ((v >> 5) & 0x1F) << 3;
        c[2] = ((v >> 10) & 0x1F) << 3;
        c[3] = ((v >> 15) & 1) << 7;
        return;
    }
    const std::uint32_t width = 4u >> vl;           // bytes per component
    for (std::uint32_t i = 0; i <= vn; ++i) {
        const std::uint8_t* e = b + i * width;
        if (width == 4) std::memcpy(&c[i], e, 4);
        else if (width == 2) {
            const std::uint16_t h = std::uint16_t(e[0] | e[1] << 8);
            c[i] = u.unsigned_ ? h : std::uint32_t(std::int32_t(std::int16_t(h)));
        } else {
            c[i] = u.unsigned_ ? e[0] : std::uint32_t(std::int32_t(std::int8_t(e[0])));
        }
    }
    if (vn == 0) c[1] = c[2] = c[3] = c[0];             // S: every field
    else if (vn == 1) { c[2] = c[0]; c[3] = c[1]; }     // V2: x y x y
}

void VifStream::UnpackData(std::uint32_t word) {
    Unpack& u = unpack_;
    for (int i = 0; i < 4; ++i) u.bytes.push_back(std::uint8_t(word >> (8 * i)));
    const std::uint32_t vl = u.format & 3, vn = (u.format >> 2) & 3;
    const bool v45 = vn == 3 && vl == 3;
    const std::uint32_t size = v45 ? 2 : (vn + 1) * (4u >> vl);
    std::size_t at = 0;
    while (u.bytes.size() - at >= size && u.written < u.num) {
        std::uint32_t c[4];
        Element(u.bytes.data() + at, c);
        at += size;
        WriteElement(c, true);
        UnpackFill();
    }
    u.bytes.erase(u.bytes.begin(), u.bytes.begin() + std::ptrdiff_t(at));
}

void VifStream::UnpackRun(const std::uint32_t* words, std::uint32_t count) {
    Unpack& u = unpack_;
    const std::uint32_t vl = u.format & 3, vn = (u.format >> 2) & 3;
    const std::uint32_t size = (vn == 3 && vl == 3) ? 2 : (vn + 1) * (4u >> vl);
    // The usual case, in one loop: unmasked, no addition (STMOD 0), CL >= WL
    // (no filling writes), elements of whole words, none part-read - each
    // stored as WriteElement would, its address kept by counting rather than
    // dividing.
    if (u.bytes.empty() && size % 4 == 0 && !u.masked && mode == 0 && vu && write_length_ != 0 &&
        write_length_ <= cycle_length_ && (u.format == 0xC || u.format == 0x8 || u.format == 0x4 ||
                                           u.format == 0x0 || u.format == 0xD || u.format == 0x5 ||
                                           u.format == 0xE)) {
        const std::uint32_t per = size / 4, wl = write_length_, cl = cycle_length_;
        std::uint32_t k = u.written, in_cycle = k % wl, base = u.address + (k / wl) * cl, used = 0;
        const bool zero_extend = u.unsigned_;
        auto half = [zero_extend](std::uint32_t h) {
            return zero_extend ? (h & 0xFFFF) : std::uint32_t(std::int32_t(std::int16_t(std::uint16_t(h))));
        };
        auto byte = [zero_extend](std::uint32_t b) {
            return zero_extend ? (b & 0xFF) : std::uint32_t(std::int32_t(std::int8_t(std::uint8_t(b))));
        };
        while (count - used >= per && k < u.num) {
            std::uint32_t* q = vu->Quad(base + in_cycle);
            const std::uint32_t* w = words + used;
            switch (u.format) {
                case 0xC: std::memcpy(q, w, 16); break;                             // V4-32
                case 0x8: std::memcpy(q, w, 12); break;                             // V3-32: w stays
                case 0x4: q[0] = w[0]; q[1] = w[1]; q[2] = w[0]; q[3] = w[1]; break; // V2-32: x y x y
                case 0x0: q[0] = q[1] = q[2] = q[3] = w[0]; break;                  // S-32
                case 0xD:                                                           // V4-16
                    q[0] = half(w[0]); q[1] = half(w[0] >> 16); q[2] = half(w[1]); q[3] = half(w[1] >> 16);
                    break;
                case 0x5: {                                                         // V2-16
                    const std::uint32_t x = half(w[0]), y = half(w[0] >> 16);
                    q[0] = x; q[1] = y; q[2] = x; q[3] = y;
                    break;
                }
                default:                                                            // V4-8
                    q[0] = byte(w[0]); q[1] = byte(w[0] >> 8); q[2] = byte(w[0] >> 16); q[3] = byte(w[0] >> 24);
                    break;
            }
            used += per;
            ++k;
            if (++in_cycle == wl) {
                in_cycle = 0;
                base += cl;
            }
        }
        u.written = k;
        words += used;
        count -= used;
    }
    // Elements of whole words, none part-read: straight from the words (the
    // bytes are the same; the byte buffer would hold just these).
    if (u.bytes.empty() && size % 4 == 0) {
        const std::uint32_t per = size / 4;
        const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(words);
        std::uint32_t used = 0;
        while (count - used >= per && u.written < u.num) {
            std::uint32_t c[4];
            Element(bytes + std::size_t(used) * 4, c);
            used += per;
            WriteElement(c, true);
            UnpackFill();
        }
        words += used;
        count -= used;
    }
    for (std::uint32_t i = 0; i < count; ++i) UnpackData(words[i]);
}

void VifStream::Start(std::uint32_t pc) {
    if (double_buffered) {
        top = tops;
        dbf ^= 1;
        tops = base + (dbf ? offset : 0);
    }
    itop = itops;
    if (!vu) return;
    vu->top = top;
    vu->itop = itop;
    vu->Run(pc);
    end_pc_ = vu->resume;
}

void VifStream::Words(const std::uint32_t* words, std::uint32_t count, std::uint32_t from) {
    for (std::uint32_t i = 0; i < count;) {
        if (pending_ && data_ == Data::Unpack) {
            const std::uint32_t run = std::min(pending_, count - i);
            pending_ -= run;
            data_words_ += run;
            data_index_ += run;
            g_dma_source = from + 4 * i;
            UnpackRun(words + i, run);
            i += run;
            continue;
        }
        if (pending_ && data_ == Data::Direct && on_direct_block) {
            const std::uint32_t run = std::min(pending_, count - i);
            pending_ -= run;
            data_words_ += run;
            data_index_ += run;
            on_direct_block(words + i, run, from + 4 * i);
            i += run;
            continue;
        }
        g_dma_source = from + 4 * i;
        Word(words[i++]);
    }
}

void VifStream::Word(std::uint32_t word) {
    if (pending_) {
        --pending_;
        ++data_words_;
        const std::uint32_t index = data_index_++;
        switch (data_) {
            case Data::Direct: if (on_direct) on_direct(word); break;
            case Data::Mpg:
                if (vu) {
                    const std::size_t at = (mpg_address_ + 4 * index) & (vu->micro.size() - 1);
                    // Games upload the same microcode again and again (Burnout 3
                    // every frame): only a word that changes makes the compiled
                    // programs be compared again.
                    if (std::memcmp(vu->micro.data() + at, &word, 4) != 0) {
                        std::memcpy(vu->micro.data() + at, &word, 4);
                        vu->MicroWritten();
                    }
                }
                break;
            case Data::Unpack: UnpackData(word); break;
            case Data::Mask: mask = word; break;
            case Data::Row: row[index & 3] = word; break;
            case Data::Col: col[index & 3] = word; break;
            default: break;
        }
        return;
    }
    data_ = Data::None;
    data_index_ = 0;
    ++commands_;
    const std::uint32_t command = (word >> 24) & 0x7F;
    ++by_command_[command];
    const std::uint32_t num = (word >> 16) & 0xFF;
    const std::uint32_t immediate = word & 0xFFFF;
    if (command >= 0x60) {
        Unpack& u = unpack_;
        u.format = command & 0xF;
        u.masked = (command & 0x10) != 0;
        u.unsigned_ = (immediate & 0x4000) != 0;
        u.address = immediate & 0x3FF;
        if (double_buffered && (immediate & 0x8000)) u.address += tops;
        u.num = num ? num : 256;
        u.written = 0;
        u.bytes.clear();
        pending_ = UnpackWords(command & 0xF, u.num);
        data_ = Data::Unpack;
        UnpackFill();
        return;
    }
    switch (command) {
        case 0x01:                                          // STCYCL
            cycle_length_ = immediate & 0xFF;
            write_length_ = immediate >> 8;
            break;
        case 0x02:                                          // OFFSET
            offset = immediate & 0x3FF;
            dbf = 0;
            tops = base;
            break;
        case 0x03: base = immediate & 0x3FF; break;         // BASE
        case 0x04: itops = immediate & 0x3FF; break;        // ITOP
        case 0x05: mode = immediate & 3; break;             // STMOD
        case 0x06:                                          // MSKPATH3
            if (on_mask_path3) on_mask_path3((immediate & 0x8000) != 0);
            break;
        case 0x14: case 0x15: Start(immediate * 8); break;  // MSCAL, MSCALF
        case 0x17: Start(end_pc_); break;                   // MSCNT
        case 0x20: pending_ = 1; data_ = Data::Mask; break; // STMASK
        case 0x30: pending_ = 4; data_ = Data::Row; break;  // STROW
        case 0x31: pending_ = 4; data_ = Data::Col; break;  // STCOL
        case 0x4A:                                          // MPG
            pending_ = (num ? num : 256) * 2;
            mpg_address_ = immediate * 8;
            data_ = Data::Mpg;
            break;
        case 0x50: case 0x51:                               // DIRECT, DIRECTHL
            pending_ = (immediate ? immediate : 0x10000) * 4;
            data_ = Data::Direct;
            break;
        case 0x13:                                          // FLUSHA: nothing waits here either
            if (on_trace) on_trace("FLUSHA");
            break;
        default: break;                                     // NOP, MARK, FLUSH*: nothing waits
    }
}

std::string VifStream::Summary() const {
    static const std::pair<std::uint32_t, const char*> kNames[] = {
        {0x00, "NOP"}, {0x01, "STCYCL"}, {0x02, "OFFSET"}, {0x03, "BASE"}, {0x04, "ITOP"},
        {0x05, "STMOD"}, {0x06, "MSKPATH3"}, {0x07, "MARK"}, {0x10, "FLUSHE"}, {0x11, "FLUSH"},
        {0x13, "FLUSHA"}, {0x14, "MSCAL"}, {0x15, "MSCALF"}, {0x17, "MSCNT"}, {0x20, "STMASK"},
        {0x30, "STROW"}, {0x31, "STCOL"}, {0x4A, "MPG"}, {0x50, "DIRECT"}, {0x51, "DIRECTHL"}};
    std::string out;
    std::uint64_t unpacks = 0;
    for (std::uint32_t c = 0x60; c < 0x80; ++c) unpacks += by_command_[c];
    for (const auto& [code, name] : kNames)
        if (by_command_[code]) out += std::string(" ") + name + " " + std::to_string(by_command_[code]);
    if (unpacks) out += " UNPACK " + std::to_string(unpacks);
    if (vu) out += "; VU: " + std::to_string(vu->pairs()) + " instruction pairs (" +
                   std::to_string(vu->compiled_pairs()) + " compiled: " + std::to_string(vu->compiled_entries()) +
                   " entered, " + std::to_string(vu->compiled_stale()) + " stale, " +
                   std::to_string(vu->compiled_checked() >> 20) + " MB checked), " + std::to_string(vu->kicks()) +
                   " XGKICKs";
    if (vu) {
        out += "; started at";
        for (const auto& [pc, n] : vu->starts()) {
            char text[32];
            std::snprintf(text, sizeof text, " 0x%X x%llu", unsigned(pc), static_cast<unsigned long long>(n));
            out += text;
        }
    }
    return out;
}

VifStream& Vif1() {
    static VifStream stream;
    return stream;
}

VifStream& Vif0() {
    static VifStream stream;
    return stream;
}

}  // namespace ps2host
