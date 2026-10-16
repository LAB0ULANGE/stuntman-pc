/*
 * Source DMA transfers, walked at once (dma.h).
 */

#include "ps2host/dma.h"

#include <cstdint>
#include <cstring>

#include "ps2host/memory.h"

namespace ps2host {

std::uint32_t g_dma_source = 0;


namespace {

constexpr std::uint32_t kScratchpad = 0x70000000u;

std::uint32_t Address(std::uint64_t tag) {
    const std::uint32_t address = std::uint32_t(tag >> 32) & 0x7FFFFFF0u;
    return (tag >> 63) ? kScratchpad | (address & 0x3FF0u) : address;
}

// A stall `taken` words into a run: the rest of the quadword it was in is
// kept for the transfer to go on with.
void KeepLeftover(DmaOutcome& outcome, const std::uint32_t* run, std::uint32_t count, std::uint32_t taken) {
    const std::uint32_t whole = (taken + 3) / 4 * 4;
    outcome.leftover_count = 0;
    for (std::uint32_t i = taken; i < whole && i < count; ++i) outcome.leftover[outcome.leftover_count++] = run[i];
}

// Sends `quadwords` from `from`; the quadwords the peripheral took, all of
// them unless it stalled (outcome.stalled, outcome.leftover).
std::uint32_t Send(std::uint32_t from, std::uint32_t quadwords, DmaOutcome& outcome, const DmaStallingSink& words) {
    Memory& m = GuestMemory();
    // Main RAM or the scratchpad, as DMA reads them: one pointer for the lot
    // (a word at a time through Read cost a tenth of a frame). Anything with
    // register hooks goes word by word.
    const std::uint32_t physical = from & 0x1FFFFFFFu;
    const bool plain = physical < 0x10000000u || (from >> 28) == 0x7;
    const std::uint8_t* p = plain && quadwords ? m.Translate(from, 16 * quadwords) : nullptr;
    if (p && reinterpret_cast<std::uintptr_t>(p) % 4 == 0) {
        const auto* run = reinterpret_cast<const std::uint32_t*>(p);
        const DmaTaken taken = words(run, 4 * quadwords, from);
        if (taken.stall) {
            outcome.stalled = true;
            KeepLeftover(outcome, run, 4 * quadwords, taken.words);
            const std::uint32_t sent = (taken.words + 3) / 4;
            outcome.quadwords += sent;
            return sent;
        }
    } else {
        for (std::uint32_t i = 0; i < 4 * quadwords; ++i) {
            const std::uint32_t value = std::uint32_t(m.Read(from + 4 * i, 4));
            const DmaTaken taken = words(&value, 1, from + 4 * i);
            if (taken.stall) {
                outcome.stalled = true;
                const std::uint32_t consumed = i + taken.words, whole = (consumed + 3) / 4 * 4;
                outcome.leftover_count = 0;
                for (std::uint32_t k = consumed; k < whole; ++k)
                    outcome.leftover[outcome.leftover_count++] = std::uint32_t(m.Read(from + 4 * k, 4));
                outcome.quadwords += whole / 4;
                return whole / 4;
            }
        }
    }
    outcome.quadwords += quadwords;
    return quadwords;
}

}  // namespace

DmaOutcome SendDma(DmaChannelState& ch, const std::function<void(std::uint32_t)>& word,
                   std::uint32_t max_tags) {
    return SendDmaBlocks(ch, [&word](const std::uint32_t* words, std::uint32_t count, std::uint32_t from) {
        for (std::uint32_t i = 0; i < count; ++i) {
            g_dma_source = from + 4 * i;
            word(words[i]);
        }
    }, max_tags);
}

DmaOutcome SendDmaBlocks(DmaChannelState& ch, const DmaBlockSink& word, std::uint32_t max_tags) {
    return SendDmaBlocksUntil(ch, [&word](const std::uint32_t* words, std::uint32_t count, std::uint32_t from) {
        word(words, count, from);
        return DmaTaken{count, false};
    }, max_tags);
}

DmaOutcome SendDmaBlocksUntil(DmaChannelState& ch, const DmaStallingSink& word, std::uint32_t max_tags) {
    DmaOutcome outcome;
    const std::uint32_t mode = (ch.chcr >> 2) & 3;
    // Normal mode, or a chain resumed with QWC still owed: data from MADR first.
    if (ch.qwc) {
        const std::uint32_t sent = Send(ch.madr, ch.qwc, outcome, word);
        ch.madr += 16 * sent;
        ch.qwc -= sent;
        if (outcome.stalled) return outcome;
    }
    if (mode != 1 || ch.end_pending) {
        ch.end_pending = false;
        outcome.ended = true;
        return outcome;
    }
    const bool transfer_tags = (ch.chcr >> 6) & 1, irq_ends = (ch.chcr >> 7) & 1;
    std::uint32_t depth = (ch.chcr >> 4) & 3;       // ASP: call stack depth
    Memory& m = GuestMemory();
    while (outcome.tags < max_tags) {
        const std::uint64_t tag = m.Read(ch.tadr, 8);
        ++outcome.tags;
        ch.chcr = (ch.chcr & 0xFFFFu) | (std::uint32_t(tag) & 0xFFFF0000u);     // TAG
        const std::uint32_t tag_address = ch.tadr;
        const std::uint32_t qwc = std::uint32_t(tag) & 0xFFFFu;
        const std::uint32_t id = std::uint32_t(tag >> 28) & 7;
        const std::uint32_t after = ch.tadr + 16;
        bool end = false;
        std::uint32_t data = after;
        switch (id) {
            case 0: data = Address(tag); end = true; break;                         // refe
            case 1: ch.tadr = after + 16 * qwc; break;                              // cnt
            case 2: ch.tadr = Address(tag); break;                                  // next
            case 3: case 4: data = Address(tag); ch.tadr = after; break;            // ref, refs
            case 5:                                                                 // call
                if (depth == 0) ch.asr0 = after + 16 * qwc;
                else ch.asr1 = after + 16 * qwc;
                if (depth < 2) ++depth;
                ch.tadr = Address(tag);
                break;
            case 6:                                                                 // ret
                if (depth == 0) end = true;
                else ch.tadr = (--depth == 0) ? ch.asr0 : ch.asr1;
                break;
            case 7: end = true; break;                                              // end
        }
        end = end || (irq_ends && (tag >> 31) & 1);
        if (transfer_tags) {
            const std::uint32_t high[2] = {std::uint32_t(m.Read(tag_address + 8, 4)),
                                           std::uint32_t(m.Read(tag_address + 12, 4))};
            const DmaTaken taken = word(high, 2, tag_address + 8);
            if (taken.stall) {                      // in the tag's VIFcodes: its data all owed
                outcome.stalled = true;
                outcome.leftover_count = 0;
                for (std::uint32_t i = taken.words; i < 2; ++i) outcome.leftover[outcome.leftover_count++] = high[i];
                ch.madr = data;
                ch.qwc = qwc;
                ch.end_pending = end;
                break;
            }
        }
        const std::uint32_t sent = Send(data, qwc, outcome, word);
        ch.madr = data + 16 * sent;
        if (outcome.stalled) {
            ch.qwc = qwc - sent;
            ch.end_pending = end;
            break;
        }
        if (end) {
            outcome.ended = true;
            break;
        }
    }
    ch.chcr = (ch.chcr & ~0x30u) | (depth << 4);
    return outcome;
}

}  // namespace ps2host
