/*
 * PATH3 held back by VIF1's MSKPATH3 (devices.cpp).
 *
 * GIF DMA data waits here while PATH3 is masked. A mask takes effect only at
 * the end of the GIF packet PATH3 is in - the one whose tag has EOP set - so
 * unmasking and masking again at once lets exactly one EOP-ended group of
 * packets through: Release(false). Release(true) lets all of it go (time
 * passing with PATH3 unmasked, or FLUSHA waiting for it).
 */

#ifndef PS2HOST_PATH3_H
#define PS2HOST_PATH3_H

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ps2host {

class Path3Queue {
public:
    void Push(std::uint32_t word) { held_.push_back(word); }
    void Push(const std::uint32_t* words, std::size_t count) { held_.insert(held_.end(), words, words + count); }
    bool empty() const { return size() == 0; }
    std::size_t size() const { return held_.size() - head_; }

    /* Hands words to `sink` (one run: words, count), up to the end of the
       first packet with EOP set (or all of them). */
    template <typename Sink>
    void ReleaseBlock(bool all, Sink&& sink) {
        const std::uint32_t* words = held_.data() + head_;
        const std::size_t held = size();
        std::size_t end = held;
        if (!all) {
            std::size_t at = 0;
            while (at + 4 <= held) {
                const std::uint64_t low = std::uint64_t(words[at]) | std::uint64_t(words[at + 1]) << 32;
                const std::uint64_t nloop = low & 0x7FFF, eop = (low >> 15) & 1;
                const std::uint64_t flg = (low >> 58) & 3, nreg = (low >> 60) ? (low >> 60) : 16;
                const std::uint64_t qwords = flg == 0 ? nloop * nreg : flg == 1 ? (nloop * nreg + 1) / 2 : nloop;
                at += 4 + std::size_t(qwords) * 4;
                if (eop) break;
            }
            end = std::min(at, held);
        }
        // Taken off before the sink runs: what it does may hold more. Into a
        // buffer kept for it, and what is left moved down only once a good
        // deal has gone (Burnout 3 holds a picture's textures here, 160,000
        // words, and lets them go a group at a time).
        // (A release within a release - the sink letting more go - takes a
        // buffer of its own: the outer one's sink is still reading going_.)
        std::vector<std::uint32_t> nested;
        std::vector<std::uint32_t>& buffer = releasing_ ? nested : going_;
        buffer.assign(words, words + end);
        head_ += end;
        if (head_ == held_.size()) {
            held_.clear();
            head_ = 0;
        } else if (head_ > held_.size() / 2 && head_ >= 65536) {
            held_.erase(held_.begin(), held_.begin() + std::ptrdiff_t(head_));
            head_ = 0;
        }
        if (buffer.empty()) return;
        if (&buffer == &nested) {
            sink(nested.data(), nested.size());
            return;
        }
        releasing_ = true;
        struct Done { bool& flag; ~Done() { flag = false; } } done{releasing_};
        sink(going_.data(), going_.size());
    }

    /* The same, a word at a time. */
    template <typename Sink>
    void Release(bool all, Sink&& sink) {
        ReleaseBlock(all, [&](const std::uint32_t* words, std::size_t count) {
            for (std::size_t i = 0; i < count; ++i) sink(words[i]);
        });
    }

private:
    std::vector<std::uint32_t> held_;
    std::size_t head_ = 0;                  // words before it have gone
    std::vector<std::uint32_t> going_;      // what ReleaseBlock hands on
    bool releasing_ = false;
};

}  // namespace ps2host

#endif  /* PS2HOST_PATH3_H */
