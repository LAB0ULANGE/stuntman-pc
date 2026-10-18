/*
 * The GS renderer on a thread of its own (CreateThreadedRenderer, the
 * "render_thread" setting): the GPU bridge's calls into it are queued, their
 * arguments copied (what they point to too), and a thread makes them in
 * order - the renderer's batching, pipelines and recording, a third of the
 * GS thread's time in Stuntman's stunt, beside the GS model and the bridge
 * rather than after them.
 *
 * A call whose answer the bridge needs waits for the queue to be run, then
 * is made at once - reading targets back, the picture shown (once a frame) -
 * but for the ones made all the time: TargetToMemory and DepthToMemory
 * (hundreds a picture: the bridge's write-backs) answer whether the image
 * exists, and that is known here from what was asked for before - a draw of
 * three vertices or more makes its target and depth buffer, as the loads do
 * (the renderer finds an image a pending batch makes: TargetToMemory). In
 * "check" they wait and are made at once too, and an answer that differs
 * from the one worked out is counted and said.
 */

#include "gpu_internal.h"

#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>

namespace ps2gpu {

namespace {

class ThreadedRenderer final : public GsRenderer {
public:
    ThreadedRenderer(std::unique_ptr<GsRenderer> inner, bool check)
        : inner_(std::move(inner)), check_(check), queue_(nullptr, 0), scale_(inner_->Scale()) {}
    ~ThreadedRenderer() override {
        if (inner_) queue_.Drain();
    }
    std::unique_ptr<GsRenderer> TakeInner() override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return std::move(inner_);
    }

    // ---- queued
    void SetScale(std::uint32_t scale) override {
        std::lock_guard<std::mutex> hold(lock_);
        scale = std::min(std::max(1u, scale), 4u);      // as the renderer keeps it (kMaxScale)
        if (scale != scale_) {
            targets_.clear();
            depths_.clear();
        }
        scale_ = scale;
        Queue([r = inner_.get(), scale](void*) { r->SetScale(scale); });
    }
    std::uint32_t Scale() const override {
        std::lock_guard<std::mutex> hold(lock_);
        return scale_;
    }
    void UploadTexture(std::uint64_t key, const std::uint32_t* pixels, std::uint32_t width,
                       std::uint32_t height) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), key, width, height](void* p) {
                  r->UploadTexture(key, static_cast<const std::uint32_t*>(p), width, height);
              },
              pixels, std::size_t(width) * height * 4,
              [&] { inner_->UploadTexture(key, pixels, width, height); });
    }
    void UseReplacement(std::uint64_t key, std::uint64_t id) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), key, id](void*) { r->UseReplacement(key, id); });
    }
    void UploadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, const std::uint32_t* z) override {
        std::lock_guard<std::mutex> hold(lock_);
        depths_.insert(Key{zbp, width, height});
        Queue([r = inner_.get(), zbp, width, height](void* p) {
                  r->UploadDepth(zbp, width, height, static_cast<const std::uint32_t*>(p));
              },
              z, std::size_t(width) * height * 4, [&] { inner_->UploadDepth(zbp, width, height, z); });
    }
    void UploadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, const std::uint32_t* pixels,
                      std::uint32_t width, std::uint32_t height) override {
        std::lock_guard<std::mutex> hold(lock_);
        Made(fbp, fbw, psm, width, height);
        Queue([r = inner_.get(), fbp, fbw, psm, width, height](void* p) {
                  r->UploadTarget(fbp, fbw, psm, static_cast<const std::uint32_t*>(p), width, height);
              },
              pixels, std::size_t(width) * height * 4,
              [&] { inner_->UploadTarget(fbp, fbw, psm, pixels, width, height); });
    }
    void ReleaseTexture(std::uint64_t key) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), key](void*) { r->ReleaseTexture(key); });
    }
    void DrawTriangles(const GsDrawState& state, const GsVertex* vertices, std::size_t count) override {
        std::lock_guard<std::mutex> hold(lock_);
        if (count >= 3) {
            // Its target, and its depth buffer as the renderer's Flush picks it.
            const auto& size = Made(state.fbp, state.fbw, state.psm, state.width, state.height);
            depths_.insert(state.depth ? Key{state.zbp, size.first, std::min(size.second, kDepthRows)}
                                       : Key{0x10000u + state.fbp, size.first, size.second});
        }
        Queue([r = inner_.get(), state, count](void* p) {
                  r->DrawTriangles(state, static_cast<const GsVertex*>(p), count);
              },
              vertices, count * sizeof(GsVertex), [&] { inner_->DrawTriangles(state, vertices, count); });
    }
    void Flush() override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get()](void*) { r->Flush(); });
    }
    void Kick() override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get()](void*) { r->Kick(); });
        if (check_ && ++kicks_ % 600 == 0)
            std::fprintf(stderr, "[ps2gpu] render thread check: %llu answers compared, %llu differed%c",
                         (unsigned long long)answers_, (unsigned long long)differ_, 10);
    }
    void MemoryWrite(std::uint32_t begin, std::uint32_t end, const std::uint8_t* bytes) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), begin, end](void* p) {
                  r->MemoryWrite(begin, end, static_cast<const std::uint8_t*>(p));
              },
              bytes, std::size_t(end - begin) * 256, [&] { inner_->MemoryWrite(begin, end, bytes); });
    }
    void MemoryPatch(const std::uint32_t* words, std::size_t count) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), count](void* p) { r->MemoryPatch(static_cast<const std::uint32_t*>(p), count); },
              words, count * 4, [&] { inner_->MemoryPatch(words, count); });
    }
    void TargetFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                          std::uint32_t height) override {
        std::lock_guard<std::mutex> hold(lock_);
        Made(fbp, fbw, psm, width, height);
        Queue([r = inner_.get(), fbp, fbw, psm, width, height](void*) {
            r->TargetFromMemory(fbp, fbw, psm, width, height);
        });
    }
    void DepthFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                         std::uint32_t zpsm) override {
        std::lock_guard<std::mutex> hold(lock_);
        depths_.insert(Key{zbp, width, height});
        Queue([r = inner_.get(), zbp, width, height, fbw, zpsm](void*) {
            r->DepthFromMemory(zbp, width, height, fbw, zpsm);
        });
    }
    void TargetRowsFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                              std::uint32_t height, std::uint32_t first_row, std::uint32_t rows) override {
        std::lock_guard<std::mutex> hold(lock_);
        Made(fbp, fbw, psm, width, height);
        Queue([r = inner_.get(), fbp, fbw, psm, width, height, first_row, rows](void*) {
            r->TargetRowsFromMemory(fbp, fbw, psm, width, height, first_row, rows);
        });
    }
    void DepthRowsFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                             std::uint32_t zpsm, std::uint32_t first_row, std::uint32_t rows) override {
        std::lock_guard<std::mutex> hold(lock_);
        depths_.insert(Key{zbp, width, height});
        Queue([r = inner_.get(), zbp, width, height, fbw, zpsm, first_row, rows](void*) {
            r->DepthRowsFromMemory(zbp, width, height, fbw, zpsm, first_row, rows);
        });
    }
    void TextureFromMemory(std::uint64_t key, const GsTextureDecode& decode) override {
        std::lock_guard<std::mutex> hold(lock_);
        // The CLUT it points at, as it is now (the model's goes on changing).
        Queue([r = inner_.get(), key, decode](void* p) {
                  GsTextureDecode d = decode;
                  if (d.clut) d.clut = static_cast<const std::uint32_t*>(p);
                  r->TextureFromMemory(key, d);
              },
              decode.clut, decode.clut ? 1024 : 0, [&] { inner_->TextureFromMemory(key, decode); });
    }
    void ClutFromMemory(const GsClutLoad& load) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), load](void*) { r->ClutFromMemory(load); });
    }
    void ClutRows(const std::uint32_t* clut, std::uint32_t rows) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), rows](void* p) { r->ClutRows(static_cast<const std::uint32_t*>(p), rows); }, clut,
              1024, [&] { inner_->ClutRows(clut, rows); });
    }
    void SetExactOverlap(bool exact) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), exact](void*) { r->SetExactOverlap(exact); });
    }
    void SetPreparedAxes(bool on) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), on](void*) { r->SetPreparedAxes(on); });
    }
    void SetDateArea(bool batch) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), batch](void*) { r->SetDateArea(batch); });
    }
    void ReloadKeepsColour(bool keep) override {
        std::lock_guard<std::mutex> hold(lock_);
        Queue([r = inner_.get(), keep](void*) { r->ReloadKeepsColour(keep); });
    }

    // ---- answered from what was asked for before (or, in "check", waited for and compared)
    bool TargetToMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t cols,
                        std::uint32_t rows, std::uint32_t first_row) override {
        std::lock_guard<std::mutex> hold(lock_);
        const bool known = targets_.count(Key{fbp, fbw, psm}) != 0;
        if (check_) {
            queue_.Drain();
            return Compare("TargetToMemory", fbp, known, inner_->TargetToMemory(fbp, fbw, psm, cols, rows, first_row));
        }
        Queue([r = inner_.get(), fbp, fbw, psm, cols, rows, first_row](void*) {
            r->TargetToMemory(fbp, fbw, psm, cols, rows, first_row);
        });
        return known;
    }
    bool DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                       std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows, std::uint32_t first_row) override {
        std::lock_guard<std::mutex> hold(lock_);
        const bool known = depths_.count(Key{zbp, width, height}) != 0;
        if (check_) {
            queue_.Drain();
            return Compare("DepthToMemory", zbp, known,
                           inner_->DepthToMemory(zbp, width, height, fbw, zpsm, cols, rows, first_row));
        }
        Queue([r = inner_.get(), zbp, width, height, fbw, zpsm, cols, rows, first_row](void*) {
            r->DepthToMemory(zbp, width, height, fbw, zpsm, cols, rows, first_row);
        });
        return known;
    }
    bool BlockCompression() const override { return inner_->BlockCompression(); }     // (the device's)

    // ---- waited for, then made at once
    bool UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width, std::uint32_t height,
                           std::uint32_t levels, const std::uint8_t* data, std::size_t bytes) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->UploadReplacement(id, format, width, height, levels, data, bytes);
    }
    bool HasReplacement(std::uint64_t id) const override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->HasReplacement(id);
    }
    bool HasTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm) const override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->HasTarget(fbp, fbw, psm);
    }
    void MemoryRead(std::uint32_t begin, std::uint32_t end, std::uint8_t* bytes) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        inner_->MemoryRead(begin, end, bytes);
    }
    bool SetRecordThread(bool on) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->SetRecordThread(on);
    }
    int SetDateInPass(int mode) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->SetDateInPass(mode);
    }
    bool DateInPassCheck(std::uint64_t* batches, std::uint64_t* differ) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->DateInPassCheck(batches, differ);
    }
    bool ReadClut(std::uint32_t* clut) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->ReadClut(clut);
    }
    bool ReadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                   std::vector<std::uint32_t>* z) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->ReadDepth(zbp, width, height, z);
    }
    bool ReadTexture(std::uint64_t key, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                     std::uint32_t* height) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->ReadTexture(key, pixels, width, height);
    }
    bool DisplayTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint64_t* image, int* layout,
                       std::uint32_t* width, std::uint32_t* height) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->DisplayTarget(fbp, fbw, psm, image, layout, width, height);
    }
    bool WeaveField(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t first_line,
                    std::uint32_t lines, std::uint32_t parity, std::uint64_t* image, int* layout,
                    std::uint32_t* width, std::uint32_t* height) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->WeaveField(fbp, fbw, psm, first_line, lines, parity, image, layout, width, height);
    }
    bool ReadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                    std::uint32_t* width, std::uint32_t* height) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->ReadTarget(fbp, fbw, psm, pixels, width, height);
    }
    bool ReadWoven(std::vector<std::uint32_t>* pixels, std::uint32_t* width, std::uint32_t* height) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->ReadWoven(pixels, width, height);
    }
    bool ReadTargetFull(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                        std::uint32_t* width, std::uint32_t* height) override {
        std::lock_guard<std::mutex> hold(lock_);
        queue_.Drain();
        return inner_->ReadTargetFull(fbp, fbw, psm, pixels, width, height);
    }
    bool RenderThreadCheck(std::uint64_t* answers, std::uint64_t* differ) override {
        std::lock_guard<std::mutex> hold(lock_);
        *answers = answers_;
        *differ = differ_;
        return check_;
    }

private:
    using Key = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    static constexpr std::uint32_t kDepthRows = 512;         // as gs_renderer.cpp's
    // Payloads above this are not copied: the queue run, the call made at once.
    static constexpr std::size_t kMaxPayload = std::size_t(512) << 10;

    template <typename F>
    void Queue(F&& call) {
        queue_.Add(std::forward<F>(call));
    }
    template <typename F, typename G>
    void Queue(F&& call, const void* data, std::size_t bytes, G&& at_once) {
        if (bytes > kMaxPayload) {
            queue_.Drain();
            at_once();
            return;
        }
        queue_.Add(std::forward<F>(call), data, bytes);
    }
    /* A target the renderer makes (or has): its size as first made. */
    const std::pair<std::uint32_t, std::uint32_t>& Made(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                                         std::uint32_t width, std::uint32_t height) {
        return targets_.emplace(Key{fbp, fbw, psm}, std::make_pair(width, height)).first->second;
    }
    bool Compare(const char* what, std::uint32_t bp, bool known, bool actual) {
        ++answers_;
        if (known != actual && differ_++ < 20)
            std::fprintf(stderr, "[ps2gpu] render thread check: %s %X answered %d, worked out %d\n", what,
                         unsigned(bp), int(actual), int(known));
        return actual;
    }

    std::unique_ptr<GsRenderer> inner_;
    const bool check_;
    mutable std::mutex lock_;               // one caller at a time: the queue has one producer
    mutable CommandStream queue_;
    std::uint32_t scale_ = 1;
    std::map<Key, std::pair<std::uint32_t, std::uint32_t>> targets_;
    std::set<Key> depths_;
    std::uint64_t answers_ = 0, differ_ = 0, kicks_ = 0;
};

}  // namespace

std::unique_ptr<GsRenderer> CreateThreadedRenderer(std::unique_ptr<GsRenderer> inner, bool check) {
    return std::make_unique<ThreadedRenderer>(std::move(inner), check);
}

}  // namespace ps2gpu
