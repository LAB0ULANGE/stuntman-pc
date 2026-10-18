/* The renderer's Vulkan commands recorded on a thread of their own
   (VulkanGpu::SetRecordThread, the "vulkan_thread" setting).

   The renderer decides what to record on its own thread - batches, layouts,
   the pass left open - and records as before; each vkCmd* it calls on one of
   its command buffers is queued here instead, its arguments copied (what
   they point to too), and a thread makes the call into the driver: a third
   of the GS thread's time in Stuntman's stunt. Beginning, ending and
   submitting a command buffer are queued the same way, and VulkanGpu::Flush
   waits for the queue to be run: what reaches the GPU, and in what order,
   is what it was. Other command buffers (the display's) are recorded as
   they are called.

   Included in namespace ps2gpu by gpu_internal.h: these vkCmd* hide volk's
   there. */

#ifndef PS2GPU_COMMAND_STREAM_H
#define PS2GPU_COMMAND_STREAM_H

class CommandStream {
public:
    explicit CommandStream(const VkCommandBuffer* owned, int count) {
        for (int i = 0; i < count && i < 4; ++i) owned_[i] = owned[i];
        write_ = read_ = NewBlock();
        // PS2GPU_STREAM_DELAY_US (tests): the thread that slow, at each run of
        // calls - what waits for it must wait.
        if (const char* delay = std::getenv("PS2GPU_STREAM_DELAY_US")) delay_us_ = std::atoi(delay);
        thread_ = std::thread([this] { Run(); });
    }
    ~CommandStream() {
        Drain();
        {
            std::lock_guard<std::mutex> hold(lock_);
            stop_ = true;
            wake_.notify_one();
        }
        thread_.join();
        delete read_;
        for (Block* b : free_) delete b;
    }
    bool Owns(VkCommandBuffer c) const { return c == owned_[0] || c == owned_[1] || c == owned_[2] || c == owned_[3]; }

    /* `call(payload)` on the thread, in turn; `bytes` of `data` copied to
       its payload first. */
    template <typename F>
    void Add(F&& call, const void* data = nullptr, std::size_t bytes = 0) {
        using Call = std::decay_t<F>;
        static_assert(std::is_trivially_destructible_v<Call>, "a queued call keeps nothing to free");
        static_assert(alignof(Call) <= 16, "a queued call's alignment");
        const std::size_t call_at = Round(sizeof(Entry)), payload_at = call_at + Round(sizeof(Call));
        const std::size_t size = payload_at + Round(bytes);
        if (write_at_ + size > kBlock) {
            Block* next = NewBlock();
            write_->next.store(next, std::memory_order_release);
            write_ = next;
            write_at_ = 0;
        }
        unsigned char* at = write_->data + write_at_;
        Entry* e = new (at) Entry{&Invoke<Call>, std::uint32_t(size), std::uint32_t(payload_at)};
        (void)e;
        new (at + call_at) Call(std::forward<F>(call));
        if (bytes) std::memcpy(at + payload_at, data, bytes);
        write_at_ += size;
        write_->filled.store(write_at_, std::memory_order_release);
        added_.fetch_add(1, std::memory_order_seq_cst);
        if (idle_.load(std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> hold(lock_);
            wake_.notify_one();
        }
    }
    /* Waits until everything added has run. */
    void Drain() {
        const std::uint64_t target = added_.load(std::memory_order_seq_cst);
        for (int spin = 0; spin < 4000; ++spin) {
            if (done_.load(std::memory_order_acquire) >= target) return;
            _mm_pause();
        }
        std::unique_lock<std::mutex> hold(lock_);
        drained_.wait(hold, [&] { return done_.load(std::memory_order_acquire) >= target; });
    }
    /* The thread's time running calls (RecordThreadBusy). */
    std::uint64_t busy_ns() const { return busy_ns_.load(std::memory_order_relaxed); }

private:
    struct Entry {
        void (*run)(void* call, void* payload);
        std::uint32_t size, payload;
    };
    static constexpr std::size_t kBlock = std::size_t(1) << 20;
    struct Block {
        std::atomic<std::size_t> filled{0};
        std::atomic<Block*> next{nullptr};
        alignas(16) unsigned char data[kBlock];
    };
    static std::size_t Round(std::size_t n) { return (n + 15) & ~std::size_t(15); }
    template <typename Call>
    static void Invoke(void* call, void* payload) { (*static_cast<Call*>(call))(payload); }
    Block* NewBlock() {
        {
            std::lock_guard<std::mutex> hold(free_lock_);
            if (!free_.empty()) {
                Block* b = free_.back();
                free_.pop_back();
                b->filled.store(0, std::memory_order_relaxed);
                b->next.store(nullptr, std::memory_order_relaxed);
                return b;
            }
        }
        return new Block;
    }
    void Run() {
        for (;;) {
            const std::size_t filled = read_->filled.load(std::memory_order_acquire);
            if (read_at_ < filled) {
                if (delay_us_ > 0) std::this_thread::sleep_for(std::chrono::microseconds(delay_us_));
                const auto began = std::chrono::steady_clock::now();
                std::uint64_t ran = 0;
                while (read_at_ < filled) {
                    unsigned char* at = read_->data + read_at_;
                    Entry* e = reinterpret_cast<Entry*>(at);
                    e->run(at + Round(sizeof(Entry)), at + e->payload);
                    read_at_ += e->size;
                    ++ran;
                }
                busy_ns_.fetch_add(std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - began).count()),
                                   std::memory_order_relaxed);
                done_.fetch_add(ran, std::memory_order_seq_cst);
                continue;
            }
            Block* next = read_->next.load(std::memory_order_acquire);
            if (next && read_at_ >= read_->filled.load(std::memory_order_acquire)) {
                {
                    std::lock_guard<std::mutex> hold(free_lock_);
                    free_.push_back(read_);
                }
                read_ = next;
                read_at_ = 0;
                continue;
            }
            // Nothing queued: a moment's spin (the next call comes soon), then asleep.
            bool more = false;
            for (int spin = 0; spin < 2000 && !more; ++spin) {
                _mm_pause();
                more = read_->filled.load(std::memory_order_acquire) > read_at_ ||
                       read_->next.load(std::memory_order_acquire) != nullptr;
            }
            if (more) continue;
            std::unique_lock<std::mutex> hold(lock_);
            idle_.store(true, std::memory_order_seq_cst);
            drained_.notify_all();
            wake_.wait(hold, [&] {
                return stop_ || added_.load(std::memory_order_seq_cst) > done_.load(std::memory_order_seq_cst);
            });
            idle_.store(false, std::memory_order_seq_cst);
            if (stop_ && added_.load() == done_.load()) return;
        }
    }

    VkCommandBuffer owned_[4] = {};
    Block* write_ = nullptr;        // the renderer's thread's
    std::size_t write_at_ = 0;
    Block* read_ = nullptr;         // the recording thread's
    std::size_t read_at_ = 0;
    std::mutex free_lock_;
    std::vector<Block*> free_;
    std::atomic<std::uint64_t> added_{0}, done_{0}, busy_ns_{0};
    std::atomic<bool> idle_{false};
    bool stop_ = false;
    int delay_us_ = 0;
    std::mutex lock_;
    std::condition_variable wake_, drained_;
    std::thread thread_;
};

/* The renderer's stream while it records on its thread (one renderer). */
inline CommandStream* g_stream = nullptr;

inline CommandStream* StreamFor(VkCommandBuffer c) { return g_stream && g_stream->Owns(c) ? g_stream : nullptr; }

// The commands the renderer uses, queued where its stream owns the command
// buffer, else made at once. Pointed-to arguments go in the payload.

inline void vkCmdBindPipeline(VkCommandBuffer c, VkPipelineBindPoint point, VkPipeline p) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdBindPipeline(c, point, p); });
    ::vkCmdBindPipeline(c, point, p);
}
inline void vkCmdBindDescriptorSets(VkCommandBuffer c, VkPipelineBindPoint point, VkPipelineLayout layout,
                                    std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets,
                                    std::uint32_t dynamic_count, const std::uint32_t* dynamic_offsets) {
    if (CommandStream* s = StreamFor(c)) {
        unsigned char data[16 * sizeof(VkDescriptorSet) + 16 * sizeof(std::uint32_t)];
        const std::size_t a = count * sizeof(VkDescriptorSet), b = dynamic_count * sizeof(std::uint32_t);
        if (count <= 16 && dynamic_count <= 16) {
            std::memcpy(data, sets, a);
            if (b) std::memcpy(data + a, dynamic_offsets, b);
            return s->Add(
                [=](void* p) {
                    ::vkCmdBindDescriptorSets(c, point, layout, first, count, static_cast<VkDescriptorSet*>(p),
                                              dynamic_count,
                                              reinterpret_cast<std::uint32_t*>(static_cast<unsigned char*>(p) + a));
                },
                data, a + b);
        }
        s->Drain();
    }
    ::vkCmdBindDescriptorSets(c, point, layout, first, count, sets, dynamic_count, dynamic_offsets);
}
inline void vkCmdPushConstants(VkCommandBuffer c, VkPipelineLayout layout, VkShaderStageFlags stages,
                               std::uint32_t offset, std::uint32_t size, const void* values) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void* p) { ::vkCmdPushConstants(c, layout, stages, offset, size, p); }, values, size);
    ::vkCmdPushConstants(c, layout, stages, offset, size, values);
}
inline void vkCmdSetViewport(VkCommandBuffer c, std::uint32_t first, std::uint32_t count, const VkViewport* v) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void* p) { ::vkCmdSetViewport(c, first, count, static_cast<VkViewport*>(p)); }, v,
                      count * sizeof(VkViewport));
    ::vkCmdSetViewport(c, first, count, v);
}
inline void vkCmdSetScissor(VkCommandBuffer c, std::uint32_t first, std::uint32_t count, const VkRect2D* r) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void* p) { ::vkCmdSetScissor(c, first, count, static_cast<VkRect2D*>(p)); }, r,
                      count * sizeof(VkRect2D));
    ::vkCmdSetScissor(c, first, count, r);
}
inline void vkCmdDraw(VkCommandBuffer c, std::uint32_t vertices, std::uint32_t instances, std::uint32_t first,
                      std::uint32_t first_instance) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void*) { ::vkCmdDraw(c, vertices, instances, first, first_instance); });
    ::vkCmdDraw(c, vertices, instances, first, first_instance);
}
inline void vkCmdDispatch(VkCommandBuffer c, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdDispatch(c, x, y, z); });
    ::vkCmdDispatch(c, x, y, z);
}
inline void vkCmdBindVertexBuffers(VkCommandBuffer c, std::uint32_t first, std::uint32_t count,
                                   const VkBuffer* buffers, const VkDeviceSize* offsets) {
    if (CommandStream* s = StreamFor(c)) {
        if (count <= 8) {
            unsigned char data[8 * sizeof(VkBuffer) + 8 * sizeof(VkDeviceSize)];
            const std::size_t a = count * sizeof(VkBuffer);
            std::memcpy(data, buffers, a);
            std::memcpy(data + a, offsets, count * sizeof(VkDeviceSize));
            return s->Add(
                [=](void* p) {
                    ::vkCmdBindVertexBuffers(c, first, count, static_cast<VkBuffer*>(p),
                                             reinterpret_cast<VkDeviceSize*>(static_cast<unsigned char*>(p) + a));
                },
                data, a + count * sizeof(VkDeviceSize));
        }
        s->Drain();
    }
    ::vkCmdBindVertexBuffers(c, first, count, buffers, offsets);
}
inline void vkCmdSetDepthTestEnable(VkCommandBuffer c, VkBool32 on) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdSetDepthTestEnable(c, on); });
    ::vkCmdSetDepthTestEnable(c, on);
}
inline void vkCmdSetDepthWriteEnable(VkCommandBuffer c, VkBool32 on) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdSetDepthWriteEnable(c, on); });
    ::vkCmdSetDepthWriteEnable(c, on);
}
inline void vkCmdSetDepthCompareOp(VkCommandBuffer c, VkCompareOp op) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdSetDepthCompareOp(c, op); });
    ::vkCmdSetDepthCompareOp(c, op);
}
inline void vkCmdSetStencilTestEnable(VkCommandBuffer c, VkBool32 on) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdSetStencilTestEnable(c, on); });
    ::vkCmdSetStencilTestEnable(c, on);
}
inline void vkCmdSetStencilReference(VkCommandBuffer c, VkStencilFaceFlags faces, std::uint32_t value) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void*) { ::vkCmdSetStencilReference(c, faces, value); });
    ::vkCmdSetStencilReference(c, faces, value);
}
inline void vkCmdEndRendering(VkCommandBuffer c) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdEndRendering(c); });
    ::vkCmdEndRendering(c);
}
inline void vkCmdBeginRendering(VkCommandBuffer c, const VkRenderingInfo* info) {
    if (CommandStream* s = StreamFor(c)) {
        // The info and its attachments, the pointers made again on the thread.
        struct Copy {
            VkRenderingInfo info;
            VkRenderingAttachmentInfo colours[4], depth, stencil;
        } copy{};
        const bool plain = !info->pNext && info->colorAttachmentCount <= 4;
        if (plain) {
            copy.info = *info;
            for (std::uint32_t i = 0; i < info->colorAttachmentCount; ++i) copy.colours[i] = info->pColorAttachments[i];
            if (info->pDepthAttachment) copy.depth = *info->pDepthAttachment;
            if (info->pStencilAttachment) copy.stencil = *info->pStencilAttachment;
            const bool depth = info->pDepthAttachment != nullptr, stencil = info->pStencilAttachment != nullptr;
            return s->Add(
                [=](void* p) {
                    Copy& k = *static_cast<Copy*>(p);
                    k.info.pColorAttachments = k.info.colorAttachmentCount ? k.colours : nullptr;
                    k.info.pDepthAttachment = depth ? &k.depth : nullptr;
                    k.info.pStencilAttachment = stencil ? &k.stencil : nullptr;
                    ::vkCmdBeginRendering(c, &k.info);
                },
                &copy, sizeof copy);
        }
        s->Drain();
    }
    ::vkCmdBeginRendering(c, info);
}
inline void vkCmdPipelineBarrier(VkCommandBuffer c, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                 VkDependencyFlags flags, std::uint32_t memory_count, const VkMemoryBarrier* memory,
                                 std::uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                 std::uint32_t image_count, const VkImageMemoryBarrier* images) {
    if (CommandStream* s = StreamFor(c)) {
        struct Copy {
            VkMemoryBarrier memory[4];
            VkBufferMemoryBarrier buffers[4];
            VkImageMemoryBarrier images[8];
        } copy{};
        if (memory_count <= 4 && buffer_count <= 4 && image_count <= 8) {
            for (std::uint32_t i = 0; i < memory_count; ++i) copy.memory[i] = memory[i];
            for (std::uint32_t i = 0; i < buffer_count; ++i) copy.buffers[i] = buffers[i];
            for (std::uint32_t i = 0; i < image_count; ++i) copy.images[i] = images[i];
            return s->Add(
                [=](void* p) {
                    const Copy& k = *static_cast<Copy*>(p);
                    ::vkCmdPipelineBarrier(c, src, dst, flags, memory_count, k.memory, buffer_count, k.buffers,
                                           image_count, k.images);
                },
                &copy, sizeof copy);
        }
        s->Drain();
    }
    ::vkCmdPipelineBarrier(c, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count, images);
}
inline void vkCmdCopyBufferToImage(VkCommandBuffer c, VkBuffer buffer, VkImage image, VkImageLayout layout,
                                   std::uint32_t count, const VkBufferImageCopy* regions) {
    if (CommandStream* s = StreamFor(c))
        return s->Add(
            [=](void* p) {
                ::vkCmdCopyBufferToImage(c, buffer, image, layout, count, static_cast<VkBufferImageCopy*>(p));
            },
            regions, count * sizeof(VkBufferImageCopy));
    ::vkCmdCopyBufferToImage(c, buffer, image, layout, count, regions);
}
inline void vkCmdCopyImageToBuffer(VkCommandBuffer c, VkImage image, VkImageLayout layout, VkBuffer buffer,
                                   std::uint32_t count, const VkBufferImageCopy* regions) {
    if (CommandStream* s = StreamFor(c))
        return s->Add(
            [=](void* p) {
                ::vkCmdCopyImageToBuffer(c, image, layout, buffer, count, static_cast<VkBufferImageCopy*>(p));
            },
            regions, count * sizeof(VkBufferImageCopy));
    ::vkCmdCopyImageToBuffer(c, image, layout, buffer, count, regions);
}
inline void vkCmdCopyImage(VkCommandBuffer c, VkImage src, VkImageLayout src_layout, VkImage dst,
                           VkImageLayout dst_layout, std::uint32_t count, const VkImageCopy* regions) {
    if (CommandStream* s = StreamFor(c))
        return s->Add(
            [=](void* p) {
                ::vkCmdCopyImage(c, src, src_layout, dst, dst_layout, count, static_cast<VkImageCopy*>(p));
            },
            regions, count * sizeof(VkImageCopy));
    ::vkCmdCopyImage(c, src, src_layout, dst, dst_layout, count, regions);
}
inline void vkCmdCopyBuffer(VkCommandBuffer c, VkBuffer src, VkBuffer dst, std::uint32_t count,
                            const VkBufferCopy* regions) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void* p) { ::vkCmdCopyBuffer(c, src, dst, count, static_cast<VkBufferCopy*>(p)); },
                      regions, count * sizeof(VkBufferCopy));
    ::vkCmdCopyBuffer(c, src, dst, count, regions);
}
inline void vkCmdUpdateBuffer(VkCommandBuffer c, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                              const void* data) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void* p) { ::vkCmdUpdateBuffer(c, buffer, offset, size, p); }, data, std::size_t(size));
    ::vkCmdUpdateBuffer(c, buffer, offset, size, data);
}
inline void vkCmdFillBuffer(VkCommandBuffer c, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                            std::uint32_t value) {
    if (CommandStream* s = StreamFor(c))
        return s->Add([=](void*) { ::vkCmdFillBuffer(c, buffer, offset, size, value); });
    ::vkCmdFillBuffer(c, buffer, offset, size, value);
}
inline void vkCmdClearColorImage(VkCommandBuffer c, VkImage image, VkImageLayout layout,
                                 const VkClearColorValue* colour, std::uint32_t count,
                                 const VkImageSubresourceRange* ranges) {
    if (CommandStream* s = StreamFor(c)) {
        if (count <= 4) {
            struct Copy {
                VkClearColorValue colour;
                VkImageSubresourceRange ranges[4];
            } copy{};
            copy.colour = *colour;
            for (std::uint32_t i = 0; i < count; ++i) copy.ranges[i] = ranges[i];
            return s->Add(
                [=](void* p) {
                    const Copy& k = *static_cast<Copy*>(p);
                    ::vkCmdClearColorImage(c, image, layout, &k.colour, count, k.ranges);
                },
                &copy, sizeof copy);
        }
        s->Drain();
    }
    ::vkCmdClearColorImage(c, image, layout, colour, count, ranges);
}
inline void vkCmdClearDepthStencilImage(VkCommandBuffer c, VkImage image, VkImageLayout layout,
                                        const VkClearDepthStencilValue* value, std::uint32_t count,
                                        const VkImageSubresourceRange* ranges) {
    if (CommandStream* s = StreamFor(c)) {
        if (count <= 4) {
            struct Copy {
                VkClearDepthStencilValue value;
                VkImageSubresourceRange ranges[4];
            } copy{};
            copy.value = *value;
            for (std::uint32_t i = 0; i < count; ++i) copy.ranges[i] = ranges[i];
            return s->Add(
                [=](void* p) {
                    const Copy& k = *static_cast<Copy*>(p);
                    ::vkCmdClearDepthStencilImage(c, image, layout, &k.value, count, k.ranges);
                },
                &copy, sizeof copy);
        }
        s->Drain();
    }
    ::vkCmdClearDepthStencilImage(c, image, layout, value, count, ranges);
}
inline void vkCmdWriteTimestamp(VkCommandBuffer c, VkPipelineStageFlagBits stage, VkQueryPool pool,
                                std::uint32_t query) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdWriteTimestamp(c, stage, pool, query); });
    ::vkCmdWriteTimestamp(c, stage, pool, query);
}
inline void vkCmdResetQueryPool(VkCommandBuffer c, VkQueryPool pool, std::uint32_t first, std::uint32_t count) {
    if (CommandStream* s = StreamFor(c)) return s->Add([=](void*) { ::vkCmdResetQueryPool(c, pool, first, count); });
    ::vkCmdResetQueryPool(c, pool, first, count);
}

#endif  /* PS2GPU_COMMAND_STREAM_H */
