/* The Vulkan objects behind ps2gpu::Gpu, for the library's own sources. */

#ifndef PS2GPU_GPU_INTERNAL_H
#define PS2GPU_GPU_INTERNAL_H

#include <volk.h>

#include <immintrin.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "ps2gpu/gs_renderer.h"

namespace ps2gpu {

/* SubmitStats: submissions, the time spent waiting for the GPU, and what
   waited (a read-back of which kind, a resize, a full ring). */
inline std::uint64_t g_submits = 0;
/* The one queue the renderer and the display share: Vulkan wants its
   submissions, presents and the device's wait-idle from one thread at a
   time, and the display presents from a thread of its own. */
inline std::mutex g_queue_lock;
inline double g_wait_seconds = 0;
inline GpuWaitObserver g_wait_observer = nullptr;      // SetGpuWaitObserver
/* The GPU's time on the command buffers handed over, in all (timestamps at
   each one's start and end; GpuBusyNanoseconds). */
inline std::atomic<std::uint64_t> g_gpu_ns{0};
/* PS2HOST_GPU_PROFILE: the GPU's time by what it did (GpuProfileReport). */
enum ProfileKind { kProfOther, kProfDraw, kProfSwizzle, kProfLinear, kProfMemory, kProfFeedback, kProfDecode,
                   kProfUpload, kProfClear, kProfDate, kProfReadback, kProfKinds };
inline std::atomic<std::uint64_t> g_profile_ns[kProfKinds] = {};
struct WaitReason { const char* why; std::uint64_t count; double seconds; };
inline WaitReason g_waits[16] = {};
/* Whether the device can read a target being drawn, in its own pass
   (VK_KHR_dynamic_rendering_local_read, and its feature). */
inline bool HasLocalRead(VkPhysicalDevice gpu) {
    std::uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> list(n);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n, list.data());
    bool found = false;
    for (const VkExtensionProperties& e : list)
        if (std::string(e.extensionName) == VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME) found = true;
    if (!found) return false;
    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR local{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &local;
    vkGetPhysicalDeviceFeatures2(gpu, &features);
    return local.dynamicRenderingLocalRead == VK_TRUE;
}
inline void CountWait(const char* why, double seconds) {
    for (WaitReason& w : g_waits) {
        if (!w.why) w.why = why;
        if (w.why == why || std::string(w.why) == why) {
            ++w.count;
            w.seconds += seconds;
            return;
        }
    }
}

#include "command_stream.h"

class VulkanGpu final : public Gpu {
public:
    ~VulkanGpu() override;
    bool Init(std::string* error);
    /* On a device someone else made and keeps (a Display's): only the command
       pool, command buffers and fences are this one's own. */
    bool InitShared(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, std::uint32_t family, VkQueue queue,
                    bool dual_source, const std::string& describe, std::string* error);
    std::string Describe() const override { return describe_; }

    std::uint32_t FindMemory(std::uint32_t bits, VkMemoryPropertyFlags wanted) const;
    /* Records into the open command buffer (opening one if none is), to run
       at the next Submit: a frame's work goes to the GPU in one submission.
       A rendering pass left open (RecordInPass) is ended first. */
    template <typename F>
    void Record(F&& record);
    /* The same, leaving a rendering pass a previous call left open: for draws
       that continue it (end_pass says how to end one). */
    template <typename F>
    void RecordInPass(F&& record);
    std::function<void(VkCommandBuffer)> end_pass;
    /* Runs what has been recorded and waits for it, and for everything
       Flush handed over before; then what was waiting for that (AfterSubmit:
       freeing staging buffers, say) runs. */
    bool Submit(const char* why = "other");
    /* Hands what has been recorded to the GPU without waiting for it, so that
       the GPU works while the CPU goes on: the next recording goes into the
       next of kSlots command buffers (waiting only if that one is still
       running). What waited on the work handed over runs once it has run. */
    bool Flush();
    void AfterSubmit(std::function<void()> done) { after_submit_.push_back(std::move(done)); }
    /* Its command buffers' commands made on a thread of their own
       (command_stream.h), or as they are recorded; what was recorded is
       submitted first. Returns whether they are. */
    bool SetRecordThread(bool on);
    /* That thread's time making them, in all. */
    std::uint64_t RecordThreadBusyNs() const { return stream_ ? stream_->busy_ns() : 0; }
    bool recording() const { return open_; }
    /* PS2HOST_GPU_PROFILE: the GPU's time since the last mark (or the command
       buffer's start) counted as `kind` - a timestamp as the work before it
       ends. */
    void Mark(VkCommandBuffer c, int kind) {
        if (!profile_ || marks_[current_].size() >= kMarks) return;
        vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profile_,
                            std::uint32_t(current_) * kMarks + std::uint32_t(marks_[current_].size()));
        marks_[current_].push_back(kind);
    }
    /* Counts command buffers begun: what was set in one is not in the next. */
    std::uint64_t recording_id() const { return recordings_; }
    static constexpr int kSlots = 3;
    /* The command buffer recording goes into (or will). */
    int slot() const { return current_; }
    /* Records with `record`, submits, and waits for it to finish. */
    template <typename F>
    bool Run(F&& record, const char* why = "other") {
        Record(std::forward<F>(record));
        return Submit(why);
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;           /* the current slot's */
    VkCommandBuffer cmds[kSlots] = {};
    VkFence fences[kSlots] = {};
    bool dual_source = false;       /* dualSrcBlend: the GS's blending */
    bool logic_op = false;          /* logicOp: FBMSK's partly masked channels */
    bool bc = false;                /* textureCompressionBC: texture packs' DDS uploaded as they are */
    bool local_read = false;        /* dynamicRenderingLocalRead: DATE set up within the draws' pass */
    bool owns_device = true;

private:
    std::string describe_;
    bool open_ = false;
    std::uint64_t recordings_ = 0;
    // Two timestamps a slot: as its command buffer starts and as it ends.
    VkQueryPool timestamps_ = VK_NULL_HANDLE;
    double timestamp_ns_ = 0;               // a tick, in nanoseconds
    bool stamped_[kSlots] = {};
    void MakeTimestamps();
    static constexpr std::uint32_t kMarks = 65536;
    VkQueryPool profile_ = VK_NULL_HANDLE;
    std::vector<int> marks_[kSlots];
    std::vector<std::function<void()>> after_submit_;
    int current_ = 0;
    bool in_flight_[kSlots] = {};
    std::vector<std::function<void()>> slot_done_[kSlots];
    CommandStream* stream_ = nullptr;       // SetRecordThread
    /* Waits for a slot's work, if handed over, and runs what waited on it;
       the time it waited goes to `why`. */
    bool Retire(int slot, const char* why);
};

template <typename F>
void VulkanGpu::RecordInPass(F&& record) {
    if (!open_) {
        const auto begin_buffer = [c = cmd](void*) {
            vkResetCommandBuffer(c, 0);
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(c, &begin);
        };
        if (CommandStream* s = StreamFor(cmd)) s->Add(begin_buffer);
        else begin_buffer(nullptr);
        open_ = true;
        ++recordings_;
        if (timestamps_) {
            vkCmdResetQueryPool(cmd, timestamps_, 2 * std::uint32_t(current_), 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps_, 2 * std::uint32_t(current_));
        }
        if (profile_) {
            vkCmdResetQueryPool(cmd, profile_, std::uint32_t(current_) * kMarks, kMarks);
            marks_[current_].clear();
            Mark(cmd, kProfOther);
        }
    }
    record(cmd);
}

template <typename F>
void VulkanGpu::Record(F&& record) {
    RecordInPass([&](VkCommandBuffer c) {
        if (end_pass) end_pass(c);
        record(c);
    });
}

inline bool VulkanGpu::Retire(int slot, const char* why) {
    if (!in_flight_[slot]) return true;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = vkWaitForFences(device, 1, &fences[slot], VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    const auto end = std::chrono::steady_clock::now();
    const double waited = std::chrono::duration<double>(end - start).count();
    g_wait_seconds += waited;
    if (g_wait_observer && waited > 0.0001)
        g_wait_observer(why, std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count()),
                        std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count()));
    CountWait(why, waited);
    in_flight_[slot] = false;
    if (stamped_[slot]) {
        std::uint64_t ticks[2] = {};
        if (vkGetQueryPoolResults(device, timestamps_, 2 * std::uint32_t(slot), 2, sizeof ticks, ticks,
                                  sizeof ticks[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ticks[1] > ticks[0])
            g_gpu_ns += std::uint64_t(double(ticks[1] - ticks[0]) * timestamp_ns_);
        stamped_[slot] = false;
    }
    if (profile_ && marks_[slot].size() > 1) {
        std::vector<std::uint64_t> ticks(marks_[slot].size());
        if (vkGetQueryPoolResults(device, profile_, std::uint32_t(slot) * kMarks, std::uint32_t(ticks.size()),
                                  ticks.size() * sizeof ticks[0], ticks.data(), sizeof ticks[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            for (std::size_t i = 1; i < ticks.size(); ++i)
                if (ticks[i] > ticks[i - 1])
                    g_profile_ns[marks_[slot][i]] += std::uint64_t(double(ticks[i] - ticks[i - 1]) * timestamp_ns_);
        marks_[slot].clear();
    }
    std::vector<std::function<void()>> done;
    done.swap(slot_done_[slot]);
    for (auto& f : done) f();
    return ok;
}

inline bool VulkanGpu::Flush() {
    if (!open_) return true;
    if (end_pass) end_pass(cmd);
    Mark(cmd, kProfOther);
    open_ = false;
    if (timestamps_) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps_, 2 * std::uint32_t(current_) + 1);
        stamped_[current_] = true;
    }
    // Ended and submitted where it was recorded: on the thread, waited for
    // (the queue's order is the order things were handed over in).
    VkResult submitted = VK_SUCCESS;
    const auto end_and_submit = [c = cmd, d = device, q = queue, f = fences[current_], out = &submitted](void*) {
        vkEndCommandBuffer(c);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &c;
        vkResetFences(d, 1, &f);
        std::lock_guard<std::mutex> hold(g_queue_lock);
        *out = vkQueueSubmit(q, 1, &submit, f);
    };
    if (CommandStream* s = StreamFor(cmd)) {
        s->Add(end_and_submit);
        s->Drain();
    } else {
        end_and_submit(nullptr);
    }
    const bool ok = submitted == VK_SUCCESS;
    ++g_submits;
    in_flight_[current_] = ok;
    slot_done_[current_].swap(after_submit_);
    if (!ok) {                                      // nothing will run: nothing to wait for
        for (auto& f : slot_done_[current_]) f();
        slot_done_[current_].clear();
    }
    current_ = (current_ + 1) % kSlots;
    cmd = cmds[current_];
    return Retire(current_, "a full ring") && ok;
}

inline bool VulkanGpu::Submit(const char* why) {
    bool ok = Flush();
    // Oldest first: current_ is free, the one after it was handed over first.
    for (int i = 1; i <= kSlots; ++i) ok = Retire((current_ + i) % kSlots, why) && ok;
    std::vector<std::function<void()>> done;        // waited on nothing recorded
    done.swap(after_submit_);
    for (auto& f : done) f();
    return ok;
}

inline bool VulkanGpu::SetRecordThread(bool on) {
    if (on == (stream_ != nullptr)) return on;
    Submit("the recording thread");
    if (on) {
        stream_ = new CommandStream(cmds, kSlots);
        g_stream = stream_;
    } else {
        g_stream = nullptr;
        delete stream_;
        stream_ = nullptr;
    }
    return on;
}

}  // namespace ps2gpu

#endif  /* PS2GPU_GPU_INTERNAL_H */
