/*
 * The port's sound (--audio): the SPU2's mixed output (spu2.cpp) played through
 * the system's wave device. Windows' waveOut needs nothing beyond the system,
 * like the window; a lower-latency device can replace it behind the same two
 * calls.
 *
 * The SPU2 produces samples as the IOP runs, by the clock, so production
 * matches playback on average. Buffers are queued as they fill; when every
 * buffer is still queued the newest samples are dropped rather than the
 * emulation waiting, and an empty queue simply plays silence until the next.
 *
 * --audio-wav records the same output to a WAV file instead of (or as well as)
 * playing it: how sound is checked without anyone listening.
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ps2host/run.h"
#include <cmath>

#pragma comment(lib, "winmm.lib")

namespace ps2host {

namespace {

constexpr int kBuffers = 8;
constexpr DWORD kFrames = 1024;                 // 21 ms at 48 kHz

struct Device {
    HWAVEOUT out = nullptr;
    WAVEHDR headers[kBuffers] = {};
    std::vector<short> data[kBuffers];
    int filling = 0;                            // the buffer being filled
    DWORD filled = 0;                           // frames in it
    unsigned long long dropped = 0;
    bool mute = false;                          // PS2HOST_AUDIO_MUTE: silence played
    bool steer = false;                         // SetAudioSteer
    AudioSteer steering;
    std::vector<short> stretched;
    std::FILE* wav = nullptr;
    unsigned long long wav_frames = 0;

    // The WAV header's sizes are known only at the end. The device is
    // stopped and closed before the buffers go: the system's audio thread
    // was still copying from the queued ones as the process freed them (a
    // crash at exit, one run in two).
    ~Device() { Close(); }
    void Close() {
        if (out) {
            waveOutReset(out);
            for (WAVEHDR& h : headers)
                if (h.dwFlags & WHDR_PREPARED) waveOutUnprepareHeader(out, &h, sizeof h);
            waveOutClose(out);
            out = nullptr;
        }
        if (!wav) return;
        const unsigned data = unsigned(wav_frames * 4);
        WriteHeader(data);
        std::fclose(wav);
        wav = nullptr;
    }
    void WriteHeader(unsigned data) {
        unsigned char h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                               16, 0, 0, 0, 1, 0, 2, 0, 0x80, 0xBB, 0, 0, 0, 0xEE, 2, 0,
                               4, 0, 16, 0, 'd', 'a', 't', 'a', 0, 0, 0, 0};
        const unsigned riff = 36 + data;
        std::memcpy(h + 4, &riff, 4);
        std::memcpy(h + 40, &data, 4);
        std::fseek(wav, 0, SEEK_SET);
        std::fwrite(h, 1, 44, wav);
        std::fseek(wav, 0, SEEK_END);
    }
};

Device& D() {
    static Device device;
    return device;
}

}  // namespace

void CloseAudio() { D().Close(); }

void SetAudioSteer(bool on) { D().steer = on; }

double SteerRatio(double level, double target, double capacity) {
    // Faster (above 1) the fuller, up to 1% when half the queue past the target.
    const double r = 1.0 + 0.02 * (level - target) / capacity;
    return std::min(1.01, std::max(0.99, r));
}

void AudioSteer::Stretch(const short* frames, unsigned count, double ratio, std::vector<short>* out) {
    // Linear interpolation between consecutive frames, `ratio` input frames to
    // an output frame, the position kept across batches (no seam between them).
    out->clear();
    if (!count) return;
    auto at = [&](long long i, int ch) -> double {
        return i < 0 ? double(last_[ch]) : double(frames[std::size_t(i) * 2 + ch]);
    };
    // position_ is from the last frame of the batch before (index -1).
    while (position_ < double(count) - 1.0) {
        const long long i = (long long)std::floor(position_);
        const double f = position_ - double(i);
        for (int ch = 0; ch < 2; ++ch) {
            const double v = at(i, ch) + (at(i + 1, ch) - at(i, ch)) * f;
            out->push_back(short(std::lround(std::min(32767.0, std::max(-32768.0, v)))));
        }
        position_ += ratio;
    }
    position_ -= double(count);
    last_[0] = frames[std::size_t(count - 1) * 2];
    last_[1] = frames[std::size_t(count - 1) * 2 + 1];
}

bool OpenAudio() {
    Device& d = D();
    WAVEFORMATEX format = {};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 2;
    format.nSamplesPerSec = 48000;
    format.wBitsPerSample = 16;
    format.nBlockAlign = 4;
    format.nAvgBytesPerSec = 48000 * 4;
    const MMRESULT opened = waveOutOpen(&d.out, WAVE_MAPPER, &format, 0, 0, CALLBACK_NULL);
    if (opened != MMSYSERR_NOERROR) {
        d.out = nullptr;
        return false;
    }
    // PS2HOST_AUDIO_MUTE=1: played, unheard - the device measured
    // (PS2HOST_AUDIO_LOG) in silence: silence written, not the volume set (a
    // program's wave volume is the system's mixer's, which keeps it).
    d.mute = std::getenv("PS2HOST_AUDIO_MUTE") != nullptr;
    DWORD volume = 0;
    if (waveOutGetVolume(d.out, &volume) == MMSYSERR_NOERROR) {
        // PS2HOST_AUDIO_RESET_VOLUME=1: this program's volume in the mixer
        // back to full (an earlier build's mute set it to nothing, kept).
        if (std::getenv("PS2HOST_AUDIO_RESET_VOLUME")) {
            waveOutSetVolume(d.out, 0xFFFFFFFFu);
            waveOutGetVolume(d.out, &volume);
        }
        std::fprintf(stderr, "[audio] playing; this program's volume in the system's mixer %u%%\n",
                     unsigned((volume & 0xFFFF) * 100 / 0xFFFF));
    }
    for (int i = 0; i < kBuffers; ++i) {
        d.data[i].assign(kFrames * 2, 0);
        d.headers[i].lpData = reinterpret_cast<LPSTR>(d.data[i].data());
        d.headers[i].dwBufferLength = kFrames * 4;
        d.headers[i].dwFlags = WHDR_DONE;           // free
    }
    return true;
}

bool OpenWavDump(const std::string& path) {
    Device& d = D();
    d.wav = std::fopen(path.c_str(), "wb");
    if (!d.wav) return false;
    d.WriteHeader(0);
    return true;
}

/* PS2HOST_AUDIO_LOG=1: every two seconds, how the sound came - the longest
   wait between the SPU2's batches of samples and how many waited past 10, 20
   and 40 ms - and what the device made of it: buffers it ran out of (none
   queued when the next was given), samples dropped (every buffer queued),
   the fewest it had queued. */
struct AudioLog {
    bool on = std::getenv("PS2HOST_AUDIO_LOG") != nullptr;
    std::chrono::steady_clock::time_point last{}, since{};
    double longest_ms = 0;
    unsigned over10 = 0, over20 = 0, over40 = 0, batches = 0;
    unsigned ran_out = 0;
    unsigned long long dropped_before = 0;
    int fewest_queued = 99;
    unsigned long long frames = 0, queued_sum = 0;   // made; buffers queued, summed over batches
};
AudioLog& Log() {
    static AudioLog log;
    return log;
}

int Queued(Device& d) {
    int n = 0;
    for (const WAVEHDR& h : d.headers)
        if ((h.dwFlags & WHDR_PREPARED) && !(h.dwFlags & WHDR_DONE)) ++n;
    return n;
}

void NoteBatch(Device& d, unsigned count) {
    AudioLog& l = Log();
    const auto now = std::chrono::steady_clock::now();
    if (l.last.time_since_epoch().count()) {
        const double ms = std::chrono::duration<double, std::milli>(now - l.last).count();
        l.longest_ms = std::max(l.longest_ms, ms);
        l.over10 += ms > 10;
        l.over20 += ms > 20;
        l.over40 += ms > 40;
    } else {
        l.since = now;
    }
    l.last = now;
    ++l.batches;
    l.frames += count;
    if (d.out) {
        l.fewest_queued = std::min(l.fewest_queued, Queued(d));
        l.queued_sum += unsigned(Queued(d));
    }
    if (now - l.since >= std::chrono::seconds(2)) {
        std::fprintf(stderr, "[audio] %u batches, longest wait %.1f ms (%u past 10 ms, %u past 20, %u past 40); "
                             "device: ran out %u times, %llu samples dropped, fewest queued %d, %.1f on average; "
                             "%.0f samples a second made\n",
                     l.batches, l.longest_ms, l.over10, l.over20, l.over40, l.ran_out,
                     d.dropped - l.dropped_before, l.fewest_queued,
                     l.batches ? double(l.queued_sum) / l.batches : 0.0,
                     double(l.frames) / std::chrono::duration<double>(now - l.since).count());
        l = AudioLog{};
        l.last = now;
        l.since = now;
        l.dropped_before = d.dropped;
    }
}

void AudioFrames(const short* frames, unsigned count) {
    Device& d = D();
    if (Log().on) NoteBatch(d, count);
    // Steered (audio_steer): played a touch faster or slower so that what
    // is queued stays near kSteerTarget - made and played at the same rate,
    // the queue stayed wherever a burst had left it: nearly full, ~150 ms
    // behind, and each burst past it dropped (a crackle).
    if (d.out && d.steer && !d.wav) {
        const double level = double(Queued(d)) * kFrames + d.filled;
        const double ratio = SteerRatio(level, kSteerTarget * double(kFrames), double(kBuffers) * kFrames);
        d.steering.Stretch(frames, count, ratio, &d.stretched);
        frames = d.stretched.data();
        count = unsigned(d.stretched.size() / 2);
    }
    if (d.wav) {
        std::fwrite(frames, 4, count, d.wav);
        d.wav_frames += count;
    }
    if (!d.out) return;
    while (count) {
        WAVEHDR& h = d.headers[d.filling];
        if (!(h.dwFlags & WHDR_DONE)) {             // every buffer queued: drop
            d.dropped += count;
            return;
        }
        const DWORD take = std::min<DWORD>(count, kFrames - d.filled);
        if (d.mute) std::memset(d.data[d.filling].data() + d.filled * 2, 0, take * 4);
        else std::memcpy(d.data[d.filling].data() + d.filled * 2, frames, take * 4);
        d.filled += take;
        frames += take * 2;
        count -= take;
        if (d.filled == kFrames) {
            if (Log().on && Queued(d) == 0) ++Log().ran_out;       // the device had nothing to play
            if (h.dwFlags & WHDR_PREPARED) waveOutUnprepareHeader(d.out, &h, sizeof h);
            h.dwFlags = 0;
            waveOutPrepareHeader(d.out, &h, sizeof h);
            waveOutWrite(d.out, &h, sizeof h);
            d.filling = (d.filling + 1) % kBuffers;
            d.filled = 0;
        }
    }
}

}  // namespace ps2host
