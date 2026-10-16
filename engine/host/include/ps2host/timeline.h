/*
 * The frame-pacing profiler: a timeline of every thread's work and of every
 * frame's way to the screen (PS2HOST_PROFILE=file). Each thread records into
 * its own buffer - what it was busy with (the profile.h sections, the VIF1
 * worker's and the GS thread's busy spans, the EE's idle waits), when each
 * vertical blank came, when each picture was due and when it was shown, the
 * pictures dropped and the blanks lost - and at the end:
 *
 *  - the file: Chrome's trace format (ui.perfetto.dev or chrome://tracing
 *    shows it), a lane a thread;
 *  - a summary (stderr): the frames shown, their intervals' percentiles, the
 *    dropped frames (a picture dropped before it was shown, a present
 *    interval past one and a half frames, a blank the EE lost) and for the
 *    worst, which thread was busy in the frame before it.
 *
 * PS2HOST_PROFILE_FROM=N starts recording at frame N (the race, not the
 * boot). Off (the variable unset) every call returns at once.
 */
#ifndef PS2HOST_TIMELINE_H
#define PS2HOST_TIMELINE_H

#include <cstdint>

namespace ps2host::timeline {

/* Whether recording is on (PS2HOST_PROFILE set, from its first frame). */
bool On();
/* Now, in the timeline's clock (steady, nanoseconds). */
std::uint64_t Now();
/* The calling thread's lane name ("EE", "GS", "VIF1", "present"...). */
void NameThread(const char* name);
/* A span of work on the calling thread, [from, to); one of the same name
   ending within 20 us before it is extended instead (the file stays small). */
void Span(const char* name, std::uint64_t from_ns, std::uint64_t to_ns);
/* A moment on the calling thread. */
void Instant(const char* name, std::uint64_t at_ns, std::uint64_t value = 0);
/* A vertical blank delivered to the game: frame `frame` begins. */
void Frame(std::uint64_t frame);
/* A blank the EE was too busy for: lost to the game. */
void BlankLost();
/* A picture shown (due when, shown when), or dropped before it was. */
void Presented(std::uint64_t due_ns, std::uint64_t shown_ns);
void Dropped(std::uint64_t due_ns);
/* The file and the summary: at the run's end. */
void Finish();

}  // namespace ps2host::timeline

#endif  // PS2HOST_TIMELINE_H
