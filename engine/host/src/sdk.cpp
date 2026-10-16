/*
 * Host implementations of Sony SDK library routines, applied by name.
 *
 * Library routines that drive the hardware - wait for the graphics paths, poll
 * a DMA channel - are better replaced whole than emulated register by register:
 * the routine's contract is small, and on the host the hardware it waits for is
 * already done. A game's analysis names such routines (config/names.csv, from
 * diagnostic strings and the cross-game signature library); every routine named
 * here is overridden wherever the game's dispatch table gives that name, so a new
 * game gets these for free once its routines are named.
 */

#include <cstring>

#include "ps2host/host.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

void Return(lifted::Context& c, u64 value) { c.r[2] = u64(s64(s32(u32(value)))); }

// sceGsSyncPath(mode, timeout): wait until the paths to the GS are idle - VIF1,
// GIF and VU1 - and return 0, or a negative value on timeout. The host's
// transfers complete when they are started, so the paths are always idle.
void GsSyncPath(lifted::Context& c) { Return(c, 0); }

struct SdkRoutine {
    const char* name;
    GuestFunction function;
};

const SdkRoutine kRoutines[] = {
    {"sceGsSyncPath", GsSyncPath},
};

}  // namespace

void ApplySdkOverrides(const FunctionEntry* table, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!table[i].name) continue;
        for (const SdkRoutine& routine : kRoutines) {
            if (std::strcmp(table[i].name, routine.name) == 0) {
                Override(table[i].address, routine.function, routine.name);
                Note(std::string("sdk ") + routine.name,
                     std::string("SDK routine ") + routine.name + " runs on the host");
            }
        }
    }
}

}  // namespace ps2host
