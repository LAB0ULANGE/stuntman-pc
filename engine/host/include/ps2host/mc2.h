/*
 * A memory card the host keeps as a directory, for games that use libmc2.
 *
 * libmc2 (the SDK's second memory card library) takes one request at a time
 * on a worker thread of its own and hands each to a handler for its command;
 * the handlers speak to the card through DBCMAN and MC2_S1 on the IOP, which
 * drives the SIO2 and authenticates the card (MagicGate). The host stands in
 * one level up: these are the handlers, over a directory (--memcard DIR) -
 * the game's queue, worker and polling stay as they are. A port binds them
 * to the addresses its libmc2 has them at (games/burnout3/port/overrides.cpp;
 * the contracts are in games/burnout3/docs/FINDINGS.md, "Memory cards").
 *
 * Each takes the worker's arguments in $a0, $a1, ... ($a0 the socket) and
 * leaves the result word in $v0: 0 or a count for success, else 0x8101xxxx
 * (0x81010000 | errno, libmc2's own at 0x81019xxx). Paths are relative to
 * the card's root (no game here changes directory); a path that would leave
 * the directory is refused.
 */

#ifndef PS2HOST_MC2_H
#define PS2HOST_MC2_H

#include <string>

#include "ps2rt/lifted.h"

namespace ps2host {
namespace mc2 {

/* The card's directory; empty when no card was given (the game then finds
   none, through its own handlers). */
const std::string& CardDirectory();
void SetCardDirectory(const std::string& directory);

/* (socket, SceMc2InfoParam*): type 2 (a PS2 card), formatted, free KB. */
void GetInfo(lifted::Context& c);
/* (socket): the directory is the card, always formatted; nothing is erased. */
void Format(lifted::Context& c);
/* (socket, name, buffer, offset, size): bytes read; size 0 reads to the end. */
void ReadFile(lifted::Context& c);
/* (socket, name, buffer, offset, size): the file must exist; never truncates. */
void WriteFile(lifted::Context& c);
/* (socket, name, mode): a file, or a directory when mode has 0x20. */
void Create(lifted::Context& c);
/* (socket, name, offset, maxent, SceMc2DirParam*, int* count): `name` is a
   directory and a pattern (* and ?); entries from the offset-th on. */
void GetDir(lifted::Context& c);
/* (socket, name, entry*): the raw directory entry libmc2's lookup fills (mode
   +0, size +4, created +8, modified +0x18, name +0x40); 0x81010011 when found,
   as libmc2's has it. SearchFile is made of it. */
void Lookup(lifted::Context& c);

}  // namespace mc2
}  // namespace ps2host

#endif  /* PS2HOST_MC2_H */
