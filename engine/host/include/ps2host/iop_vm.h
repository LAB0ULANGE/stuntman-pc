/*
 * Running a game's own IOP modules.
 *
 * The SDK's IOP services are host code (iop.cpp and its servers), but games ship
 * modules of their own - sound drivers above all - whose protocols are the
 * game's. Rather than reimplementing each, the host runs them: an R3000
 * interpreter over IOP RAM loads the module (an IRX, a relocatable ELF) from the
 * disc, relocates it, and binds its imports to host implementations of the IOP
 * libraries (iop_libs.cpp, by library and export number from
 * engine/data/iop_exports.csv). Its threads run under the IOP kernel's
 * priority scheduler; its SIF command handlers receive the EE's commands; what
 * it sends comes back to the EE as the IOP's would.
 *
 * The IOP runs when there is something for it to do: after a module starts,
 * after a command arrives, and every vertical blank (for its timers and the
 * sound hardware's interrupts). It runs until every IOP thread waits.
 */

#ifndef PS2HOST_IOP_VM_H
#define PS2HOST_IOP_VM_H

#include <cstdint>
#include <string>
#include <vector>

namespace ps2host {
namespace iop {

/* A module, by file name ("RWA.IRX"), that runs in the interpreter when the
   game loads it. Ports register theirs. */
void RunModule(const std::string& file_name);
bool RunsModule(const std::string& path);

/* Loads and starts a module image; its start routine's result (0: resident,
   1: no resident, 2: removable) goes to *result. False if it cannot be loaded. */
bool LoadModule(const std::string& path, const std::vector<std::uint8_t>& image,
                const std::vector<std::uint8_t>& args, std::int32_t* result);

/* An EE command (not one of the SIF system's) for a handler a module added;
   `data_dest`/`data` is the extra data DMA'd to IOP memory before it. False when
   no module handles the command. */
bool HasCommandHandler(std::uint32_t command);
void Command(std::uint32_t command, const std::vector<std::uint8_t>& packet);

/* IOP memory from the modules' heap, for the EE's iopheap service; 0 if none. */
std::uint32_t Allocate(std::uint32_t size);
void Free(std::uint32_t address);

/* RPC servers a module registered (sceSifRegisterRpc). For a bind: the
   server's address and its receive buffers, false if no module serves `sid`. */
bool RpcServer(std::uint32_t sid, std::uint32_t* server, std::uint32_t* buffer,
               std::uint32_t* cbuffer);
bool OwnsServer(std::uint32_t server);
/* A call to such a server: its arguments are already in the server's buffer
   (the EE DMA'd them there); the module's thread runs the function, and the
   answer goes back as the IOP's would (QueueRpcEnd). */
void RpcCall(std::uint32_t server, std::uint32_t procedure, std::uint32_t size,
             std::uint32_t receive, std::uint32_t receive_size,
             const std::vector<std::uint8_t>& packet);

/* Runs the IOP until every thread waits, or it has run `instructions` (a
   busy IOP can't hang the EE; the EE's idle wait gives it a short slice, so
   a vertical blank is never kept waiting). */
void Run(std::uint64_t instructions = 20'000'000);

/* Log every library call a module makes (--iop-trace). */
void SetTrace(bool on);

/* Send the SPU2's mixed output to the host's sound device (--audio). */
void EnableSound();

/* For the report: one line per IOP thread. */
std::string Report();

}  // namespace iop
}  // namespace ps2host

#endif  /* PS2HOST_IOP_VM_H */
