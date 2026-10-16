/*
 * The IOP's module loader (server 0x80000006, loadfile), on the host.
 *
 * Nothing runs on an IOP here: what a module would provide, the host provides,
 * one RPC server at a time. So loading a module succeeds without running
 * anything - it is noted, given a module ID, and reported as having started
 * and stayed resident - and the servers it would have registered answer from
 * the host when the game binds them (or are noted as missing).
 *
 * The protocol, as Burnout 3's EE-side library (0x00113578-0x00113F98) speaks it:
 *   procedure 0xFF  the module's version, 4 characters: the library compares it
 *                   with the release it was built for and refuses to load
 *                   anything otherwise. "2800" is SDK release 2.8.
 *   procedure 0     load a module from a path. The request:
 *                     0x000  argument length       (the answer: module ID)
 *                     0x004  -                     (the answer: start result)
 *                     0x008  path, 252 bytes ("cdrom0:\\IOP\\USBD.IRX;1")
 *                     0x104  arguments, 252 bytes
 * Any other procedure is noted and fails.
 */

#include <cstring>
#include <string>
#include <vector>

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/iop_vm.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

// The release the IOP's loadfile answers: until the game reboots the IOP with
// a replacement image, the one Burnout 3's ROM-booted IOP was taken to have.
char g_version[4] = {'2', '8', '0', '0'};
constexpr s32 kNoSuchProcedure = -1;
constexpr u32 kResidentEnd = 0;     // a module's start result: started, stays loaded

std::vector<u8> Words(s32 first, u32 second) {
    std::vector<u8> out(8);
    std::memcpy(&out[0], &first, 4);
    std::memcpy(&out[4], &second, 4);
    return out;
}

// A module the port runs (iop_vm.h): read from the disc and started.
std::vector<u8> RunFromDisc(s32 id, const std::string& path, const std::vector<u8>& data) {
    std::string disc_path = path;
    const std::size_t colon = disc_path.find(':');
    if (colon != std::string::npos) disc_path = disc_path.substr(colon + 1);
    Disc::File file;
    if (!GuestDisc().IsOpen() || !GuestDisc().Find(disc_path, &file))
        throw Unimplemented{"IOP: module " + path + " is to run, but is not on the disc"};
    std::vector<u8> image((file.size + kSectorSize - 1) / kSectorSize * kSectorSize);
    GuestDisc().Read(file.lsn, u32(image.size() / kSectorSize), image.data());
    image.resize(file.size);
    s32 arg_length = 0;
    std::memcpy(&arg_length, &data[0], 4);
    std::vector<u8> args;
    if (arg_length > 0 && data.size() >= 0x104 + std::size_t(arg_length))
        args.assign(data.begin() + 0x104, data.begin() + 0x104 + arg_length);
    s32 result = 0;
    if (!iop::LoadModule(path, image, args, &result))
        throw Unimplemented{"IOP: " + path + " is not a module the interpreter can load"};
    return Words(id, u32(result));
}

std::vector<u8> LoadFile(u32 procedure, const std::vector<u8>& data) {
    static s32 next_id = 1;
    if (procedure == 0xFF) return std::vector<u8>(g_version, g_version + 4);
    if (procedure == 0 && data.size() >= 0x104) {
        const char* begin = reinterpret_cast<const char*>(&data[8]);
        const std::string path(begin, strnlen(begin, 252));
        const s32 id = next_id++;
        if (iop::RunsModule(path)) return RunFromDisc(id, path, data);
        Note("module " + path, "IOP: module " + path + " loaded as " + std::to_string(id) +
                                   " (its services are the host's)");
        return Words(id, kResidentEnd);
    }
    Note("loadfile " + std::to_string(procedure),
         "IOP: loadfile procedure " + std::to_string(procedure) + " is not implemented (failed)");
    return Words(kNoSuchProcedure, 0);
}

}  // namespace

void InstallLoadFile() { RegisterRpcServer(0x80000006u, LoadFile); }

// An IOP replacement image (IOPRPnnn.IMG) holds its modules, each with its
// library tag - eight characters of name, four of release: "loadfile2340".
// Its loadfile's release is what loadfile answers from then on.
void UseIopImage(const std::string& path) {
    std::string disc_path = path;
    const std::size_t colon = disc_path.find(':');
    if (colon != std::string::npos) disc_path = disc_path.substr(colon + 1);
    std::vector<u8> image;
    if (!GuestDisc().IsOpen() || !GuestDisc().ReadFile(disc_path, &image)) {
        Note("iop image " + path, "IOP: reboot image " + path + " is not on the disc");
        return;
    }
    static const char kTag[] = "loadfile";
    for (std::size_t i = 0; i + 12 <= image.size(); ++i) {
        if (std::memcmp(&image[i], kTag, 8) != 0) continue;
        bool digits = true;
        for (int k = 8; k < 12; ++k) digits = digits && image[i + k] >= '0' && image[i + k] <= '9';
        if (!digits) continue;
        std::memcpy(g_version, &image[i + 8], 4);
        Note("iop image " + path, "IOP: rebooted with " + path + ": loadfile " +
                                      std::string(g_version, 4));
        return;
    }
    Note("iop image " + path, "IOP: reboot image " + path + " has no loadfile tag");
}

}  // namespace ps2host
