/*
 * The IOP's controller manager, dbcman (server 0x80001300), on the host: the
 * devices behind the SDK's libdbc and, above it, libpad2. One DualShock 2 is
 * attached on port 0, driven by the window's game controller (SDL3, any one
 * SDL knows) and by the keyboard while the window has focus.
 *
 * The protocol, as Burnout 3's copies of libdbc (0x00215F18-0x00216D20) and
 * libpad2 (0x00215620-0x00215EE0) speak it:
 *
 *   0x80001363  version: answer word 0 = major << 8 | minor (libdbc 3.0 needs 3)
 *   0x80001304  sceDbcSetWorkAddr: +4 the EE work area; its first 16 words are
 *               each socket's status, 1 when a device is there
 *   0x80001301  sceDbcCreateSocket: +8 the port, +0x28 and +0x2C the EE
 *               addresses of the socket's two 0x80-byte data blocks; answer
 *               +0x24 the socket (-1: none)
 *   0x80001302  sceDbcDeleteSocket: +0 the socket
 *   0x80001303  sceDbcGetDepNumber: +0 the socket; answer +4 (0: present)
 *   0x8000131B  sceDbcSendData2: +0 the socket, +4 a header, +8 the length,
 *               +0xC the data: a command's length (word), its data's length
 *               (word), the data, the command. Command 3 with two bytes of
 *               data sets a DualShock 2's motors (Burnout 3's 0x00217070):
 *               the small motor in bit 0, the large one's 0-255 above it.
 *
 * The device writes its state into the socket's blocks, the newer one
 * marked by the larger counter at +0x7C. A block:
 *   +0 state (1: connected)   +2 data length   +3 profile length
 *   +4 status (non-zero: valid)   +0x1C the data, then the profile
 * The profile is a bitmask of 40 inputs; libpad2 lays the data out by it
 * (0x00215E30): inputs 16-31 and 35-38 are a byte each (analog), the others
 * a bit each, in input order. A DualShock 2 has inputs 0-31: 16 buttons
 * (active low: a pressed button's bit is 0), the four stick bytes, then 12
 * pressures - 18 bytes,
 * profile FF FF FF FF 00.
 */

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ps2host/host.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

constexpr u32 kVersion = 0x0300;    // dbcman 3.0, as libdbc 3.0 expects
constexpr u32 kDataBytes = 18, kProfileBytes = 5;

struct Socket {
    u32 port = 0, block[2] = {0, 0};
    u32 counter = 0;
};

struct Dbc {
    u32 work = 0;
    std::map<s32, Socket> sockets;
};

// Ports 0 and 1 are the controller ports, 2 and 3 the memory card slots. A
// socket opens on any of them; what is plugged in shows in its blocks. There is
// a DualShock 2 on port 0 and nothing else - no second pad, no memory card.
constexpr u32 kPorts = 4;
bool Present(u32 port) { return port == 0; }

Dbc& D() {
    static Dbc dbc;
    return dbc;
}

u32 At(const std::vector<u8>& data, std::size_t offset) {
    u32 value = 0;
    if (offset + 4 <= data.size()) std::memcpy(&value, &data[offset], 4);
    return value;
}

void Put(std::vector<u8>& data, std::size_t offset, u32 value) {
    if (offset + 4 <= data.size()) std::memcpy(&data[offset], &value, 4);
}

// The keyboard as a DualShock 2: (input, key). Inputs 0-15 are the buttons in
// libpad's order: Select, L3, R3, Start, Up, Right, Down, Left, L2, R2, L1, R1,
// Triangle, Circle, Cross, Square.
const std::pair<int, int> kKeys[] = {
    {0, VK_BACK}, {3, VK_RETURN}, {4, VK_UP}, {5, VK_RIGHT}, {6, VK_DOWN}, {7, VK_LEFT},
    {8, '1'}, {9, '3'}, {10, 'Q'}, {11, 'E'}, {12, 'V'}, {13, 'C'}, {14, 'X'}, {15, 'Z'},
};

bool Pressed(int key) { return WindowFocused() && HostKeyDown(key); }

// Scripted presses (--press): (input, first frame, last frame).
struct Press { int input; u64 from, to; };
std::vector<Press>& Script() {
    static std::vector<Press> presses;
    return presses;
}
u64 g_frame = 0;
u16 g_held = 0;                     // PadButtonsHeld
u16 (*g_pad_hook)() = nullptr;      // SetPadHook

void PadData(u8* data) {
    u16 buttons = 0xFFFF;                                   // active low: nothing pressed
    for (const auto& [input, key] : kKeys)
        if (Pressed(key)) buttons &= u16(~(1u << input));
    for (const Press& p : Script())
        if (g_frame >= p.from && g_frame <= p.to) buttons &= u16(~(1u << p.input));
    HostPadState pad;
    const bool controller = HostPad(&pad);
    buttons &= u16(~pad.buttons);
    if (g_pad_hook) buttons &= u16(~g_pad_hook());
    g_held = u16(~buttons);
    data[0] = u8(buttons);
    data[1] = u8(buttons >> 8);
    // Then, as libpad's DualShock 2 report has them: the sticks (right X, Y,
    // left X, Y), centred without a controller, and the pressure of the
    // pressure-sensitive buttons - right, left, up, down, triangle, circle,
    // cross, square, L1, R1, L2, R2 - full when pressed, the triggers' own
    // travel for L2 and R2. (Sticks read as 0 are pushed fully up and left:
    // with them there, the frontend saw input held and never a press.)
    for (u32 i = 2; i < 6; ++i) data[i] = controller ? pad.sticks[i - 2] : 0x80;
    static const int kPressure[12] = {5, 7, 4, 6, 12, 13, 14, 15, 10, 11, 8, 9};
    for (u32 i = 0; i < 12; ++i) data[6 + i] = (buttons >> kPressure[i]) & 1 ? 0 : 0xFF;
    if (controller && pad.l2 && !(buttons & (1u << 8))) data[6 + 10] = std::max<u8>(pad.l2, 1);
    if (controller && pad.r2 && !(buttons & (1u << 9))) data[6 + 11] = std::max<u8>(pad.r2, 1);
}

std::vector<u8> Server(u32 procedure, const std::vector<u8>& request) {
    Dbc& d = D();
    std::vector<u8> answer(request.size() < 0x30 ? 0x30 : request.size(), 0);
    std::copy(request.begin(), request.end(), answer.begin());
    switch (procedure) {
        case 0x80001363u:
            Put(answer, 0, kVersion);
            return answer;
        case 0x80001304u:
            d.work = At(request, 4);
            return answer;
        case 0x80001301u: {
            Socket s;
            s.port = At(request, 8);
            s.block[0] = At(request, 0x28);
            s.block[1] = At(request, 0x2C);
            s32 id = -1;
            if (s.port < kPorts)
                for (s32 n = 0; n < 16 && id < 0; ++n)
                    if (!d.sockets.count(n)) id = n;
            if (id >= 0) d.sockets[id] = s;
            char text[128];
            std::snprintf(text, sizeof text, "DBC: socket %d on port %u (blocks 0x%08X, 0x%08X)%s",
                          int(id), unsigned(s.port), unsigned(s.block[0]), unsigned(s.block[1]),
                          id < 0 ? ": no such port" : Present(s.port) ? "" : ": nothing plugged in");
            Note("dbc socket " + std::to_string(s.port), text);
            Put(answer, 0x24, u32(id));
            return answer;
        }
        case 0x80001302u:
            d.sockets.erase(s32(At(request, 0)));
            return answer;
        case 0x80001303u: {
            const auto found = d.sockets.find(s32(At(request, 0)));
            Put(answer, 4, found != d.sockets.end() && Present(found->second.port) ? 0 : u32(-1));
            return answer;
        }
        case 0x8000131Bu: {
            const u32 command_bytes = At(request, 0xC), data_bytes = At(request, 0x10);
            if (data_bytes == 2 && command_bytes == 1 && request.size() > 0x16 && request[0x16] == 3) {
                const u32 value = u32(request[0x14]) | u32(request[0x15]) << 8;
                const u8 small = (value & 1) ? 255 : 0, large = u8(std::min(255u, value >> 1));
                if (std::getenv("PS2HOST_RUMBLE_LOG"))
                    std::fprintf(stderr, "[pad] motors: large %u small %u (0x%04X)\n", unsigned(large),
                                 unsigned(small), unsigned(value));
                HostRumble(large, small);
            } else {
                char text[128];
                std::snprintf(text, sizeof text, "DBC: SendData2 command length %u, data length %u (ignored)",
                              unsigned(command_bytes), unsigned(data_bytes));
                Note("dbc senddata2 " + std::to_string(command_bytes) + "/" + std::to_string(data_bytes), text);
            }
            return answer;
        }
        case 0x8000131Au: {
            // sceDbcReceiveData: +0 socket, +4 command, +8 room. Answer: +0x20C
            // the result (< 0: failed), +8 the length, +0xC the data - whose
            // first byte libpad2 reads as the device's state (1: connected).
            if (answer.size() < 0x210) answer.resize(0x210, 0);
            const auto found = d.sockets.find(s32(At(request, 0)));
            const bool there = found != d.sockets.end() && Present(found->second.port);
            Put(answer, 0x20C, there ? 0 : u32(-1));
            Put(answer, 8, there ? 1 : 0);
            answer[0xC] = there ? 1 : 0;
            return answer;
        }
        default: {
            char text[96];
            std::snprintf(text, sizeof text, "IOP: dbcman procedure 0x%08X is not implemented (zeros)",
                          unsigned(procedure));
            Note("dbcman " + std::to_string(procedure), text);
            std::fill(answer.begin(), answer.end(), 0);
            return answer;
        }
    }
}

/*
 * The older controller module, padman (servers 0x80000100 and 0x80000101),
 * behind libpad 2.x - as Mister Mosquito's copy (0x0010AF50-0x0010C100,
 * "PsIIlibpad  2300", padman 4.x) speaks it. A call is procedure 1 with a
 * 0x80-byte buffer: +0 the command, then its arguments; the answer's result
 * is at +0xC.
 *
 *   0x12  version: +0xC = major << 8 | minor (the library needs major 4)
 *   0x10  init      0x0F  end
 *   0x01  scePadPortOpen: +4 port, +8 slot, +0x10 the EE pad area (two
 *         0x80-byte frames); answer +0x14 the IOP address libpad DMAs its
 *         actuator settings to
 *   0x0E  scePadPortClose: +4 port, +8 slot
 *
 * The module writes the pad's state into the older of the two frames each
 * frame, numbering it past the other:
 *   +0x00 the report: 0 (ok), the mode ID (0x73: DualShock, analog), the
 *         buttons (active low), the sticks; its length at +0x60
 *   +0x58 u32 frame number (libpad reads the frame with the larger)
 *   +0x50 u16 mode table (pad types: 4 digital, 7 DualShock), +0x64 2 (a
 *         controller with modes), +0x65 the mode ID, +0x68 modes, +0x69 the
 *         current mode, +0x6A actuators
 *   +0x70 state (6: stable), +0x71 request state (0: done, 2: busy),
 *   +0x72 1 (a controller is there)
 */
struct PadArea {
    u32 port = 0, slot = 0, area = 0;
    u32 frame = 0;
};
std::map<u64, PadArea>& PadAreas() {
    static std::map<u64, PadArea> areas;
    return areas;
}
constexpr u8 kPadmanMode = 0x73;                // DualShock, analog: 3 halfwords after the header

std::vector<u8> Padman(u32 /*procedure*/, const std::vector<u8>& request) {
    std::vector<u8> answer(request.size() < 0x80 ? 0x80 : request.size(), 0);
    std::copy(request.begin(), request.end(), answer.begin());
    const u32 command = At(request, 0);
    switch (command) {
        case 0x12:
            Put(answer, 0xC, 0x0400);
            return answer;
        case 0x10: case 0x0F:
            Put(answer, 0xC, 1);
            return answer;
        case 0x01: {
            PadArea p;
            p.port = At(request, 4);
            p.slot = At(request, 8);
            p.area = At(request, 0x10);
            PadAreas()[u64(p.port) << 32 | p.slot] = p;
            char text[96];
            std::snprintf(text, sizeof text, "PAD: port %u slot %u open (pad area 0x%08X)%s",
                          unsigned(p.port), unsigned(p.slot), unsigned(p.area),
                          Present(p.port) && p.slot == 0 ? "" : ": nothing plugged in");
            Note("padman open " + std::to_string(p.port), text);
            Put(answer, 0xC, 1);
            Put(answer, 0x14, 0x001F0000u + 0x100 * (p.port * 4 + p.slot));
            return answer;
        }
        case 0x0E:
            PadAreas().erase(u64(At(request, 4)) << 32 | At(request, 8));
            Put(answer, 0xC, 1);
            return answer;
        default: {
            char text[96];
            std::snprintf(text, sizeof text, "PAD: padman command 0x%X (answered done)", unsigned(command));
            Note("padman " + std::to_string(command), text);
            Put(answer, 0xC, 1);
            return answer;
        }
    }
}

void PadmanFrame() {
    Memory& m = GuestMemory();
    for (auto& [key, p] : PadAreas()) {
        const u32 f = p.area + 0x80 * ((p.frame + 1) & 1);
        ++p.frame;
        const bool present = Present(p.port) && p.slot == 0;
        for (u32 i = 0; i < 0x80; i += 4) m.Write(f + i, 0, 4);
        m.Write(f + 0x58, p.frame, 4);
        m.Write(f + 0x70, present ? 6 : 0, 1);           // stable, or disconnected
        m.Write(f + 0x71, 0, 1);
        if (!present) continue;
        u8 data[kDataBytes];
        PadData(data);
        if (std::getenv("PS2HOST_PAD_LOG") && (data[0] != 0xFF || data[1] != 0xFF))
            std::fprintf(stderr, "[pad] frame %llu port %u area 0x%08X frame# %u buttons %02X%02X\n",
                         (unsigned long long)g_frame, unsigned(p.port), unsigned(f), unsigned(p.frame), data[1], data[0]);
        const u32 length = 2 + 2 * (kPadmanMode & 0xF);
        m.Write(f + 0, 0, 1);
        m.Write(f + 1, kPadmanMode, 1);
        for (u32 i = 0; i + 2 < length; ++i) m.Write(f + 2 + i, data[i], 1);
        m.Write(f + 0x60, length, 4);
        m.Write(f + 0x50, 4, 2);                         // digital,
        m.Write(f + 0x52, 7, 2);                         // DualShock
        m.Write(f + 0x64, 2, 1);
        m.Write(f + 0x65, kPadmanMode, 1);
        m.Write(f + 0x68, 2, 1);
        m.Write(f + 0x69, 1, 1);
        m.Write(f + 0x6A, 2, 1);
        m.Write(f + 0x72, 1, 1);
    }
}

}  // namespace

bool ScriptPress(const std::string& button, u64 from, u64 to) {
    static const char* const kNames[16] = {"select", "l3", "r3", "start", "up", "right", "down", "left",
                                           "l2", "r2", "l1", "r1", "triangle", "circle", "cross", "square"};
    for (int i = 0; i < 16; ++i)
        if (button == kNames[i]) {
            Script().push_back({i, from, to});
            return true;
        }
    return false;
}

// Each frame: every socket's device writes its state into the next block.
void DbcFrame() {
    ++g_frame;
    PadmanFrame();
    Dbc& d = D();
    Memory& m = GuestMemory();
    for (auto& [id, s] : d.sockets) {
        const bool present = Present(s.port);
        if (d.work) m.Write(d.work + 4 * u32(id), present ? 1 : 0, 4);
        const u32 b = s.block[(s.counter + 1) & 1];
        if (!b) continue;
        if (!present) {                                 // an empty port: disconnected
            m.Write(b + 0, 0, 1);
            m.Write(b + 4, 0, 4);
            m.Write(b + 0x7C, ++s.counter, 4);
            continue;
        }
        u8 data[kDataBytes];
        PadData(data);
        m.Write(b + 0, 1, 1);                           // connected
        m.Write(b + 1, 0, 1);
        m.Write(b + 2, kDataBytes, 1);
        m.Write(b + 3, kProfileBytes, 1);
        m.Write(b + 4, 1, 4);                           // valid
        for (u32 i = 0; i < kDataBytes; ++i) m.Write(b + 0x1C + i, data[i], 1);
        const u8 profile[kProfileBytes] = {0xFF, 0xFF, 0xFF, 0xFF, 0x00};
        for (u32 i = 0; i < kProfileBytes; ++i) m.Write(b + 0x1C + kDataBytes + i, profile[i], 1);
        m.Write(b + 0x7C, ++s.counter, 4);
    }
}

void InstallDbc() {
    RegisterRpcServer(0x80001300u, Server);
    // libdbc binds a client of its own to each send (SendData2, SendData3):
    // their servers' ids are the procedures', and dbcman answers them alike.
    RegisterRpcServer(0x8000131Bu, Server);
    RegisterRpcServer(0x8000131Cu, Server);
    RegisterRpcServer(0x80000100u, Padman);
    RegisterRpcServer(0x80000101u, Padman);
}

u16 PadButtonsHeld() { return g_held; }

void SetPadHook(u16 (*hook)()) { g_pad_hook = hook; }

namespace {
struct MouseStep {
    u64 frame;
    float x, y;
    char action;            // 'L', 'R', 'U', 'D' or 0
};
std::vector<MouseStep>& MouseScript() {
    static std::vector<MouseStep> steps;
    return steps;
}
}  // namespace

bool ScriptMouse(const std::string& spec) {
    MouseStep step{};
    char action[2] = {};
    const int n = std::sscanf(spec.c_str(), "%llu:%f,%f,%1[LRUD]", reinterpret_cast<unsigned long long*>(&step.frame),
                              &step.x, &step.y, action);
    if (n < 3) return false;
    step.action = action[0];
    MouseScript().push_back(step);
    std::sort(MouseScript().begin(), MouseScript().end(),
              [](const MouseStep& a, const MouseStep& b) { return a.frame < b.frame; });
    return true;
}

bool ScriptedMouse(HostMouseState* state) {
    if (MouseScript().empty()) return false;
    *state = HostMouseState{};
    for (const MouseStep& s : MouseScript()) {
        if (s.frame > g_frame) break;
        state->over = true;
        state->x = s.x;
        state->y = s.y;
        if (s.frame == g_frame) {
            state->moved = true;
            if (s.action == 'L') state->pressed |= 1;
            if (s.action == 'R') state->pressed |= 4;
            if (s.action == 'U') state->wheel += 1.0f;
            if (s.action == 'D') state->wheel -= 1.0f;
        }
    }
    return true;
}

}  // namespace ps2host
