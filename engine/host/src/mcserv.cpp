/*
 * libmc's memory card server (mcserv, RPC 0x80000400) over a directory: the
 * same card mc2.h keeps for libmc2's games (--memcard DIR; none given, no
 * card is in the slot).
 *
 * The protocol, as Mister Mosquito's libmc 2.3 speaks it (EE 0x0010DA80-
 * 0x0010F000). Each call is NOWAIT with a 0x30-byte request (0x414 when it
 * carries a path, at +0x14) and a one-word answer, the result; libmc's
 * sceMcSync finishes the call, and for some an end function copies what the
 * module DMA'd into an EE buffer of the library's:
 *
 *   0xFE  init: answer three words - result, mcserv's and mcman's versions
 *         (libmc wants at least 0x20A and 0x20E)
 *   0x01  getinfo: +4 port, +8 slot, +0xC/+0x10/+0x14 whether format, free
 *         and type are wanted, +0x1C the EE buffer they go to (+0 type,
 *         +4 free KB, +0x90 format); result 0 (the same card), -1 (a new,
 *         formatted card), -10 (no card)
 *   0x02  open: +0 port, +4 slot, +8 mode (1 read, 2 write, 3 both, 0x200
 *         create, 0x400 truncate; 0x40 makes a directory), +0x14 path;
 *         result a descriptor
 *   0x03  close: +0 fd           0x0A  flush: +0 fd
 *   0x04  seek: +0 fd, +0x10 offset, +0x14 origin (0 set, 1 current, 2 end)
 *   0x05  read: +0 fd, +0xC size, +0x18 EE buffer, +0x1C a record for the
 *         buffer's unaligned ends { size1, size2, dest1, dest2, bytes... }
 *         that the end function copies out - here every byte goes straight
 *         to the buffer and the record says none
 *   0x06  write: +0 fd, +0x14 the bytes before the buffer's 16-byte
 *         alignment, sent at +0x20, then +0xC bytes from EE address +0x18
 *   0x0C  chdir: +0 port, +4 slot, +0x10 an EE buffer for the directory left,
 *         +0x14 path
 *   0x0D  getdir: +0 port, +4 slot, +8 0 to start (1 to go on), +0xC entries
 *         at most, +0x10 the EE table (64 bytes an entry: created, modified,
 *         size, attributes, name at +0x20), +0x14 a path with * and ?
 *   0x0E  setfileinfo (accepted)  0x0F  delete: +0x14 path
 *   0x10  format (the directory is always formatted)
 *   0x14  accepted              0x15  slots on a port: 1
 *
 * Errors: -4 no such entry, -5 refused, -7 too many open.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include "ps2host/host.h"
#include "ps2host/mc2.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

namespace fs = std::filesystem;

constexpr s32 kNoCard = -10, kNewFormatted = -1, kNoEntry = -4, kRefused = -5, kTooMany = -7;
constexpr u32 kCardKb = 8000;

struct Open {
    fs::path path;
    u32 position = 0;
    bool write = false;
};

struct Card {
    std::map<s32, Open> files;
    std::string current = "/";
    bool seen = false;                      // a getinfo has reported the card once
    std::vector<fs::path> listing;          // getdir's matches, for "go on"
    std::size_t listed = 0;
};

Card& C() {
    static Card card;
    return card;
}

u32 At(const std::vector<u8>& data, std::size_t offset) {
    u32 value = 0;
    if (offset + 4 <= data.size()) std::memcpy(&value, &data[offset], 4);
    return value;
}

std::vector<u8> Result(s32 value) {
    std::vector<u8> out(4);
    std::memcpy(out.data(), &value, 4);
    return out;
}

std::string PathAt(const std::vector<u8>& data, std::size_t offset) {
    std::string path;
    for (std::size_t i = offset; i < data.size() && data[i]; ++i) path += char(data[i]);
    return path;
}

bool Present(u32 port, u32 slot) { return !mc2::CardDirectory().empty() && port == 0 && slot == 0; }

// A card path (absolute, or from the current directory) as a host path under
// the card's directory; false for one that would leave it.
bool HostPath(const std::string& card_path, fs::path* out) {
    std::string full = !card_path.empty() && card_path[0] == '/' ? card_path : C().current + "/" + card_path;
    std::vector<std::string> parts;
    std::size_t at = 0;
    while (at <= full.size()) {
        std::size_t end = full.find('/', at);
        if (end == std::string::npos) end = full.size();
        const std::string part = full.substr(at, end - at);
        at = end + 1;
        if (part.empty() || part == ".") continue;
        if (part == "..") {
            if (parts.empty()) return false;
            parts.pop_back();
            continue;
        }
        parts.push_back(part);
    }
    fs::path path = fs::u8path(mc2::CardDirectory());
    for (const std::string& p : parts) path /= fs::u8path(p);
    *out = path;
    return true;
}

bool Matches(const char* pattern, const char* name) {
    if (!*pattern) return !*name;
    if (*pattern == '*') return Matches(pattern + 1, name) || (*name && Matches(pattern, name + 1));
    if (!*name) return false;
    return (*pattern == '?' || *pattern == *name) && Matches(pattern + 1, name + 1);
}

void PutDate(u8* out, const fs::path& path) {
    std::error_code error;
    std::time_t when = std::time(nullptr);
    const auto stamp = fs::last_write_time(path, error);
    if (!error)
        when = std::chrono::system_clock::to_time_t(
            std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                stamp - fs::file_time_type::clock::now() + std::chrono::system_clock::now()));
    std::tm t{};
    localtime_s(&t, &when);
    out[0] = 0;
    out[1] = u8(t.tm_sec);
    out[2] = u8(t.tm_min);
    out[3] = u8(t.tm_hour);
    out[4] = u8(t.tm_mday);
    out[5] = u8(t.tm_mon + 1);
    const u16 year = u16(t.tm_year + 1900);
    std::memcpy(out + 6, &year, 2);
}

// A getdir entry: created, modified, size, attributes, name.
void PutEntry(u32 address, const fs::path& path, const std::string& name) {
    std::error_code error;
    u8 entry[64] = {};
    PutDate(entry, path);
    PutDate(entry + 8, path);
    const bool directory = fs::is_directory(path, error);
    const u32 size = directory ? 0 : u32(fs::file_size(path, error));
    const u16 attributes = directory ? 0x8427 : 0x8417;
    std::memcpy(entry + 0x10, &size, 4);
    std::memcpy(entry + 0x14, &attributes, 2);
    std::memcpy(entry + 0x20, name.c_str(), std::min<std::size_t>(name.size(), 31));
    Memory& m = GuestMemory();
    for (u32 i = 0; i < 64; ++i) m.Write(address + i, entry[i], 1);
}

u32 UsedKb() {
    std::error_code error;
    u64 bytes = 0;
    for (const auto& e : fs::recursive_directory_iterator(fs::u8path(mc2::CardDirectory()), error))
        if (e.is_regular_file(error)) bytes += (e.file_size(error) + 1023) / 1024 * 1024;
    return u32(bytes / 1024);
}

std::vector<u8> Server(u32 procedure, const std::vector<u8>& data) {
    Card& card = C();
    Memory& m = GuestMemory();
    std::error_code error;
    switch (procedure) {
        case 0xFE: {
            std::vector<u8> out = Result(0);
            const std::vector<u8> serv = Result(0x20A), man = Result(0x20E);
            out.insert(out.end(), serv.begin(), serv.end());
            out.insert(out.end(), man.begin(), man.end());
            return out;
        }
        case 0x01: {
            const u32 port = At(data, 4), slot = At(data, 8), info = At(data, 0x1C);
            const bool present = Present(port, slot);
            if (present) fs::create_directories(fs::u8path(mc2::CardDirectory()), error);
            if (info) {
                m.Write(info + 0, present ? 2 : 0, 4);                     // a PS2 card
                m.Write(info + 4, present ? kCardKb - std::min(kCardKb, UsedKb()) : 0, 4);
                m.Write(info + 0x90, present ? 1 : 0, 4);                  // formatted
            }
            if (!present) return Result(kNoCard);
            const s32 result = card.seen ? 0 : kNewFormatted;
            card.seen = true;
            return Result(result);
        }
        case 0x02: {
            if (!Present(At(data, 0), At(data, 4))) return Result(kNoCard);
            const u32 mode = At(data, 8);
            const std::string name = PathAt(data, 0x14);
            fs::path path;
            if (!HostPath(name, &path)) return Result(kRefused);
            if (mode & 0x40) {                                  // a directory
                if (fs::exists(path, error)) return Result(kNoEntry);    // it is there already
                fs::create_directories(path, error);
                Note("mc mkdir " + name, "MC: made the directory " + name);
                return Result(error ? kRefused : 0);
            }
            const bool exists = fs::is_regular_file(path, error);
            if (!exists && !(mode & 0x200)) return Result(kNoEntry);
            if (!exists || (mode & 0x400)) {
                fs::create_directories(path.parent_path(), error);
                std::ofstream(path, std::ios::binary | std::ios::trunc);
            }
            s32 fd = 0;
            while (card.files.count(fd)) ++fd;
            if (fd >= 32) return Result(kTooMany);
            card.files[fd] = {path, 0, (mode & 2) != 0};
            Note("mc open " + name, "MC: opened " + name);
            return Result(fd);
        }
        case 0x03:
            card.files.erase(s32(At(data, 0)));
            return Result(0);
        case 0x04: {
            auto f = card.files.find(s32(At(data, 0)));
            if (f == card.files.end()) return Result(kNoEntry);
            const s32 offset = s32(At(data, 0x10));
            const u32 origin = At(data, 0x14);
            const s64 size = s64(fs::file_size(f->second.path, error));
            s64 to = origin == 0 ? offset : origin == 1 ? s64(f->second.position) + offset : size + offset;
            if (to < 0) to = 0;
            f->second.position = u32(to);
            return Result(s32(to));
        }
        case 0x05: {
            auto f = card.files.find(s32(At(data, 0)));
            if (f == card.files.end()) return Result(kNoEntry);
            const u32 size = At(data, 0xC), buffer = At(data, 0x18), ends = At(data, 0x1C);
            std::ifstream in(f->second.path, std::ios::binary);
            in.seekg(f->second.position);
            std::vector<char> bytes(size);
            in.read(bytes.data(), size);
            const u32 got = u32(in.gcount());
            for (u32 i = 0; i < got; ++i) m.Write(buffer + i, u8(bytes[i]), 1);
            if (ends)
                for (u32 k = 0; k < 16; ++k) m.Write(ends + k, 0, 1);
            f->second.position += got;
            return Result(s32(got));
        }
        case 0x06: {
            auto f = card.files.find(s32(At(data, 0)));
            if (f == card.files.end()) return Result(kNoEntry);
            const u32 head = At(data, 0x14), size = At(data, 0xC), from = At(data, 0x18);
            std::vector<char> bytes;
            for (u32 i = 0; i < head && 0x20 + i < data.size(); ++i) bytes.push_back(char(data[0x20 + i]));
            for (u32 i = 0; i < size; ++i) bytes.push_back(char(m.Read(from + i, 1)));
            std::fstream out(f->second.path, std::ios::binary | std::ios::in | std::ios::out);
            out.seekp(f->second.position);
            out.write(bytes.data(), std::streamsize(bytes.size()));
            f->second.position += u32(bytes.size());
            Note("mc write " + f->second.path.filename().string(),
                 "MC: wrote " + f->second.path.filename().u8string());
            return Result(s32(bytes.size()));
        }
        case 0x0A: case 0x0E: case 0x10: case 0x14:
            return Result(0);
        case 0x15:
            return Result(1);
        case 0x0C: {
            if (!Present(At(data, 0), At(data, 4))) return Result(kNoCard);
            const u32 left = At(data, 0x10);
            if (left)
                for (u32 i = 0; i <= card.current.size(); ++i)
                    m.Write(left + i, i < card.current.size() ? u8(card.current[i]) : 0, 1);
            const std::string name = PathAt(data, 0x14);
            fs::path path;
            if (!HostPath(name, &path) || !fs::is_directory(path, error)) return Result(kNoEntry);
            std::string next = !name.empty() && name[0] == '/' ? name : card.current + "/" + name;
            card.current = next.empty() ? "/" : next;
            return Result(0);
        }
        case 0x0D: {
            if (!Present(At(data, 0), At(data, 4))) return Result(kNoCard);
            const u32 go_on = At(data, 8), table = At(data, 0x10);
            const s32 most = s32(At(data, 0xC));
            const std::string name = PathAt(data, 0x14);
            if (!go_on) {
                card.listing.clear();
                card.listed = 0;
                const std::size_t slash = name.rfind('/');
                const std::string directory = slash == std::string::npos ? "" : name.substr(0, slash + 1);
                const std::string pattern = slash == std::string::npos ? name : name.substr(slash + 1);
                fs::path path;
                if (!HostPath(directory.empty() ? "." : directory, &path) || !fs::is_directory(path, error))
                    return Result(kNoEntry);
                if (directory != "/" && !directory.empty()) {
                    if (Matches(pattern.c_str(), ".")) card.listing.push_back(path / ".");
                    if (Matches(pattern.c_str(), "..")) card.listing.push_back(path / "..");
                }
                for (const auto& e : fs::directory_iterator(path, error))
                    if (Matches(pattern.c_str(), e.path().filename().u8string().c_str()))
                        card.listing.push_back(e.path());
            }
            s32 count = 0;
            while (card.listed < card.listing.size() && count < most) {
                const fs::path& p = card.listing[card.listed++];
                const std::string entry_name = p.filename().u8string();
                PutEntry(table + 64 * u32(count), entry_name == "." || entry_name == ".." ? p.parent_path() : p,
                         entry_name);
                ++count;
            }
            return Result(count);
        }
        case 0x0F: {
            if (!Present(At(data, 0), At(data, 4))) return Result(kNoCard);
            fs::path path;
            if (!HostPath(PathAt(data, 0x14), &path) || !fs::exists(path, error)) return Result(kNoEntry);
            fs::remove(path, error);
            return Result(error ? kRefused : 0);
        }
        default: {
            char text[96];
            std::snprintf(text, sizeof text, "MC: mcserv procedure 0x%X is not implemented (failed)",
                          unsigned(procedure));
            Note("mcserv " + std::to_string(procedure), text);
            return Result(kRefused);
        }
    }
}

}  // namespace

// PS2HOST_MC_LOG=1: every call and its result.
std::vector<u8> Logged(u32 procedure, const std::vector<u8>& data) {
    std::vector<u8> answer = Server(procedure, data);
    if (std::getenv("PS2HOST_MC_LOG")) {
        s32 result = 0;
        if (answer.size() >= 4) std::memcpy(&result, answer.data(), 4);
        std::fprintf(stderr, "[mc] procedure 0x%X: %08X %08X %08X %08X \"%s\" -> %d\n", unsigned(procedure),
                     unsigned(At(data, 0)), unsigned(At(data, 4)), unsigned(At(data, 8)), unsigned(At(data, 0xC)),
                     PathAt(data, 0x14).c_str(), int(result));
    }
    return answer;
}

void InstallMcserv() { RegisterRpcServer(0x80000400u, Logged); }

}  // namespace ps2host
