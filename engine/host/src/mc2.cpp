/*
 * libmc2's handlers over a directory (mc2.h).
 */

#include "ps2host/mc2.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

#include "ps2host/memory.h"
#include "ps2rt/generated.h"

namespace ps2host {
namespace mc2 {

namespace {

namespace fs = std::filesystem;

constexpr u32 kFound = 0x81010011;        // EEXIST: the lookup's "here it is"
constexpr u32 kNoEntry = 0x81010002;      // ENOENT
constexpr u32 kNotDirectory = 0x81010014; // ENOTDIR
constexpr u32 kInvalid = 0x81010016;      // EINVAL
constexpr u32 kNoSpace = 0x8101001C;      // ENOSPC
constexpr u32 kBadName = 0x8101005B;
constexpr u32 kTooDeep = 0x81019004;      // a directory under a directory under the root
constexpr u32 kCardKb = 8000;             // an 8 MB card's clusters, less its file system
constexpr u16 kDirectoryMode = 0x8427, kFileMode = 0x8417, kWritten = 0x80;

std::string& Directory() {
    static std::string directory;
    return directory;
}

std::string GuestString(u32 address, u32 limit = 0x80) {
    std::string text;
    for (u32 i = 0; i < limit; ++i) {
        const char ch = char(generated::ReadU8(address + i));
        if (!ch) break;
        text += ch;
    }
    return text;
}

void CopyToGuest(u32 address, const u8* bytes, std::size_t size) {
    if (u8* p = GuestMemory().Translate(address, u32(size))) {
        std::memcpy(p, bytes, size);
        return;
    }
    for (std::size_t i = 0; i < size; ++i) generated::Write8(address + u32(i), bytes[i]);
}

void CopyFromGuest(u32 address, u8* bytes, std::size_t size) {
    if (const u8* p = GuestMemory().Translate(address, u32(size))) {
        std::memcpy(bytes, p, size);
        return;
    }
    for (std::size_t i = 0; i < size; ++i) bytes[i] = u8(generated::ReadU8(address + u32(i)));
}

// The card's path for `name` (relative to the root, '/' separated): its
// components, none of them "..". False for a name that is no path.
bool Components(const std::string& name, std::vector<std::string>* parts) {
    std::string part;
    for (std::size_t i = 0; i <= name.size(); ++i) {
        if (i == name.size() || name[i] == '/') {
            if (part == "..") return false;
            if (!part.empty() && part != ".") parts->push_back(part);
            part.clear();
        } else {
            part += name[i];
        }
    }
    return true;
}

fs::path HostPath(const std::vector<std::string>& parts) {
    fs::path path = fs::u8path(Directory());
    for (const std::string& part : parts) path /= fs::u8path(part);
    return path;
}

// sceMc2StDateTime: reserved, second, minute, hour, day, month, year (u16).
void DateTime(const fs::path& path, u8 out[8]) {
    std::error_code error;
    const auto written = fs::last_write_time(path, error);
    std::time_t when = 0;
    if (!error) {
        // The file clock's time on the system clock (C++17 has no clock_cast).
        const auto system = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            written - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
        when = std::chrono::system_clock::to_time_t(system);
    }
    std::tm local{};
    localtime_s(&local, &when);
    const u32 year = u32(local.tm_year + 1900);
    const u8 bytes[8] = {0, u8(local.tm_sec), u8(local.tm_min), u8(local.tm_hour), u8(local.tm_mday),
                         u8(local.tm_mon + 1), u8(year), u8(year >> 8)};
    std::memcpy(out, bytes, 8);
}

struct Entry {
    std::string name;
    fs::path path;
    bool directory = false;
    u32 size = 0;
    u16 mode() const { return directory ? kDirectoryMode : u16(kFileMode | (size ? kWritten : 0)); }
};

Entry Describe(const fs::path& path) {
    Entry e;
    e.name = path.filename().u8string();
    e.path = path;
    std::error_code error;
    e.directory = fs::is_directory(path, error);
    if (!e.directory) e.size = u32(fs::file_size(path, error));
    return e;
}

u32 UsedKb() {
    u32 kb = 2;                                 // the root's own entries
    std::error_code error;
    for (auto it = fs::recursive_directory_iterator(fs::u8path(Directory()), error);
         !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
        if (it->is_directory(error)) kb += 2;
        else kb += u32((it->file_size(error) + 1023) / 1024);
    }
    return kb;
}

u32 FreeKb() {
    const u32 used = UsedKb();
    return used >= kCardKb ? 0 : kCardKb - used;
}

bool Matches(const char* pattern, const char* name) {
    if (!*pattern) return !*name;
    if (*pattern == '*') return Matches(pattern + 1, name) || (*name && Matches(pattern, name + 1));
    if (!*name) return false;
    return (*pattern == '?' || *pattern == *name) && Matches(pattern + 1, name + 1);
}

// The directory's entries in a fixed order (by name): a real card lists them
// in the order they were made, which a directory on the host does not keep.
std::vector<Entry> List(const fs::path& directory) {
    std::vector<Entry> entries;
    std::error_code error;
    for (auto it = fs::directory_iterator(directory, error); !error && it != fs::directory_iterator();
         it.increment(error))
        entries.push_back(Describe(it->path()));
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.name < b.name; });
    return entries;
}

// SceMc2DirParam: created +0, modified +8, size +0x10, attributes +0x14, name +0x18.
void WriteDirParam(u32 address, const Entry& e) {
    u8 bytes[0x38] = {};
    DateTime(e.path, bytes);
    std::memcpy(bytes + 8, bytes, 8);           // the host keeps one time: modified stands for both
    const u32 size = e.size;
    std::memcpy(bytes + 0x10, &size, 4);
    const u16 mode = e.mode();
    std::memcpy(bytes + 0x14, &mode, 2);
    std::memcpy(bytes + 0x18, e.name.c_str(), std::min<std::size_t>(e.name.size(), 31));
    CopyToGuest(address, bytes, sizeof bytes);
}

// The common start: the name, as a path on the card. A result other than 0
// is the request's answer.
u32 Resolve(lifted::Context& c, std::vector<std::string>* parts) {
    const std::string name = GuestString(u32(c.r[5]), 0x80);
    if (!Components(name, parts)) return kInvalid;
    std::error_code error;
    fs::create_directories(fs::u8path(Directory()), error);
    return 0;
}

}  // namespace

const std::string& CardDirectory() { return Directory(); }
void SetCardDirectory(const std::string& directory) { Directory() = directory; }

void GetInfo(lifted::Context& c) {
    const u32 info = u32(c.r[5]);
    std::error_code error;
    fs::create_directories(fs::u8path(Directory()), error);
    if (info) {
        generated::Write32(info, 2);            // a PS2 memory card
        generated::Write32(info + 4, 1);        // formatted
        generated::Write32(info + 8, FreeKb());
    }
    c.r[2] = 0;
}

void Format(lifted::Context& c) { c.r[2] = 0; }

void ReadFile(lifted::Context& c) {
    std::vector<std::string> parts;
    if (const u32 refused = Resolve(c, &parts)) { c.r[2] = refused; return; }
    const u32 buffer = u32(c.r[6]), offset = u32(c.r[7]);
    u32 size = u32(c.r[8]);
    const fs::path path = HostPath(parts);
    std::error_code error;
    if (parts.empty() || !fs::exists(path, error) || fs::is_directory(path, error)) { c.r[2] = kNoEntry; return; }
    const u32 length = u32(fs::file_size(path, error));
    if (offset > length) { c.r[2] = kInvalid; return; }
    if (size == 0 || offset + size > length) size = length - offset;
    std::vector<u8> bytes(size);
    std::ifstream file(path, std::ios::binary);
    file.seekg(offset);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    CopyToGuest(buffer, bytes.data(), bytes.size());
    c.r[2] = size;
}

void WriteFile(lifted::Context& c) {
    std::vector<std::string> parts;
    if (const u32 refused = Resolve(c, &parts)) { c.r[2] = refused; return; }
    const u32 buffer = u32(c.r[6]), offset = u32(c.r[7]), size = u32(c.r[8]);
    const fs::path path = HostPath(parts);
    std::error_code error;
    if (parts.empty() || !fs::exists(path, error) || fs::is_directory(path, error)) { c.r[2] = kNoEntry; return; }
    const u32 length = u32(fs::file_size(path, error));
    if (offset > length) { c.r[2] = kInvalid; return; }
    const u32 grows = offset + size > length ? (offset + size + 1023) / 1024 - (length + 1023) / 1024 : 0;
    if (grows > FreeKb()) { c.r[2] = kNoSpace; return; }
    std::vector<u8> bytes(size);
    CopyFromGuest(buffer, bytes.data(), bytes.size());
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(offset);
    file.write(reinterpret_cast<const char*>(bytes.data()), size);
    c.r[2] = size;
}

void Create(lifted::Context& c) {
    std::vector<std::string> parts;
    if (const u32 refused = Resolve(c, &parts)) { c.r[2] = refused; return; }
    const u32 mode = u32(c.r[6]) ? u32(c.r[6]) : kFileMode;
    const bool directory = (mode & 0x20) != 0;
    if (parts.empty() || parts.back().size() > 31) { c.r[2] = kBadName; return; }
    if (directory && parts.size() > 1) { c.r[2] = kTooDeep; return; }
    const fs::path path = HostPath(parts);
    std::error_code error;
    if (fs::exists(path, error)) { c.r[2] = kFound; return; }
    if (!fs::is_directory(path.parent_path(), error)) { c.r[2] = kNoEntry; return; }
    if (FreeKb() < (directory ? 2u : 1u)) { c.r[2] = kNoSpace; return; }
    if (directory) fs::create_directory(path, error);
    else std::ofstream(path, std::ios::binary);
    c.r[2] = 0;
}

void GetDir(lifted::Context& c) {
    const std::string name = GuestString(u32(c.r[5]), 0x80);
    const u32 offset = u32(c.r[6]);
    const s32 maxent = s32(c.r[7]);
    const u32 entries = u32(c.r[8]), count = u32(c.r[9]);
    const std::size_t slash = name.rfind('/');
    const std::string directory = slash == std::string::npos ? "" : name.substr(0, slash);
    const std::string pattern = slash == std::string::npos ? name : name.substr(slash + 1);
    std::vector<std::string> parts;
    if (pattern.empty() || !Components(directory, &parts)) { c.r[2] = kInvalid; return; }
    const fs::path path = HostPath(parts);
    std::error_code error;
    fs::create_directories(fs::u8path(Directory()), error);
    if (!fs::exists(path, error)) { c.r[2] = kNoEntry; return; }
    if (!fs::is_directory(path, error)) { c.r[2] = kNotDirectory; return; }
    const std::vector<Entry> listed = List(path);
    u32 matched = 0;
    for (std::size_t i = offset; i < listed.size(); ++i) {
        if (!Matches(pattern.c_str(), listed[i].name.c_str())) continue;
        if (maxent > 0) {
            if (matched >= u32(maxent)) break;
            WriteDirParam(entries + matched * 0x38, listed[i]);
        }
        ++matched;
    }
    if (count) generated::Write32(count, matched);
    c.r[2] = 0;
}

void Lookup(lifted::Context& c) {
    std::vector<std::string> parts;
    if (const u32 refused = Resolve(c, &parts)) { c.r[2] = refused; return; }
    const u32 entry = u32(c.r[6]);
    const fs::path path = HostPath(parts);
    std::error_code error;
    if (parts.empty() || !fs::exists(path, error)) { c.r[2] = kNoEntry; return; }
    const Entry e = Describe(path);
    if (entry) {
        u8 bytes[0x60] = {};
        const u16 mode = e.mode();
        std::memcpy(bytes, &mode, 2);
        std::memcpy(bytes + 4, &e.size, 4);
        DateTime(e.path, bytes + 8);
        DateTime(e.path, bytes + 0x18);
        std::memcpy(bytes + 0x40, e.name.c_str(), std::min<std::size_t>(e.name.size(), 31));
        CopyToGuest(entry, bytes, sizeof bytes);
    }
    c.r[2] = kFound;
}

}  // namespace mc2
}  // namespace ps2host
