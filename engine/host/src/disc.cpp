#include "ps2host/disc.h"

#include <cctype>
#include <cstring>
#include <string>

namespace ps2host {

namespace {

u32 Le32(const u8* p) { return u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16 | u32(p[3]) << 24; }

std::string Upper(std::string s) {
    for (char& ch : s) ch = char(std::toupper(static_cast<unsigned char>(ch)));
    return s;
}

// A directory record's name without its version (";1") and trailing dot.
std::string Bare(std::string name) {
    const auto semicolon = name.find(';');
    if (semicolon != std::string::npos) name.resize(semicolon);
    if (!name.empty() && name.back() == '.') name.pop_back();
    return Upper(name);
}

}  // namespace

Disc::~Disc() {
    if (file_) std::fclose(file_);
}

namespace {

// The image a .cue sheet names in its first FILE line, beside the sheet.
std::string CueImage(const std::string& cue_path) {
    std::FILE* cue = std::fopen(cue_path.c_str(), "rb");
    if (!cue) return "";
    std::string text;
    char chunk[4096];
    for (std::size_t n; (n = std::fread(chunk, 1, sizeof chunk, cue)) > 0;) text.append(chunk, n);
    std::fclose(cue);
    for (std::size_t at = 0; at < text.size();) {
        std::size_t end = text.find_first_of("\r\n", at);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(at, end - at);
        at = end + 1;
        line.erase(0, line.find_first_not_of(" \t"));
        if (Upper(line.substr(0, 5)) != "FILE ") continue;
        std::string name = line.substr(5);
        name.erase(0, name.find_first_not_of(" \t"));
        if (!name.empty() && name[0] == '"') {
            const std::size_t close = name.find('"', 1);
            name = name.substr(1, close == std::string::npos ? std::string::npos : close - 1);
        } else {
            name = name.substr(0, name.find_first_of(" \t"));
        }
        const std::size_t slash = cue_path.find_last_of("/\\");
        return slash == std::string::npos ? name : cue_path.substr(0, slash + 1) + name;
    }
    return "";
}

const u8 kSync[12] = {0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00};

u8 ToBcd(u32 v) { return u8((v / 10) << 4 | (v % 10)); }

}  // namespace

bool Disc::Open(const std::string& path) {
    if (file_) std::fclose(file_);
    file_ = nullptr;
    memory_.clear();
    std::string image = path;
    if (image.size() > 4 && Upper(image.substr(image.size() - 4)) == ".CUE") {
        image = CueImage(path);
        if (image.empty()) return false;
    }
    file_ = std::fopen(image.c_str(), "rb");
    if (!file_) return false;
    _fseeki64(file_, 0, SEEK_END);
    size_ = u64(_ftelli64(file_));
    DetectFormat();
    return true;
}

void Disc::Use(std::vector<u8> image) {
    memory_ = std::move(image);
    size_ = memory_.size();
    DetectFormat();
}

// A raw image: every sector starts with the sync pattern, and sector 16 - the
// volume descriptor, on every PS2 disc - says the mode.
void Disc::DetectFormat() {
    data_offset_ = 0;
    u8 head[16];
    if (size_ >= 17 * u64(kRawSectorSize) && ReadBytes(16 * u64(kRawSectorSize), 16, head) &&
        std::memcmp(head, kSync, 12) == 0 && (head[15] == 1 || head[15] == 2))
        data_offset_ = head[15] == 1 ? 16 : 24;
    sectors_ = u32(size_ / (data_offset_ ? kRawSectorSize : kSectorSize));
}

bool Disc::IsCd() const {
    // A CD holds at most about 360,000 sectors (80 minutes); a DVD image is raw-less.
    return data_offset_ != 0 || sectors_ <= 360000;
}

bool Disc::ReadBytes(u64 offset, u64 length, u8* out) {
    if (offset + length > size_) return false;
    if (!memory_.empty()) {
        std::memcpy(out, memory_.data() + offset, size_t(length));
        return true;
    }
    if (!file_ || _fseeki64(file_, s64(offset), SEEK_SET) != 0) return false;
    return std::fread(out, 1, size_t(length), file_) == length;
}

bool Disc::Read(u32 lsn, u32 count, u8* out) {
    if (u64(lsn) + count > sectors_) return false;
    if (!data_offset_) return ReadBytes(u64(lsn) * kSectorSize, u64(count) * kSectorSize, out);
    // Raw: read the whole run at once, then keep each sector's data.
    std::vector<u8> raw(std::size_t(count) * kRawSectorSize);
    if (!ReadBytes(u64(lsn) * kRawSectorSize, raw.size(), raw.data())) return false;
    for (u32 i = 0; i < count; ++i)
        std::memcpy(out + std::size_t(i) * kSectorSize, &raw[std::size_t(i) * kRawSectorSize + data_offset_],
                    kSectorSize);
    return true;
}

bool Disc::ReadPattern(u32 lsn, u32 count, u32 pattern, u8* out) {
    if (pattern != 1 && pattern != 2) return Read(lsn, count, out);
    if (u64(lsn) + count > sectors_) return false;
    const u32 size = PatternSize(pattern);
    // 2340 starts at the header (offset 12), 2328 after the subheader (24).
    const u32 from = pattern == 2 ? 12 : 24;
    std::vector<u8> raw(kRawSectorSize);
    for (u32 i = 0; i < count; ++i) {
        u8* dst = out + std::size_t(i) * size;
        if (data_offset_) {
            if (!ReadBytes(u64(lsn + i) * kRawSectorSize, kRawSectorSize, raw.data())) return false;
            std::memcpy(dst, &raw[from], size);
            continue;
        }
        // An .iso: a Mode 2 Form 1 sector made around its data.
        std::memset(raw.data(), 0, raw.size());
        const u32 address = lsn + i + 150;          // the 2-second lead-in
        raw[12] = ToBcd(address / 75 / 60);
        raw[13] = ToBcd(address / 75 % 60);
        raw[14] = ToBcd(address % 75);
        raw[15] = 2;
        raw[18] = raw[22] = 0x08;                   // subheader submode: data
        if (!Read(lsn + i, 1, &raw[24])) return false;
        std::memcpy(dst, &raw[from], size);
    }
    return true;
}

bool Disc::Find(const std::string& path, File* out) {
    // The primary volume descriptor, sector 16, holds the root directory's record.
    std::vector<u8> sector(kSectorSize);
    if (!Read(16, 1, sector.data()) || sector[0] != 1 || std::memcmp(&sector[1], "CD001", 5) != 0)
        return false;
    File directory{Le32(&sector[156 + 2]), Le32(&sector[156 + 10])};
    std::string rest = path;
    for (char& ch : rest) if (ch == '/') ch = '\\';
    while (!rest.empty()) {
        while (!rest.empty() && rest[0] == '\\') rest.erase(0, 1);
        if (rest.empty()) break;
        const auto end = rest.find('\\');
        const std::string part = Bare(rest.substr(0, end));
        rest = end == std::string::npos ? "" : rest.substr(end);
        // Scan the directory's records for this part.
        const u32 sectors = (directory.size + kSectorSize - 1) / kSectorSize;
        std::vector<u8> data(size_t(sectors) * kSectorSize);
        if (!Read(directory.lsn, sectors, data.data())) return false;
        bool found = false;
        for (u32 at = 0; at < directory.size;) {
            const u8 length = data[at];
            if (length == 0) {                  // records never cross a sector boundary
                at = (at / kSectorSize + 1) * kSectorSize;
                continue;
            }
            const u8 name_length = data[at + 32];
            const std::string name(reinterpret_cast<const char*>(&data[at + 33]), name_length);
            if (name_length > 1 || (name_length == 1 && name[0] > 1)) {
                if (Bare(name) == part) {
                    directory = File{Le32(&data[at + 2]), Le32(&data[at + 10])};
                    found = true;
                    break;
                }
            }
            at += length;
        }
        if (!found) return false;
    }
    *out = directory;
    return true;
}

bool Disc::ReadFile(const std::string& path, std::vector<u8>* out) {
    File file;
    if (!Find(path, &file)) return false;
    out->assign(size_t(file.size + kSectorSize - 1) / kSectorSize * kSectorSize, 0);
    if (!Read(file.lsn, u32(out->size() / kSectorSize), out->data())) return false;
    out->resize(file.size);
    return true;
}

std::string Disc::BootPath() {
    std::vector<u8> cnf;
    if (!ReadFile("SYSTEM.CNF", &cnf)) return "";
    const std::string text(cnf.begin(), cnf.end());
    for (std::size_t at = 0; at < text.size();) {
        std::size_t end = text.find_first_of("\r\n", at);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(at, end - at);
        at = end + 1;
        // BOOT2 = cdrom0:\SLUS_210.50;1
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) continue;
        std::string key = line.substr(0, equals);
        key.erase(key.find_last_not_of(" \t") + 1);
        if (Upper(key) != "BOOT2") continue;
        std::string value = line.substr(equals + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        const std::size_t colon = value.find(':');
        return colon == std::string::npos ? value : value.substr(colon + 1);
    }
    return "";
}

Disc& GuestDisc() {
    static Disc disc;
    return disc;
}

}  // namespace ps2host
