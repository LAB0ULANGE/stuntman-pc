/*
 * The game disc, read from the user's own image of it: 2048-byte sectors and the
 * ISO 9660 file system on them, as the CD/DVD services the game calls (through
 * the IOP) need them. Nothing from the disc is part of a port.
 *
 * An image is an .iso (2048-byte sectors, a DVD's usual form) or a raw CD image
 * (.bin of 2352-byte sectors: sync, header, Mode 1 data at 16 or Mode 2 data at
 * 24), opened directly or through the .cue sheet naming it.
 */

#ifndef PS2HOST_DISC_H
#define PS2HOST_DISC_H

#include <cstdio>
#include <string>
#include <vector>

#include "ps2rt/types.h"

namespace ps2host {

constexpr u32 kSectorSize = 2048;
constexpr u32 kRawSectorSize = 2352;

class Disc {
public:
    struct File {
        u32 lsn = 0;        // first sector
        u32 size = 0;       // bytes
        std::string name;   // its directory record's identifier ("CARS.BIN;1"); "" for the root
        u8 date[7] = {};    // the record's: years since 1900, month, day, hour, minute, second, GMT offset
    };

    ~Disc();

    /* Open an image file: an .iso, a raw .bin, or a .cue naming one. */
    bool Open(const std::string& path);
    /* Use an image held in memory instead (the host's own tests): 2048-byte
       sectors, or raw 2352-byte ones when it starts with a sector's sync. */
    void Use(std::vector<u8> image);
    bool IsOpen() const { return file_ != nullptr || !memory_.empty(); }
    /* A CD rather than a DVD: a raw image, or an .iso small enough for a CD. */
    bool IsCd() const;

    /* Read `count` sectors' 2048 data bytes from `lsn`; false past the end. */
    bool Read(u32 lsn, u32 count, u8* out);
    /* Read sectors as sceCdRead's data pattern asks: 2048 bytes (0), 2328 (1:
       Mode 2 data with the form 2 area) or 2340 (2: header, subheader, data).
       An .iso has no raw bytes: the header is made, the subheader zero. */
    bool ReadPattern(u32 lsn, u32 count, u32 pattern, u8* out);
    static u32 PatternSize(u32 pattern) { return pattern == 1 ? 2328 : pattern == 2 ? 2340 : kSectorSize; }

    /* Look a path up - "\\DATA\\CARS.BIN;1" as the SDK writes it; separators may
       be either slash, the version suffix is optional, case does not matter. */
    bool Find(const std::string& path, File* out);
    /* A whole file's bytes (a path as Find takes it); false if it is not there. */
    bool ReadFile(const std::string& path, std::vector<u8>* out);
    /* The executable the console boots: SYSTEM.CNF's BOOT2 line, as a path on
       the disc ("\\SLUS_210.50;1"); "" when the image has none (not a PS2 disc). */
    std::string BootPath();

private:
    void DetectFormat();
    bool ReadBytes(u64 offset, u64 length, u8* out);

    std::FILE* file_ = nullptr;
    std::vector<u8> memory_;
    u64 size_ = 0;
    u32 data_offset_ = 0;       // within a raw sector: 16 (Mode 1), 24 (Mode 2); 0 for an .iso
    u32 sectors_ = 0;
};

/* The disc the running game reads. */
Disc& GuestDisc();

}  // namespace ps2host

#endif  /* PS2HOST_DISC_H */
