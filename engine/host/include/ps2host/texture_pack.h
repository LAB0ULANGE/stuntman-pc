#pragma once
/* Texture packs: PCSX2's texture replacements, used as PCSX2 uses them.

   A pack is a folder of .dds and .png files - in folders of their own if it
   likes; only a file's name counts - each named for the texture it replaces
   as PCSX2 names it (GSTextureReplacements.cpp): XXH3 hashes of the texture's
   GS memory and of its CLUT, its region, and its format and size -

     <texture>-<bits>.dds              <texture>-<clut>-<bits>.dds
     <texture>-r<w>x<h>-<bits>.dds     <texture>-<clut>-r<w>x<h>-<bits>.dds

   hashes in hex, bits PSM | TW << 6 | TH << 10 (TCC at bit 14, ignored) |
   TEXA's TA0 << 15, AEM << 23, TA1 << 24 for 24- and 16-bit formats. The
   hash covers what PCSX2's HashTextureLevel does: a texture's blocks as GS
   memory holds them, or its texels expanded (16- and 24-bit formats, a
   region, one smaller than a block) - so a pack made for PCSX2 replaces the
   same textures here.

   Files are read in the background, the first time their texture is drawn:
   the texture as the game has it until then (a frame or two). */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ps2host/gs.h"

namespace ps2host {
namespace texture_pack {

/* A replacement as the renderer takes it (GsRenderer::UploadReplacement),
   under its file's `id`: RGBA8 (format 0; alpha as the GS's, 0x80 is 1.0, as
   PCSX2's packs have it) or BC1, BC2, BC3, BC7 (1-4) as the DDS stores them;
   `levels` largest first, one after another in `data`. A PNG's levels below
   the first are made here; a DDS has the ones it has (as PCSX2). */
struct Image {
    std::uint64_t id = 0;
    std::uint32_t format = 0, width = 0, height = 0, levels = 1;
    std::vector<std::uint8_t> data;
};
enum Format : std::uint32_t { kRgba8 = 0, kBc1 = 1, kBc2 = 2, kBc3 = 3, kBc7 = 4 };

/* Whether the GPU takes block-compressed images; if not, DDS files are
   decoded to RGBA8 as they are read. */
void SetBlockCompression(bool on);

/* Which texture: PCSX2's TextureName, as its file names spell it. */
struct Name {
    std::uint64_t texture = 0, clut = 0;
    std::uint32_t region_w = 0, region_h = 0;
    std::uint32_t bits = 0;
    bool operator==(const Name& o) const {
        return texture == o.texture && clut == o.clut && region_w == o.region_w && region_h == o.region_h &&
               bits == o.bits;
    }
    std::string Text() const;           // as a file is named, without its extension
};

/* The pack in `directory` (and its folders): the files it has, by name. The
   number of them; 0 for none or no such folder. */
std::size_t Open(const std::string& directory);
std::size_t Count();

/* Whether replacements are drawn (Tab, the "texture_pack" setting). Each
   change bumps Generation: textures the game has already drawn come back as
   it has them, or replaced. */
bool Enabled();
void SetEnabled(bool on);
std::uint64_t Generation();

/* PCSX2's name for the texture TEX0 describes, CLAMP's region of it, with
   TEXA, from `memory` - and `clut`, the GS model's 256 entries (Gs::Clut). */
Name NameOf(const gs::Memory& memory, std::uint64_t tex0, std::uint64_t clamp, std::uint64_t texa,
            const std::uint32_t* clut);

/* The replacement for `name`: null if the pack has none - or while its file
   is still being read (asked for here: `*pending` says so; a later call has
   it). */
std::shared_ptr<const Image> Find(const Name& name, bool* pending);

}  // namespace texture_pack
}  // namespace ps2host
