// GS local memory's layout: a pixel's byte address (GS User's Manual,
// chapter 8; the tables are engine/host/src/gs_memory.cpp's). The including
// shader declares push constants `p` with bp (base, 256-byte blocks), bw
// (width, 64-pixel units) and psm.
const uint PSMCT32 = 0x00u, PSMCT24 = 0x01u, PSMCT16 = 0x02u, PSMCT16S = 0x0Au;
const uint PSMZ32 = 0x30u, PSMZ24 = 0x31u, PSMZ16 = 0x32u, PSMZ16S = 0x3Au;

const uint kBlock32[32] = uint[](0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23,
                                 8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31);
const uint kBlockZ32[32] = uint[](24, 25, 28, 29, 8, 9, 12, 13, 26, 27, 30, 31, 10, 11, 14, 15,
                                  16, 17, 20, 21, 0, 1, 4, 5, 18, 19, 22, 23, 2, 3, 6, 7);
const uint kBlock16[32] = uint[](0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15,
                                 16, 18, 24, 26, 17, 19, 25, 27, 20, 22, 28, 30, 21, 23, 29, 31);
const uint kBlock16S[32] = uint[](0, 2, 16, 18, 1, 3, 17, 19, 8, 10, 24, 26, 9, 11, 25, 27,
                                  4, 6, 20, 22, 5, 7, 21, 23, 12, 14, 28, 30, 13, 15, 29, 31);
const uint kBlockZ16[32] = uint[](24, 26, 16, 18, 25, 27, 17, 19, 28, 30, 20, 22, 29, 31, 21, 23,
                                  8, 10, 0, 2, 9, 11, 1, 3, 12, 14, 4, 6, 13, 15, 5, 7);
const uint kBlockZ16S[32] = uint[](24, 26, 8, 10, 25, 27, 9, 11, 16, 18, 0, 2, 17, 19, 1, 3,
                                   28, 30, 12, 14, 29, 31, 13, 15, 20, 22, 4, 6, 21, 23, 5, 7);
const uint kColumn32[64] = uint[](0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                  16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31,
                                  32, 33, 36, 37, 40, 41, 44, 45, 34, 35, 38, 39, 42, 43, 46, 47,
                                  48, 49, 52, 53, 56, 57, 60, 61, 50, 51, 54, 55, 58, 59, 62, 63);
const uint kColumn16[128] = uint[](
    0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27,
    4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31,
    32, 34, 40, 42, 48, 50, 56, 58, 33, 35, 41, 43, 49, 51, 57, 59,
    36, 38, 44, 46, 52, 54, 60, 62, 37, 39, 45, 47, 53, 55, 61, 63,
    64, 66, 72, 74, 80, 82, 88, 90, 65, 67, 73, 75, 81, 83, 89, 91,
    68, 70, 76, 78, 84, 86, 92, 94, 69, 71, 77, 79, 85, 87, 93, 95,
    96, 98, 104, 106, 112, 114, 120, 122, 97, 99, 105, 107, 113, 115, 121, 123,
    100, 102, 108, 110, 116, 118, 124, 126, 101, 103, 109, 111, 117, 119, 125, 127);

const uint kMask = (4u << 20) - 1u;

uint Page(uint px, uint py) { return p.bp + 32u * (py * max(p.bw, 1u) + px); }

bool Is16() { return p.psm == PSMCT16 || p.psm == PSMCT16S || p.psm == PSMZ16 || p.psm == PSMZ16S; }

// The byte address of pixel (x, y).
uint Address(uint x, uint y) {
    if (Is16()) {
        const uint i = ((y % 64u) / 8u) * 4u + (x % 64u) / 16u;
        uint b;
        if (p.psm == PSMCT16) b = kBlock16[i];
        else if (p.psm == PSMCT16S) b = kBlock16S[i];
        else if (p.psm == PSMZ16) b = kBlockZ16[i];
        else b = kBlockZ16S[i];
        const uint block = Page(x / 64u, y / 64u) + b;
        return (block * 256u + kColumn16[(y % 8u) * 16u + x % 16u] * 2u) & kMask;
    }
    const uint i = ((y % 32u) / 8u) * 8u + (x % 64u) / 8u;
    const uint b = (p.psm == PSMZ32 || p.psm == PSMZ24) ? kBlockZ32[i] : kBlock32[i];
    const uint block = Page(x / 64u, y / 32u) + b;
    return (block * 256u + kColumn32[(y % 8u) * 8u + x % 8u] * 4u) & kMask;
}
