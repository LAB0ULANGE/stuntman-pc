#version 450
#extension GL_GOOGLE_include_directive : require
// A depth buffer from GS local memory on the GPU, drawn: one triangle over
// the image (date.vert's), each pixel's Z read from memory as the swizzle
// reads it and written as the depth (Z / 2^32) - a depth image cannot be a
// storage image, and through a linear buffer a 3x depth buffer's copy was
// most of the GPU's time. Pixels outside the rows asked for are left alone.

layout(std430, set = 0, binding = 0) readonly buffer Memory { uint mem[]; };
layout(push_constant) uniform Params {
    uint bp;            // base, in 256-byte blocks
    uint bw;            // width, in 64-pixel units
    uint psm;
    uint cols, rows;    // the rectangle moved, rows from row0
    uint stride;        // (the swizzle's)
    uint to_memory;     // (the swizzle's)
    uint depth;         // (the swizzle's)
    uint row0;          // the first row
    uint scale;         // the image's pixels to a GS pixel, each way (the internal resolution)
} p;

#include "gs_address.glsl"

void main() {
    const uint k = max(p.scale, 1u);
    const uvec2 at = uvec2(gl_FragCoord.xy) / k;
    if (at.x >= p.cols || at.y < p.row0 || at.y >= p.row0 + p.rows) discard;
    const uint a = Address(at.x, at.y);
    const uint w = mem[a >> 2];
    uint v;
    if (Is16()) v = (w >> ((a & 2u) * 8u)) & 0xFFFFu;
    else if (p.psm == PSMCT24 || p.psm == PSMZ24) v = w & 0xFFFFFFu;
    else v = w;
    gl_FragDepth = float(v) * (1.0 / 4294967296.0);
}
