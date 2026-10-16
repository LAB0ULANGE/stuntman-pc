#version 450
// The destination alpha test's setup pass: where the target's alpha bit (the
// top bit of its alpha, 0x80 and up) equals DATM, the pixel survives and the
// stencil is set; the draw that follows is tested against it.

layout(set = 0, binding = 0) uniform sampler2D target;

layout(push_constant) uniform Params {
    layout(offset = 16) uint datm;
} p;

void main() {
    float alpha = texelFetch(target, ivec2(gl_FragCoord.xy), 0).a * 255.0;
    uint bit = alpha >= 127.5 ? 1u : 0u;
    if (bit != p.datm) discard;
}
