#version 450
// The setup within the pass checked (gs_renderer.cpp's date_in_pass_ 2):
// drawn where the stencil is the value the test sets, each pixel whose
// alpha bit (read as date.frag reads it) says otherwise is counted.

layout(early_fragment_tests) in;

layout(set = 0, binding = 0) uniform sampler2D target;
layout(std430, set = 1, binding = 0) buffer Counts { uint differ; };

layout(push_constant) uniform Params {
    layout(offset = 16) uint expect;    // the alpha bit a pixel of this stencil value has
} p;

void main() {
    float alpha = texelFetch(target, ivec2(gl_FragCoord.xy), 0).a * 255.0;
    uint bit = alpha >= 127.5 ? 1u : 0u;
    if (bit != p.expect) atomicAdd(differ, 1u);
}
