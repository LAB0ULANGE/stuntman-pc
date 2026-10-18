#version 450
// The destination alpha test's setup within the draws' own pass (the target
// read where it is being drawn: VK_KHR_dynamic_rendering_local_read), as
// date.frag: where the alpha bit equals DATM the pixel survives and the
// stencil is set. DATM 2: every pixel (the stencil cleared first).

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput target;

layout(push_constant) uniform Params {
    layout(offset = 16) uint datm;
} p;

void main() {
    float alpha = subpassLoad(target).a * 255.0;
    uint bit = alpha >= 127.5 ? 1u : 0u;
    if (p.datm < 2u && bit != p.datm) discard;
}
