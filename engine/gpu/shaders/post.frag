#version 450
// The window's picture, scaled from the game's as the blit did, and on the
// way (the display's "antialiasing" and "sharpen" settings):
//   FXAA - an edge's stairs smoothed: where the luma steps, the edge's
//   direction and its two ends found along it, and the pixel read that much
//   across it (Timothy Lottes' FXAA 3.11, its quality preset in spirit);
//   sharpening - each pixel against its four neighbours, less where they
//   already differ much (AMD's contrast-adaptive sharpening, simplified).
// Both in the game picture's own texels (`texel`), wherever the window's
// pixel falls in it. Nothing kept between pictures.

layout(set = 0, binding = 0) uniform sampler2D picture;
layout(push_constant) uniform Params {
    vec2 texel;         // one texel of the picture, 0-1
    uint fxaa;          // 1: FXAA
    float sharpen;      // 0 none to 1 most
} p;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

vec3 At(vec2 at) { return textureLod(picture, at, 0.0).rgb; }
float Luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

const float kEdgeMin = 0.0312, kEdgeMax = 0.125, kSubpixel = 0.75;

float Step(int i) { return i < 5 ? 1.0 : i == 5 ? 1.5 : i < 10 ? 2.0 : i == 10 ? 4.0 : 8.0; }

vec3 Fxaa(vec2 at) {
    const vec2 t = p.texel;
    const vec3 center = At(at);
    const float lc = Luma(center);
    const float ld = Luma(At(at + vec2(0.0, t.y))), lu = Luma(At(at - vec2(0.0, t.y)));
    const float ll = Luma(At(at - vec2(t.x, 0.0))), lr = Luma(At(at + vec2(t.x, 0.0)));
    const float lo = min(lc, min(min(ld, lu), min(ll, lr))), hi = max(lc, max(max(ld, lu), max(ll, lr)));
    const float range = hi - lo;
    if (range < max(kEdgeMin, hi * kEdgeMax)) return center;

    const float ldl = Luma(At(at + vec2(-t.x, t.y))), lur = Luma(At(at + vec2(t.x, -t.y)));
    const float lul = Luma(At(at - t)), ldr = Luma(At(at + t));
    const float down_up = ld + lu, left_right = ll + lr;
    const float left_corners = ldl + lul, down_corners = ldl + ldr;
    const float right_corners = ldr + lur, up_corners = lur + lul;
    const float horizontal = abs(-2.0 * ll + left_corners) + abs(-2.0 * lc + down_up) * 2.0 +
                             abs(-2.0 * lr + right_corners);
    const float vertical = abs(-2.0 * lu + up_corners) + abs(-2.0 * lc + left_right) * 2.0 +
                           abs(-2.0 * ld + down_corners);
    const bool is_horizontal = horizontal >= vertical;

    const float l1 = is_horizontal ? lu : ll, l2 = is_horizontal ? ld : lr;
    const float g1 = l1 - lc, g2 = l2 - lc;
    const bool steeper1 = abs(g1) >= abs(g2);
    const float scaled = 0.25 * max(abs(g1), abs(g2));
    float across = is_horizontal ? t.y : t.x;
    float local;
    if (steeper1) {
        across = -across;
        local = 0.5 * (l1 + lc);
    } else {
        local = 0.5 * (l2 + lc);
    }
    vec2 edge = at;
    if (is_horizontal) edge.y += across * 0.5;
    else edge.x += across * 0.5;

    // Along the edge, both ways, to where it ends.
    const vec2 along = is_horizontal ? vec2(t.x, 0.0) : vec2(0.0, t.y);
    vec2 a1 = edge - along, a2 = edge + along;
    float e1 = Luma(At(a1)) - local, e2 = Luma(At(a2)) - local;
    bool r1 = abs(e1) >= scaled, r2 = abs(e2) >= scaled;
    if (!r1) a1 -= along;
    if (!r2) a2 += along;
    for (int i = 2; i < 12 && !(r1 && r2); ++i) {
        if (!r1) e1 = Luma(At(a1)) - local;
        if (!r2) e2 = Luma(At(a2)) - local;
        r1 = abs(e1) >= scaled;
        r2 = abs(e2) >= scaled;
        if (!r1) a1 -= along * Step(i);
        if (!r2) a2 += along * Step(i);
    }
    const float d1 = is_horizontal ? at.x - a1.x : at.y - a1.y;
    const float d2 = is_horizontal ? a2.x - at.x : a2.y - at.y;
    const bool nearer1 = d1 < d2;
    const float nearest = min(d1, d2), length = d1 + d2;
    const bool smaller = lc < local;
    const bool right_way = ((nearer1 ? e1 : e2) < 0.0) != smaller;
    float offset = right_way ? 0.5 - nearest / length : 0.0;

    // Subpixel aliasing (a lone pixel, a thin line): by how much the
    // neighbourhood's average differs from the pixel.
    const float average = (2.0 * (down_up + left_right) + left_corners + right_corners) / 12.0;
    const float s1 = clamp(abs(average - lc) / range, 0.0, 1.0);
    const float s2 = (-2.0 * s1 + 3.0) * s1 * s1;
    offset = max(offset, s2 * s2 * kSubpixel);

    vec2 final = at;
    if (is_horizontal) final.y += offset * across;
    else final.x += offset * across;
    return At(final);
}

void main() {
    vec3 c = p.fxaa != 0u ? Fxaa(uv) : At(uv);
    if (p.sharpen > 0.0) {
        const vec2 t = p.texel;
        const vec3 n = At(uv - vec2(0.0, t.y)), s = At(uv + vec2(0.0, t.y));
        const vec3 w = At(uv - vec2(t.x, 0.0)), e = At(uv + vec2(t.x, 0.0));
        const vec3 lo = min(c, min(min(n, s), min(w, e))), hi = max(c, max(max(n, s), max(w, e)));
        const vec3 amount = sqrt(clamp(min(lo, 1.0 - hi) / max(hi, vec3(1.0 / 255.0)), 0.0, 1.0));
        const vec3 weight = amount * (-1.0 / mix(8.0, 5.0, p.sharpen));
        c = clamp((c + (n + s + w + e) * weight) / (1.0 + 4.0 * weight), 0.0, 1.0);
    }
    color = vec4(c, 1.0);
}
