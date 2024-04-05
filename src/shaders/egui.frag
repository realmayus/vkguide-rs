#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat int fontTextureId;

layout(location = 0) out vec4 outColor;

layout (set = 0, binding = 2) uniform sampler2D tex[];

vec3 srgb_from_linear(vec3 linear) {
    bvec3 cutoff = lessThan(linear, vec3(0.0031308));
    vec3 lower = linear * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(linear, vec3(1./2.4)) - vec3(0.055);
    return mix(higher, lower, vec3(cutoff));
}

// 0-1 sRGBA  from  0-1 linear
vec4 srgba_from_linear(vec4 linear) {
    return vec4(srgb_from_linear(linear.rgb), linear.a);
}

void main() {
    vec4 texture_color = srgba_from_linear(texture(tex[fontTextureId], inUV));
    outColor = inColor * texture_color;
}