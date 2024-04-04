#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inUV;
layout(location = 2) in flat int fontTextureId;

layout(location = 0) out vec4 outColor;

layout (set = 0, binding = 2) uniform sampler2D tex[];


void main() {

    outColor = inColor * texture(tex[3], inUV);
}