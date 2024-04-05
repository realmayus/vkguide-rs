#version 450
#extension GL_EXT_buffer_reference : require

struct Vertex {
    vec2 position;
    vec2 uv;
    uint color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{
    Vertex vertices[];
};

//push constants block
layout( push_constant ) uniform constants
{
    vec2 screen_size;
    VertexBuffer vertexBuffer;
    int fontTextureId;
} PushConstants;


layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outUV;
layout(location = 2) out flat int fontTextureId;

vec4 decodeRGBA(uint encodedColor) {
    float a = float((encodedColor & 0xFFu) >> 0) / 255.0;
    float b = float((encodedColor & 0xFF00u) >> 8) / 255.0;
    float g = float((encodedColor & 0xFF0000u) >> 16) / 255.0;
    float r = float((encodedColor & 0xFF000000u) >> 24) / 255.0;

    return vec4(r, g, b, a);
}

void main() {
    Vertex inVertex = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position =
    vec4(2.0 * inVertex.position.x / PushConstants.screen_size.x - 1.0,
    2.0 * inVertex.position.y / PushConstants.screen_size.y - 1.0, 0.0, 1.0);
    vec4 color = decodeRGBA(inVertex.color);
    outColor = color;
    outUV = vec2(inVertex.uv.x, inVertex.uv.y);
    fontTextureId = PushConstants.fontTextureId;
}