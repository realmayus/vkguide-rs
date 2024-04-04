#version 450
#extension GL_EXT_buffer_reference : require

struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 color;
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


vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, cutoff);
}

void main() {
    Vertex inVertex = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position =
    vec4(2.0 * inVertex.position.x / PushConstants.screen_size.x - 1.0,
    2.0 * inVertex.position.y / PushConstants.screen_size.y - 1.0, 0.0, 1.0);
    outColor = vec4(srgb_to_linear(inVertex.color.rgb), inVertex.color.a);
    outUV = vec2(inVertex.uv_x, inVertex.uv_y);
}