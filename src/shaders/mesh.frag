#version 450

//shader input
layout (location = 0) in vec3 inColor;
layout (set = 0, binding = 2) uniform sampler2D tex[];

//output write
layout (location = 0) out vec4 outFragColor;

void main()
{
    //return red
    outFragColor = texture(tex[2], vec2(0.0, 0.0));
}