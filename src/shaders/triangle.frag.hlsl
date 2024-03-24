
struct VertexOutput
{
    float4 position : SV_POSITION;
    [[vk::location(0)]]
    float4 color : COLOR;
};


float4 main(VertexOutput input) : SV_TARGET0
{
    return input.color;
}
