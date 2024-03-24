
struct VertexOutput
{
    float4 position : SV_POSITION;
    [[vk::location(0)]]
    float4 color : COLOR;
};

VertexOutput main(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    const float3 positions[3] = {
		float3(1.f,1.f, 0.0f),
		float3(-1.f,1.f, 0.0f),
		float3(0.f,-1.f, 0.0f)
	};

	//const array of colors for the triangle
	const float3 colors[3] = {
		float3(1.0f, 0.0f, 0.0f), //red
		float3(0.0f, 1.0f, 0.0f), //green
		float3(0.0f, 0.0f, 1.0f)  //blue
	};
    output.position = float4(positions[vertexId], 1.0);
    output.color = float4(colors[vertexId], 1.0);

    return output;
}
