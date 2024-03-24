[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture2D<float4> inputTexture;

[[vk::image_format("rgba8")]]
RWTexture2D<float4> result : register(u0);

struct VertexInput {
    int2 DTid : SV_DispatchThreadID;
    int2 GTid : SV_GroupThreadID;
};

[numthreads(16,16,1)]
void main(VertexInput input) {
  int2 texelCoord = input.DTid;

  int2 size;
  inputTexture.GetDimensions(size.x, size.y);
  if ((texelCoord.x < size.x) && (texelCoord.y < size.y)) {
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    if ((input.GTid.x != 0) && (input.GTid.y != 0)) {
        color.x = float(texelCoord.x) / float(size.x);
        color.y = float(texelCoord.y) / float(size.y);
    }
    result[texelCoord] = color;
  }
}