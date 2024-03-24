# MARVEL
mayus' amazing rust vulkan engine library 

## Compiling Shaders
To compile the HLSL shaders, you need to have DXC (DirectX Compiler) installed. It's included in the Vulkan SDK. You can use the provided `compile_shaders.sh` script to compile the shaders. It will compile all the shaders in the `shaders` directory and output the SPIR-V files in the `shaders/spirv` directory.