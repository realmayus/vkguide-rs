# compile compute shaders
for file in src/shaders/*.comp.hlsl; do
    if [[ ! -e "$file" ]]; then continue; fi
    # get the filename without the path
    filename=$(basename -- "$file")
    # get the filename without the extension
    filename_no_ext="${filename%.*}"
    # compile the shader
    dxc -spirv -T cs_6_6 -E main $file -Fo src/shaders/spirv/$filename_no_ext.spv
done


# compile vertex shaders
for file in src/shaders/*.vert; do
    if [[ ! -e "$file" ]]; then continue; fi
    filename=$(basename -- "$file")
    glslc $file -o src/shaders/spirv/$filename.spv
done

# compile fragment shaders
for file in src/shaders/*.frag; do
    if [[ ! -e "$file" ]]; then continue; fi
    filename=$(basename -- "$file")
    glslc $file -o src/shaders/spirv/$filename.spv
done
