# compile compute shaders
for file in src/shaders/*.comp.hlsl; do
    if [[ ! -e "$file" ]]; then continue; fi
    # get the filename without the path
    filename=$(basename -- "$file")
    # get the filename without the extension
    filename_no_ext="${filename%.*}"
    # compile the shader
    dxc -spirv -T cs_6_0 -E main $file -Fo src/shaders/spirv/$filename_no_ext.spv
done


# compile vertex shaders
for file in src/shaders/*.vert.hlsl; do
    if [[ ! -e "$file" ]]; then continue; fi
    filename=$(basename -- "$file")
    filename_no_ext="${filename%.*}"
    dxc -spirv -T vs_6_0 -E main $file -Fo src/shaders/spirv/$filename_no_ext.spv
done

# compile fragment shaders
for file in src/shaders/*.frag.hlsl; do
    if [[ ! -e "$file" ]]; then continue; fi
    filename=$(basename -- "$file")
    filename_no_ext="${filename%.*}"
    dxc -spirv -T ps_6_0 -E main $file -Fo src/shaders/spirv/$filename_no_ext.spv
done
