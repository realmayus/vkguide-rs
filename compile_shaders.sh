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
