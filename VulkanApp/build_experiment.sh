#!/bin/bash
export VULKAN_SDK_PATH=/Users/taarriag/Development/vulkansdk-macos-1.1.97.0
export VK_LAYER_PATH=${VULKAN_SDK_PATH}/macOS/etc/vulkan/explicit_layer.d
export VK_ICD_FILENAMES=${VULKAN_SDK_PATH}/macOS/etc/vulkan/icd.d/MoltenVK_icd.json

# TODO: Download dependencies (move to install_dependencies.sh)
# brew install glfw3 --HEAD
# brew install glm

# TODO: Compile the shaders, copy them to data/shaders.
./compile_shaders.sh

# TODO: Copy the

# TODO: Copy the dynamic libraries

# Compile the application to main.
clang++ --std=C++17 -stdlib=libstdc++ \
  HelloTriangleApplication.cpp -o main \
    -lstdc++ \
    -I/Users/taarriag/homebrew/include \
    -I${VULKAN_SDK_PATH}/macOS/include \
    -I/Users/taarriag/Development/glm/glm \
    -I/Users/taarriag/Development/stb \
    -I/Users/taarriag/Development/tinyobjloader \
    -L${VULKAN_SDK_PATH}/macOS/lib
    -L$/Users/taarriag/homebrew/lib