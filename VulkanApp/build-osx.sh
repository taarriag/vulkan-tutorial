#!/bin/bash
# Environment variables required for Vulkan on OSX
export VULKAN_SDK_PATH=/Users/taarriag/Development/vulkansdk-macos-1.1.97.0
export HOMEBREW_PATH=/Users/taarriag/homebrew
export GLM_PATH=/Users/taarriag/Development/glm

export VK_LAYER_PATH=/Users/taarriag/Development/vulkansdk-macos-1.1.97.0/macOS/etc/vulkan/explicit_layer.d
export VK_ICD_FILENAMES=/Users/taarriag/Development/vulkansdk-macos-1.1.97.0/macOS/etc/vulkan/icd.d/MoltenVK_icd.json

INCLUDE_PATHS=${HOMEBREW_PATH}/include ${VULKAN_SDK_PATH}/macOS/include ${GLM_PATH}/glm third_party/stb third_party/tinyobjloader
LIBRARY_PATHS=${VULKAN_SDK_PATH}/macOS/lib ${HOMEBREW_PATH}/lib





