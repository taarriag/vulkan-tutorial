VULKAN_SDK_PATH = /Users/taarriag/Development/vulkansdk-macos-1.1.97.0
HOMEBREW_PATH = /Users/taarriag/homebrew
GLM_PATH = /Users/taarriag/Development/glm

CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include -I$(HOMEBREW_PATH)/include -I$(GLM_PATH)
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib -L$(HOMEBREW_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan


export VK_LAYER_PATH=/Users/taarriag/Development/vulkansdk-macos-1.1.97.0/macOS/etc/vulkan/explicit_layer.d
export VK_ICD_FILENAMES=/Users/taarriag/Development/vulkansdk-macos-1.1.97.0/macOS/etc/vulkan/icd.d/MoltenVK_icd.json
