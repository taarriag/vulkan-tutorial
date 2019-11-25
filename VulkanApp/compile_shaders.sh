#!/bin/bash

# Compile shaders to SPIR-V format.
# TODO: Create a script at root that exports setup variables and compiles everything.
export GLSL_VALIDATOR_PATH='/Users/taarriag/Development/vulkansdk-macos-1.1.97.0/macOS/bin/glslangValidator'
${GLSL_VALIDATOR_PATH} -V shaders/triangle.vert -o data/shaders/triangle_vert.spv
${GLSL_VALIDATOR_PATH} -V shaders/triangle.frag -o data/shaders/triangle_frag.spv
