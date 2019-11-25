#version 450
#extension GL_ARB_separate_shader_objects : enable

// TODO: Consider alignment when mapping C++ to SpirV structs 
// https://vulkan-tutorial.com/Uniform_buffers/Descriptor_pool_and_sets#page_Alignment_requirements
layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

// Pass the per-vertex colors to the fragment shader
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

// Hardcoded vertices
/*vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
  vec3(1.0, 0.0, 0.0),
  vec3(0.0, 1.0, 0.0),
  vec3(0.0, 0.0, 1.0)
);*/

void main() {
    // gl_Position is a built-in variable. Added Dummy z and w values.
    //gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    //fragColor = colors[gl_VertexIndex];
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
