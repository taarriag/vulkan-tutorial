#version 450
#extension GL_ARB_separate_shader_objects : enable

// Input variables received from the vertex shader.
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(binding = 1) uniform sampler2D texSampler;
// Output variable for each framebuffer, where location is the index of
// the frame buffer.
layout(location = 0) out vec4 outColor;

void main() {
  //outColor = vec4(fragTexCoord, 0.0, 1.0);
  //outColor = texture(texSampler, fragTexCoord);
  // Go beyond the UV borders (0.0 to 2.0 range).
  // Since the texture is set to repeat, it will be drawn again.
  //outColor = texture(texSampler, fragTexCoord * 2.0);
  //outColor = vec4(fragColor * texture(texSampler, fragTexCoord).rgb, 1.0);
  outColor = texture(texSampler, fragTexCoord);
}