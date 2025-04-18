#version 460 core

#include "../render_pipeline/render_pipeline.glsl"
#include "../render_pipeline/maths.glsl"
#include "../screen_texture_render_pipeline/screen_texture_render_pipeline.glsl"

layout(location = 0) in vec2 uv;

out vec4 fragCol;

void main() {
	fragCol = texture(screenTexture, uv);
}
