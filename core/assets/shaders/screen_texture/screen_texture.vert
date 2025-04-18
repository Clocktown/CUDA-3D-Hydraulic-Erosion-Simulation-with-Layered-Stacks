#version 460
//#extension GL_NV_shader_buffer_load : require

#include "../render_pipeline/render_pipeline.glsl"
#include "../render_pipeline/maths.glsl"
#include "../screen_texture_render_pipeline/screen_texture_render_pipeline.glsl"

layout(location = 0) out vec2 uv;

void main() 
{
    float x     = -1.0 + float((gl_VertexID & 1) << 2);
    float y     = -1.0 + float((gl_VertexID & 2) << 1);
    uv.x        = (x + 1.0) * 0.5;
    uv.y        = (y + 1.0) * 0.5;
    gl_Position = vec4(x, y, 0, 1);
}
