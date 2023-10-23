#pragma once

#include "buffer.hpp"
#include "texture.hpp"
#include "sampler.hpp"
#include <glad/glad.h>
#include <vector>

namespace onec
{

class BindGroup
{
public:
	void bind() const;
	void unbind() const;

	void attachUniformBuffer(const GLuint location, const Buffer& buffer);
	void attachStorageBuffer(const GLuint location, Buffer& buffer);
	void attachTexture(const GLuint unit, Texture& texture);
	void attachSampler(const GLuint unit, Sampler& sampler);
	void detachUniformBuffer(const GLuint location);
	void detachStorageBuffer(const GLuint location);
	void detachTexture(const GLuint unit);
	void detachSampler(const GLuint unit);
private:
	struct BufferBinding
	{
		GLuint buffer;
		GLenum target;
		GLuint location;
	};

	struct TextureBinding
	{
		GLuint texture;
		GLuint unit;
	};

	struct SamplerBinding
	{
		GLuint sampler;
		GLuint unit;
	};

	void attachBuffer(const GLenum target, const GLuint location, const Buffer& buffer);
	void detachBuffer(const GLenum target, const GLuint location);

	std::vector<BufferBinding> m_bufferBindings;
	std::vector<TextureBinding> m_textureBindings;
	std::vector<SamplerBinding> m_samplerBindings;
};

}
