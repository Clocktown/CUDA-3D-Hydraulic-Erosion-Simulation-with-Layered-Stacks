#include "bind_group.hpp"
#include "buffer.hpp"
#include "texture.hpp"
#include "sampler.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <vector>

namespace onec
{

void BindGroup::bind() const
{
	for (const BufferBinding& bufferBinding : m_bufferBindings)
	{
		GL_CHECK_ERROR(glBindBufferBase(bufferBinding.target, bufferBinding.location, bufferBinding.buffer));
	}

	for (const TextureBinding& textureBinding : m_textureBindings)
	{
		GL_CHECK_ERROR(glBindTextureUnit(textureBinding.unit, textureBinding.texture));
	}

	for (const SamplerBinding& samplerBinding : m_samplerBindings)
	{
		GL_CHECK_ERROR(glBindSampler(samplerBinding.unit, samplerBinding.sampler));
	}
}

void BindGroup::unbind() const
{
	for (const BufferBinding& bufferBinding : m_bufferBindings)
	{
		GL_CHECK_ERROR(glBindBufferBase(bufferBinding.target, bufferBinding.location, GL_NONE));
	}

	for (const TextureBinding& textureBinding : m_textureBindings)
	{
		GL_CHECK_ERROR(glBindTextureUnit(textureBinding.unit, GL_NONE));
	}

	for (const SamplerBinding& samplerBinding : m_samplerBindings)
	{
		GL_CHECK_ERROR(glBindSampler(samplerBinding.unit, GL_NONE));
	}
}

void BindGroup::attachUniformBuffer(const GLuint location, const Buffer& buffer)
{
	attachBuffer(GL_UNIFORM_BUFFER, location, buffer);
}

void BindGroup::attachStorageBuffer(const GLuint location, Buffer& buffer)
{
	attachBuffer(GL_SHADER_STORAGE_BUFFER, location, buffer);
}

void BindGroup::attachTexture(const GLuint unit, Texture& texture)
{
	for (TextureBinding& textureBinding : m_textureBindings)
	{
		if (textureBinding.unit == unit)
		{
			textureBinding.texture = texture.getHandle();
			return;
		}
	}

	m_textureBindings.emplace_back(texture.getHandle(), unit);
}

void BindGroup::attachSampler(const GLuint unit, Sampler& sampler)
{
	for (SamplerBinding& samplerBinding : m_samplerBindings)
	{
		if (samplerBinding.unit == unit)
		{
			samplerBinding.sampler = sampler.getHandle();
			return;
		}
	}

	m_samplerBindings.emplace_back(sampler.getHandle(), unit);
}

void BindGroup::attachBuffer(const GLenum target, const GLuint location, const Buffer& buffer)
{
	for (BufferBinding& bufferBinding : m_bufferBindings)
	{
		if (bufferBinding.target == target && bufferBinding.location == location)
		{
			bufferBinding.buffer = const_cast<Buffer&>(buffer).getHandle();

			return;
		}
	}

	m_bufferBindings.emplace_back(const_cast<Buffer&>(buffer).getHandle(), target, location);
}

void BindGroup::detachUniformBuffer(const GLuint location)
{
	detachBuffer(GL_UNIFORM_BUFFER, location);
}

void BindGroup::detachStorageBuffer(const GLuint location)
{
	detachBuffer(GL_SHADER_STORAGE_BUFFER, location);
}

void BindGroup::detachTexture(const GLuint unit)
{
	for (size_t i{ 0 }; i < m_textureBindings.size(); ++i)
	{
		if (m_textureBindings[i].unit == unit)
		{
			m_textureBindings[i] = m_textureBindings[m_textureBindings.size() - 1];
			m_textureBindings.pop_back();

			return;
		}
	}
}

void BindGroup::detachSampler(const GLuint unit)
{
	for (size_t i{ 0 }; i < m_samplerBindings.size(); ++i)
	{
		if (m_samplerBindings[i].unit == unit)
		{
			m_samplerBindings[i] = m_samplerBindings[m_samplerBindings.size() - 1];
			m_samplerBindings.pop_back();

			return;
		}
	}
}

void BindGroup::detachBuffer(const GLenum target, const GLuint location)
{
	for (size_t i{ 0 }; i < m_bufferBindings.size(); ++i)
	{
		const BufferBinding& bufferBinding{ m_bufferBindings[i] };

		if (bufferBinding.target == target && bufferBinding.location == location)
		{
			m_bufferBindings[i] = m_bufferBindings[m_bufferBindings.size() - 1];
			m_bufferBindings.pop_back();

			return;
		}
	}
}

}
