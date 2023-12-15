#include "sampler.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <utility>
#include <string>

namespace onec
{

Sampler::Sampler() :
	m_handle{ GL_NONE }
{

}

Sampler::Sampler(const SamplerState& state)
{
	create(state);
}

Sampler::Sampler(Sampler&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) }
{

}

Sampler::~Sampler()
{
	destroy();
}

Sampler& Sampler::operator=(Sampler&& other) noexcept
{
	if (this != &other)
	{
		destroy();
		m_handle = std::exchange(other.m_handle, GL_NONE);
	}

	return *this;
}

void Sampler::initialize(const SamplerState& state)
{
	destroy();
	create(state);
}

void Sampler::release()
{
	destroy();
	m_handle = GL_NONE;
}

GLuint Sampler::getHandle()
{
	return m_handle;
}

bool Sampler::isEmpty() const
{
	return m_handle == GL_NONE;
}

void Sampler::create(const SamplerState& state)
{
	GLuint handle;
	GL_CHECK_ERROR(glCreateSamplers(1, &handle));

	m_handle = handle;

	GL_CHECK_ERROR(glSamplerParameterIuiv(handle, GL_TEXTURE_MIN_FILTER, &state.minFilter));
	GL_CHECK_ERROR(glSamplerParameterIuiv(handle, GL_TEXTURE_MAG_FILTER, &state.magFilter));
	GL_CHECK_ERROR(glSamplerParameterIuiv(handle, GL_TEXTURE_WRAP_S, &state.wrapMode.x));
	GL_CHECK_ERROR(glSamplerParameterIuiv(handle, GL_TEXTURE_WRAP_T, &state.wrapMode.y));
	GL_CHECK_ERROR(glSamplerParameterIuiv(handle, GL_TEXTURE_WRAP_R, &state.wrapMode.z));
	GL_CHECK_ERROR(glSamplerParameterfv(handle, GL_TEXTURE_BORDER_COLOR, &state.borderColor.x));
	GL_CHECK_ERROR(glSamplerParameterfv(handle, GL_TEXTURE_LOD_BIAS, &state.levelOfDetailBias));
	GL_CHECK_ERROR(glSamplerParameterfv(handle, GL_TEXTURE_MIN_LOD, &state.minLevelOfDetail));
	GL_CHECK_ERROR(glSamplerParameterfv(handle, GL_TEXTURE_MAX_LOD, &state.maxLevelOfDetail));
	GL_CHECK_ERROR(glSamplerParameterfv(handle, GL_TEXTURE_MAX_ANISOTROPY, &state.maxAnisotropy));
}

void Sampler::destroy()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteSamplers(1, &m_handle));
	}
}

}

