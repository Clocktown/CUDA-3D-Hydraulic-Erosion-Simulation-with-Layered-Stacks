#include "sampler.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <utility>
#include <string>

namespace onec
{

Sampler::Sampler()
{
	GL_CHECK_ERROR(glCreateSamplers(1, &m_handle));
}

Sampler::Sampler(Sampler&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) }
{

}

Sampler::~Sampler()
{
	GL_CHECK_ERROR(glDeleteSamplers(1, &m_handle));
}

Sampler& Sampler::operator=(Sampler&& other) noexcept
{
	if (this != &other)
	{
		GL_CHECK_ERROR(glDeleteSamplers(1, &m_handle));
		m_handle = std::exchange(other.m_handle, GL_NONE);
	}

	return *this;
}

void Sampler::bind(const GLuint unit) const
{
	GL_CHECK_ERROR(glBindSampler(unit, m_handle));
}

void Sampler::unbind(const GLuint unit) const
{
	GL_CHECK_ERROR(glBindSampler(unit, GL_NONE));
}

void Sampler::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_SAMPLER, name);
}

void Sampler::setMinFilter(const GLenum minFilter)
{
	GL_CHECK_ERROR(glSamplerParameterIuiv(m_handle, GL_TEXTURE_MIN_FILTER, &minFilter));
}

void Sampler::setMagFilter(const GLenum magFilter)
{
	GL_CHECK_ERROR(glSamplerParameterIuiv(m_handle, GL_TEXTURE_MAG_FILTER, &magFilter));
}

void Sampler::setWrapModeS(const GLenum wrapModeS)
{
	GL_CHECK_ERROR(glSamplerParameterIuiv(m_handle, GL_TEXTURE_WRAP_S, &wrapModeS));
}

void Sampler::setWrapModeT(const GLenum wrapModeT)
{
	GL_CHECK_ERROR(glSamplerParameterIuiv(m_handle, GL_TEXTURE_WRAP_T, &wrapModeT));
}

void Sampler::setWrapModeR(const GLenum wrapModeR)
{
	GL_CHECK_ERROR(glSamplerParameterIuiv(m_handle, GL_TEXTURE_WRAP_R, &wrapModeR));
}

void Sampler::setBorderColor(const glm::vec4& borderColor)
{
	GL_CHECK_ERROR(glSamplerParameterfv(m_handle, GL_TEXTURE_BORDER_COLOR, &borderColor.x));
}

void Sampler::setLODBias(const float lodBias)
{
	GL_CHECK_ERROR(glSamplerParameterfv(m_handle, GL_TEXTURE_LOD_BIAS, &lodBias));
}

void Sampler::setMinLOD(const float minLOD)
{
	GL_CHECK_ERROR(glSamplerParameterfv(m_handle, GL_TEXTURE_MIN_LOD, &minLOD));
}

void Sampler::setMaxLOD(const float maxLOD)
{
	GL_CHECK_ERROR(glSamplerParameterfv(m_handle, GL_TEXTURE_MAX_LOD, &maxLOD));
}

void Sampler::setMaxAnisotropy(const float maxAnisotropy)
{
	GL_CHECK_ERROR(glSamplerParameterfv(m_handle, GL_TEXTURE_MAX_ANISOTROPY, &maxAnisotropy));
}

GLuint Sampler::getHandle()
{
	return m_handle;
}

}
