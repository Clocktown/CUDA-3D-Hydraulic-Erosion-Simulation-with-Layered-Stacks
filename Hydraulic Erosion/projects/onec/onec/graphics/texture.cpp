#include "texture.hpp"
#include "../config/gl.hpp"
#include "../utility/io.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <utility>
#include <filesystem>
#include <memory>
#include <string>

namespace onec
{

Texture::Texture() :
	m_handle{ GL_NONE },
	m_target{ GL_NONE },
	m_size{ 0 },
	m_format{ GL_NONE },
	m_mipCount{ 0 },
	m_sampleCount{ 0 }
{

}

Texture::Texture(const GLenum target, const glm::ivec3& size, const GLenum format, const int mipCount, const int sampleCount) :
	m_target{ target },
	m_size{ size },
	m_format{ format }
{
	if (m_size != glm::ivec3{ 0 })
	{
		m_mipCount = mipCount;
		m_sampleCount = sampleCount;

		create();
	}
	else
	{
		m_handle = GL_NONE;
		m_mipCount = 0;
		m_sampleCount = 0;
	}
}

Texture::Texture(const std::filesystem::path& file, const int mipCount) :
	m_target{ GL_TEXTURE_2D },
	m_mipCount{ mipCount },
	m_sampleCount{ 0 }
{
	m_size.z = 0;

	GLenum pixelFormat;
	GLenum pixelType;
	const auto data{ readImage(file, reinterpret_cast<glm::ivec2&>(m_size), m_format, pixelFormat, pixelType) };

	GL_CHECK_ERROR(glCreateTextures(GL_TEXTURE_2D, 1, &m_handle));
	GL_CHECK_ERROR(glGetInternalformativ(m_target, m_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));
	GL_CHECK_ERROR(glTextureStorage2D(m_handle, m_mipCount, m_format, m_size.x, m_size.y));
	GL_CHECK_ERROR(glTextureSubImage2D(m_handle, 0, 0, 0, m_size.x, m_size.y, pixelFormat, pixelType, data.get()));
	GL_CHECK_ERROR(glGenerateTextureMipmap(m_handle));
}

Texture::Texture(const Texture& other) :
	m_target{ other.m_target },
	m_size{ other.m_size },
	m_format{ other.m_format },
	m_mipCount{ other.m_mipCount }
{
	if (!other.isEmpty())
	{
		create();

		glm::ivec3 size{ m_target == GL_TEXTURE_1D_ARRAY ? glm::ivec3{ m_size.x, 1, m_size.y } : 
			                                               glm::ivec3{ m_size.x, glm::max(m_size.y, 1), glm::max(m_size.z, 1) } };

		for (int i{ 0 }; i < m_mipCount; ++i, size = glm::max(size / 2, 1))
		{
			GL_CHECK_ERROR(glCopyImageSubData(other.m_handle, other.m_target, i, 0, 0, 0, m_handle, m_target, i, 0, 0, 0, size.x, size.y, size.z));
		}
	}
	else
	{
		m_handle = GL_NONE;
	}
}

Texture::Texture(Texture&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_target{ std::exchange(other.m_target, GL_NONE) },
	m_size{ std::exchange(other.m_size, glm::ivec3{ 0 }) },
	m_format{ std::exchange(other.m_format, GL_NONE) },
	m_mipCount{ std::exchange(other.m_mipCount, 0) },
	m_sampleCount{ std::exchange(other.m_sampleCount, 0) }
{

}

Texture::~Texture()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteTextures(1, &m_handle));
	}
}

Texture& Texture::operator=(const Texture& other)
{
	if (this != &other)
	{
		initialize(other.m_target, other.m_size, other.m_format, other.m_mipCount);

		if (!other.isEmpty())
		{
			glm::ivec3 size{ m_target == GL_TEXTURE_1D_ARRAY ? glm::ivec3{ m_size.x, 1, m_size.y } :
												               glm::ivec3{ m_size.x, glm::max(m_size.y, 1), glm::max(m_size.z, 1) } };

			for (int i{ 0 }; i < m_mipCount; ++i, size = glm::max(size / 2, 1))
			{
				GL_CHECK_ERROR(glCopyImageSubData(other.m_handle, other.m_target, i, 0, 0, 0, m_handle, m_target, i, 0, 0, 0, size.x, size.y, size.z));
			}
		}
	}

	return *this;
}

Texture& Texture::operator=(Texture&& other) noexcept
{
	if (this != &other)
	{
		if (!isEmpty())
		{
			GL_CHECK_ERROR(glDeleteTextures(1, &m_handle));
		}

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_target = std::exchange(other.m_target, GL_NONE);
		m_size = std::exchange(other.m_size, glm::ivec3{ 0 });
		m_format = std::exchange(other.m_format, GL_NONE);
		m_mipCount = std::exchange(other.m_mipCount, 0);
		m_sampleCount = std::exchange(other.m_sampleCount, 0);
	}

	return *this;
}

void Texture::bind(const GLuint unit) const
{
	GL_CHECK_ERROR(glBindTextureUnit(unit, m_handle));
}

void Texture::unbind(const GLuint unit) const
{
	GL_CHECK_ERROR(glBindTextureUnit(unit, GL_NONE));
}

void Texture::initialize(const GLenum target, const glm::ivec3& size, const GLenum format, const int mipCount, const int sampleCount)
{
	if (m_target == target && m_size == size && m_format == format && m_mipCount == mipCount)
	{
		return;
	}

	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteTextures(1, &m_handle));
	}

	m_target = target;
	m_size = size;
	m_format = format;

	if (m_size != glm::ivec3{ 0 })
	{
		m_mipCount = mipCount;
		m_sampleCount = sampleCount;

		create();
	}
	else
	{
		m_handle = GL_NONE;
		m_mipCount = 0;
		m_sampleCount = 0;
	}
}

void Texture::release()
{
	if (!isEmpty())
	{
		GL_CHECK_ERROR(glDeleteTextures(1, &m_handle));

		m_handle = GL_NONE;
		m_target = GL_NONE;
		m_size = glm::ivec3{ 0 };
		m_format = GL_NONE;
		m_mipCount = 0;
	}
}

void Texture::generateMipmap()
{
	GL_CHECK_ERROR(glGenerateTextureMipmap(m_handle));
}

void Texture::upload(const Span<const std::byte>&& data, const GLenum format, const GLenum type, const int mipLevel)
{
	glm::ivec3 size{ glm::ivec3{ glm::pow(0.5f, static_cast<float>(mipLevel)) * glm::vec3{ m_size } } };
	size.y = glm::max(size.y, 1);
	size.z = glm::max(size.z, 1);

	upload(std::forward<const Span<const std::byte>&&>(data), format, type, glm::ivec3{ 0 }, size, mipLevel);
}

void Texture::upload(const Span<const std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& size, const int mipLevel)
{
	upload(std::forward<const Span<const std::byte>&&>(data), format, type, glm::ivec3{ 0 }, size, mipLevel);
}

void Texture::upload(const Span<const std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& offset, const glm::ivec3& size, const int mipLevel)
{
	switch (m_target)
	{
	case GL_TEXTURE_1D:
		ONEC_ASSERT(offset.y == 0, "Offset y must be equal to 0");
		ONEC_ASSERT(offset.z == 0, "Offset z must be equal to 0");
		ONEC_ASSERT(size.y == 1, "Size y must be equal to 1");
		ONEC_ASSERT(size.z == 1, "Size z must be equal to 1");

		GL_CHECK_ERROR(glTextureSubImage1D(m_handle, mipLevel, offset.x, size.x, format, type, data.getData()));
		break;
	case GL_TEXTURE_1D_ARRAY:
	case GL_TEXTURE_2D:
		ONEC_ASSERT(offset.z == 0, "Offset z must be equal to 0");
		ONEC_ASSERT(size.z == 1, "Size z must be equal to 1");

		GL_CHECK_ERROR(glTextureSubImage2D(m_handle, mipLevel, offset.x, offset.y, size.x, size.y, format, type, data.getData()));
		break;
	case GL_TEXTURE_2D_ARRAY:
	case GL_TEXTURE_CUBE_MAP:
	case GL_TEXTURE_CUBE_MAP_ARRAY:
	case GL_TEXTURE_3D:
		GL_CHECK_ERROR(glTextureSubImage3D(m_handle, mipLevel, offset.x, offset.y, offset.z, size.x, size.y, size.z, format, type, data.getData()));
		break;
	default:
		ONEC_ERROR("Target must be valid");

		break;
	}
}

void Texture::download(const Span<std::byte>&& data, const GLenum format, const GLenum type, const int mipLevel) const
{
	glm::ivec3 size{ glm::ivec3{ glm::pow(0.5f, static_cast<float>(mipLevel)) * glm::vec3{ m_size } } };
	size.y = glm::max(size.y, 1);
	size.z = glm::max(size.z, 1);

	GL_CHECK_ERROR(glGetTextureSubImage(m_handle, mipLevel, 0, 0, 0, size.x, size.y, size.z, format, type, data.getByteCount(), data.getData()));
}

void Texture::download(const Span<std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& size, const int mipLevel) const
{
	GL_CHECK_ERROR(glGetTextureSubImage(m_handle, mipLevel, 0, 0, 0, size.x, size.y, size.z, format, type, data.getByteCount(), data.getData()));
}

void Texture::download(const Span<std::byte>&& data, const GLenum format, const GLenum type, const glm::ivec3& offset, const glm::ivec3& size, const int mipLevel) const
{
	GL_CHECK_ERROR(glGetTextureSubImage(m_handle, mipLevel, offset.x, offset.y, offset.z, size.x, size.y, size.z, format, type, data.getByteCount(), data.getData()));
}

void Texture::create()
{
	GL_CHECK_ERROR(glCreateTextures(m_target, 1, &m_handle));
	GL_CHECK_ERROR(glGetInternalformativ(m_target, m_format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&m_format)));

	switch (m_target)
	{
	case GL_TEXTURE_1D:
		ONEC_ASSERT(m_size.y == 0, "Size y must be 0");
		ONEC_ASSERT(m_size.z == 0, "Size z must be 0");

		GL_CHECK_ERROR(glTextureStorage1D(m_handle, m_mipCount, m_format, m_size.x));
		break;
	case GL_TEXTURE_1D_ARRAY:
	case GL_TEXTURE_2D:
		ONEC_ASSERT(m_size.z == 0, "Size z must be 0");

		GL_CHECK_ERROR(glTextureStorage2D(m_handle, m_mipCount, m_format, m_size.x, m_size.y));
		break;
	case GL_TEXTURE_2D_MULTISAMPLE:
		ONEC_ASSERT(m_size.z == 0, "Size z must be 0");

		GL_CHECK_ERROR(glTextureStorage2DMultisample(m_handle, m_sampleCount, m_format, m_size.x, m_size.y, true));
		break;
	case GL_TEXTURE_CUBE_MAP:
		ONEC_ASSERT(m_size.z == 6, "Size z must be 6");

		GL_CHECK_ERROR(glTextureStorage2D(m_handle, m_mipCount, m_format, m_size.x, m_size.y));
		break;
	case GL_TEXTURE_2D_ARRAY:
	case GL_TEXTURE_CUBE_MAP_ARRAY:
	case GL_TEXTURE_3D:
		GL_CHECK_ERROR(glTextureStorage3D(m_handle, m_mipCount, m_format, m_size.x, m_size.y, m_size.z));
		break;
	case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
		GL_CHECK_ERROR(glTextureStorage3DMultisample(m_handle, m_sampleCount, m_format, m_size.x, m_size.y, m_size.z, true));
		break;
	default:
		ONEC_ERROR("Target must be valid");

		break;
	}
}

void Texture::setName(const std::string_view& name)
{
	GL_LABEL_OBJECT(m_handle, GL_TEXTURE, name);
}

void Texture::setMinFilter(const GLenum minFilter)
{
	GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_MIN_FILTER, &minFilter));
}

void Texture::setMagFilter(const GLenum magFilter)
{
	GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_MAG_FILTER, &magFilter));
}

void Texture::setWrapModeS(const GLenum wrapModeS)
{
	GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_S, &wrapModeS));
}

void Texture::setWrapModeT(const GLenum wrapModeT)
{
	GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_T, &wrapModeT));
}

void Texture::setWrapModeR(const GLenum wrapModeR)
{
	GL_CHECK_ERROR(glTextureParameterIuiv(m_handle, GL_TEXTURE_WRAP_R, &wrapModeR));
}

void Texture::setBorderColor(const glm::vec4& borderColor)
{
	GL_CHECK_ERROR(glTextureParameterfv(m_handle, GL_TEXTURE_BORDER_COLOR, &borderColor.x));
}

void Texture::setLODBias(const float lodBias)
{
	GL_CHECK_ERROR(glTextureParameterfv(m_handle, GL_TEXTURE_LOD_BIAS, &lodBias));
}

void Texture::setMinLOD(const float minLOD)
{
	GL_CHECK_ERROR(glTextureParameterfv(m_handle, GL_TEXTURE_MIN_LOD, &minLOD));
}

void Texture::setMaxLOD(const float maxLOD)
{
	GL_CHECK_ERROR(glTextureParameterfv(m_handle, GL_TEXTURE_MAX_LOD, &maxLOD));
}

void Texture::setMaxAnisotropy(const float maxAnisotropy)
{
	GL_CHECK_ERROR(glTextureParameterfv(m_handle, GL_TEXTURE_MAX_ANISOTROPY, &maxAnisotropy));
}

GLuint Texture::getHandle()
{
	return m_handle;
}

GLenum Texture::getTarget() const
{
	return m_target;
}

const glm::ivec3& Texture::getSize() const
{
	return m_size;
}

GLenum Texture::getFormat() const
{
	return m_format;
}

int Texture::getMipCount() const
{
	return m_mipCount;
}

int Texture::getSampleCount() const
{
	return m_sampleCount;
}

bool Texture::isEmpty() const
{
	return m_handle == GL_NONE;
}

int getMaxMipCount(const glm::ivec3& size)
{
	return 1 + static_cast<int>(glm::log2(static_cast<float>(glm::max(size.x, glm::max(size.y, size.z)))));
}

glm::vec3 getMipSize(const glm::ivec3& base, const int mipLevel)
{
	return base / (1 << mipLevel);
}

}
