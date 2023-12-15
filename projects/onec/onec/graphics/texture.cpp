#include "texture.hpp"
#include "sampler.hpp"
#include "../config/gl.hpp"
#include "../config/cu.hpp"
#include "../utility/io.hpp"
#include "../utility/span.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <utility>
#include <filesystem>
#include <string>
#include <type_traits>

namespace onec
{

Texture::Texture() :
	m_handle{ GL_NONE },
	m_bindlessHandle{ GL_NONE },
	m_bindlessImageHandle{ GL_NONE },
	m_graphicsResource{},
	m_target{ GL_NONE },
	m_size{ 0 },
	m_format{ GL_NONE },
	m_mipCount{ 0 }
{

}

Texture::Texture(const GLenum target, const glm::ivec3 size, const GLenum format, const int mipCount, const SamplerState& samplerState, const bool createBindlessHandle, const bool createBindlessImageHandle, const bool createGraphicsResource)
{
	create(target, size, format, mipCount, samplerState, createBindlessHandle, createBindlessImageHandle, createGraphicsResource);
}

Texture::Texture(const std::filesystem::path& file, int mipCount, const SamplerState& samplerState, bool createBindlessHandle, bool createBindlessImageHandle, bool createGraphicsResource)
{
	create(file, mipCount, samplerState, createBindlessHandle, createBindlessImageHandle, createGraphicsResource);
}

Texture::Texture(Texture&& other) noexcept :
	m_handle{ std::exchange(other.m_handle, GL_NONE) },
	m_bindlessHandle{ std::exchange(other.m_bindlessHandle, GL_NONE) },
	m_bindlessImageHandle{ std::exchange(other.m_bindlessImageHandle, GL_NONE) },
	m_graphicsResource{ std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{}) },
	m_target{ std::exchange(other.m_target, GL_NONE) },
	m_size{ std::exchange(other.m_size, glm::ivec3{ 0 }) },
	m_format{ std::exchange(other.m_format, GL_NONE) },
	m_mipCount{ std::exchange(other.m_mipCount, 0) }
{

}

Texture::~Texture()
{
	destroy();
}

Texture& Texture::operator=(Texture&& other) noexcept
{
	if (this != &other)
	{
		destroy();

		m_handle = std::exchange(other.m_handle, GL_NONE);
		m_bindlessHandle = std::exchange(other.m_bindlessHandle, GL_NONE);
		m_bindlessImageHandle = std::exchange(other.m_bindlessImageHandle, GL_NONE);
		m_graphicsResource = std::exchange(other.m_graphicsResource, cudaGraphicsResource_t{});
		m_target = std::exchange(other.m_target, GL_NONE);
		m_size = std::exchange(other.m_size, glm::ivec3{ 0 });
		m_format = std::exchange(other.m_format, GL_NONE);
		m_mipCount = std::exchange(other.m_mipCount, 0);
	}

	return *this;
}

void Texture::initialize(const GLenum target, const glm::ivec3 size, const GLenum format, const int mipCount, const SamplerState& samplerState, const bool createBindlessHandle, const bool createBindlessImageHandle, const bool createGraphicsResource)
{
	destroy();
	create(target, size, format, mipCount, samplerState, createBindlessHandle, createBindlessImageHandle, createGraphicsResource);
}

void Texture::initialize(const std::filesystem::path& file, int mipCount, const SamplerState& samplerState, bool createBindlessHandle, bool createBindlessImageHandle, bool createGraphicsResource)
{
	destroy();
	create(file, mipCount, samplerState, createBindlessHandle, createBindlessImageHandle, createGraphicsResource);
}

void Texture::release()
{
	destroy();

	m_handle = GL_NONE;
	m_bindlessHandle = GL_NONE;
	m_bindlessImageHandle = GL_NONE;
	m_graphicsResource = cudaGraphicsResource_t{};
	m_target = GL_NONE;
	m_size = glm::ivec3{ 0 };
	m_format = GL_NONE;
	m_mipCount = 0;
}

void Texture::generateMipmap()
{
	GL_CHECK_ERROR(glGenerateTextureMipmap(m_handle));
}

void Texture::upload(const Span<const std::byte>&& source, const GLenum format, const GLenum type, const glm::ivec3 size, const int mipLevel)
{
	upload(std::forward<const Span<const std::byte>&&>(source), format, type, glm::ivec3{ 0 }, size, mipLevel);
}

void Texture::upload(const Span<const std::byte>&& source, const GLenum format, const GLenum type, const glm::ivec3 offset, const glm::ivec3 size, const int mipLevel)
{
	switch (m_target)
	{
	case GL_TEXTURE_1D:
		ONEC_ASSERT(offset.y == 0, "Offset y must be equal to 0");
		ONEC_ASSERT(offset.z == 0, "Offset z must be equal to 0");
		ONEC_ASSERT(size.y == 1, "Size y must be equal to 1");
		ONEC_ASSERT(size.z == 1, "Size z must be equal to 1");

		GL_CHECK_ERROR(glTextureSubImage1D(m_handle, mipLevel, offset.x, size.x, format, type, source.getData()));
		break;
	case GL_TEXTURE_1D_ARRAY:
	case GL_TEXTURE_2D:
		ONEC_ASSERT(offset.z == 0, "Offset z must be equal to 0");
		ONEC_ASSERT(size.z == 1, "Size z must be equal to 1");

		GL_CHECK_ERROR(glTextureSubImage2D(m_handle, mipLevel, offset.x, offset.y, size.x, size.y, format, type, source.getData()));
		break;
	case GL_TEXTURE_2D_ARRAY:
	case GL_TEXTURE_CUBE_MAP:
	case GL_TEXTURE_CUBE_MAP_ARRAY:
	case GL_TEXTURE_3D:
		GL_CHECK_ERROR(glTextureSubImage3D(m_handle, mipLevel, offset.x, offset.y, offset.z, size.x, size.y, size.z, format, type, source.getData()));
		break;
	default:
		ONEC_ERROR("Target must be valid");

		break;
	}
}

void Texture::download(const Span<std::byte>&& destination, const GLenum format, const GLenum type, const glm::ivec3 size, const int mipLevel) const
{
	GL_CHECK_ERROR(glGetTextureSubImage(m_handle, mipLevel, 0, 0, 0, size.x, size.y, size.z, format, type, destination.getCount(), destination.getData()));
}

void Texture::download(const Span<std::byte>&& destination, const GLenum format, const GLenum type, const glm::ivec3 offset, const glm::ivec3 size, const int mipLevel) const
{
	GL_CHECK_ERROR(glGetTextureSubImage(m_handle, mipLevel, offset.x, offset.y, offset.z, size.x, size.y, size.z, format, type, destination.getCount(), destination.getData()));
}

GLuint Texture::getHandle()
{
	return m_handle;
}

GLuint64 Texture::getBindlessHandle() const
{
	return m_bindlessHandle;
}

GLuint64 Texture::getBindlessImageHandle()
{
	return m_bindlessImageHandle;
}

cudaGraphicsResource_t Texture::getGraphicsResource()
{
	return m_graphicsResource;
}

GLenum Texture::getTarget() const
{
	return m_target;
}

glm::ivec3 Texture::getSize() const
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

bool Texture::isEmpty() const
{
	return m_size.x == 0;
}

void Texture::create(const GLenum target, const glm::ivec3 size, GLenum format, const int mipCount, const SamplerState& samplerState, const bool createBindlessHandle, const bool createBindlessImageHandle, const bool createGraphicsResource)
{
	if (size != glm::ivec3{ 0 })
	{
		GLuint handle;
		GL_CHECK_ERROR(glCreateTextures(target, 1, &handle));
		GL_CHECK_ERROR(glGetInternalformativ(target, format, GL_INTERNALFORMAT_PREFERRED, 1, reinterpret_cast<GLint*>(&format)));

		m_handle = handle;
		m_mipCount = mipCount;

		bool isLayered;

		switch (target)
		{
		case GL_TEXTURE_1D:
			ONEC_ASSERT(size.y == 0, "Size y must be 0");
			ONEC_ASSERT(size.z == 0, "Size z must be 0");

			GL_CHECK_ERROR(glTextureStorage1D(handle, mipCount, format, size.x));
			isLayered = false;
			break;
		case GL_TEXTURE_1D_ARRAY:
			ONEC_ASSERT(size.z == 0, "Size z must be 0");

			GL_CHECK_ERROR(glTextureStorage2D(handle, mipCount, format, size.x, size.y));
			isLayered = true;
			break;
		case GL_TEXTURE_2D:
			ONEC_ASSERT(size.z == 0, "Size z must be 0");

			GL_CHECK_ERROR(glTextureStorage2D(handle, mipCount, format, size.x, size.y));
			isLayered = false;
			break;
		case GL_TEXTURE_CUBE_MAP:
			ONEC_ASSERT(size.z == 6, "Size z must be 6");

			GL_CHECK_ERROR(glTextureStorage2D(handle, mipCount, format, size.x, size.y));
			isLayered = false;
			break;
		case GL_TEXTURE_2D_ARRAY:
		case GL_TEXTURE_CUBE_MAP_ARRAY:
			GL_CHECK_ERROR(glTextureStorage3D(handle, mipCount, format, size.x, size.y, size.z));
			isLayered = true;
			break;
		case GL_TEXTURE_3D:
			GL_CHECK_ERROR(glTextureStorage3D(handle, mipCount, format, size.x, size.y, size.z));
			isLayered = false;
			break;
		default:
			ONEC_ERROR("Target must be valid");

			isLayered = false;
			break;
		}

		GL_CHECK_ERROR(glTextureParameterIuiv(handle, GL_TEXTURE_MIN_FILTER, &samplerState.minFilter));
		GL_CHECK_ERROR(glTextureParameterIuiv(handle, GL_TEXTURE_MAG_FILTER, &samplerState.magFilter));
		GL_CHECK_ERROR(glTextureParameterIuiv(handle, GL_TEXTURE_WRAP_S, &samplerState.wrapMode.x));
		GL_CHECK_ERROR(glTextureParameterIuiv(handle, GL_TEXTURE_WRAP_T, &samplerState.wrapMode.y));
		GL_CHECK_ERROR(glTextureParameterIuiv(handle, GL_TEXTURE_WRAP_R, &samplerState.wrapMode.z));
		GL_CHECK_ERROR(glTextureParameterfv(handle, GL_TEXTURE_BORDER_COLOR, &samplerState.borderColor.x));
		GL_CHECK_ERROR(glTextureParameterfv(handle, GL_TEXTURE_LOD_BIAS, &samplerState.levelOfDetailBias));
		GL_CHECK_ERROR(glTextureParameterfv(handle, GL_TEXTURE_MIN_LOD, &samplerState.minLevelOfDetail));
		GL_CHECK_ERROR(glTextureParameterfv(handle, GL_TEXTURE_MAX_LOD, &samplerState.maxLevelOfDetail));
		GL_CHECK_ERROR(glTextureParameterfv(handle, GL_TEXTURE_MAX_ANISOTROPY, &samplerState.maxAnisotropy));

		if (createGraphicsResource)
		{
			cudaGraphicsResource_t graphicsResource;
			CU_CHECK_ERROR(cudaGraphicsGLRegisterImage(&graphicsResource, handle, target, cudaGraphicsRegisterFlagsNone));

			m_graphicsResource = graphicsResource;
		}
		else
		{
			m_graphicsResource = cudaGraphicsResource_t{};
		}

		if (createBindlessHandle)
		{
			const GLuint64 bindlessHandle{ glGetTextureHandleARB(handle) };
			GL_CHECK_ERROR(glMakeTextureHandleResidentARB(bindlessHandle));

			m_bindlessHandle = bindlessHandle;
		}
		else
		{
			m_bindlessHandle = GL_NONE;
		}

		if (createBindlessImageHandle)
		{
			const GLuint64 bindlessImageHandle{ glGetImageHandleARB(handle, 0, isLayered, 0, format) };
			GL_CHECK_ERROR(glMakeImageHandleResidentARB(bindlessImageHandle, GL_READ_WRITE));

			m_bindlessImageHandle = bindlessImageHandle;
		}
		else
		{
			m_bindlessImageHandle = GL_NONE;
		}
	}
	else
	{
		m_handle = GL_NONE;
		m_bindlessHandle = GL_NONE;
		m_bindlessImageHandle = GL_NONE;
		m_graphicsResource = cudaGraphicsResource_t{};
		m_mipCount = 0;
	}

	m_target = target;
	m_size = size;
	m_format = format;
}

void Texture::create(const std::filesystem::path& file, int mipCount, const SamplerState& samplerState, bool createBindlessHandle, bool createBindlessImageHandle, bool createGraphicsResource)
{
	glm::ivec2 size;
	GLenum format;
	GLenum pixelFormat;
	GLenum pixelType;
	const auto data{ readImage(file, size, format, pixelFormat, pixelType) };

	create(GL_TEXTURE_2D, glm::ivec3{ size, 0 }, format, mipCount, samplerState, createBindlessHandle, createBindlessImageHandle, createGraphicsResource);

	const GLenum handle{ m_handle };
	GL_CHECK_ERROR(glTextureSubImage2D(handle, 0, 0, 0, size.x, size.y, pixelFormat, pixelType, data.get()));
	GL_CHECK_ERROR(glGenerateTextureMipmap(handle));
}

void Texture::destroy()
{
	if (!isEmpty())
	{
		const cudaGraphicsResource_t graphicsResource{ m_graphicsResource };

		if (graphicsResource != cudaGraphicsResource_t{})
		{
			CU_CHECK_ERROR(cudaGraphicsUnregisterResource(graphicsResource));
		}

		GL_CHECK_ERROR(glDeleteTextures(1, &m_handle));
	}
}

int getMaxMipCount(const glm::ivec3 size)
{
	return 1 + static_cast<int>(glm::log2(static_cast<float>(glm::max(size.x, glm::max(size.y, size.z)))));
}

glm::vec3 getMipSize(const glm::ivec3 base, const int mipLevel)
{
	return base / (1 << mipLevel);
}

}
