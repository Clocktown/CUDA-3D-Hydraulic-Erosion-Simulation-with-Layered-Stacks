#include "io.hpp"
#include "span.hpp"
#include "../config/config.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <regex>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace onec
{

std::string readFile(const std::filesystem::path& file)
{
	std::ifstream stream{ file, std::ios::in | std::ios::binary };

	ONEC_ASSERT(stream.is_open(), "Failed to open " + file.string());

	const std::string destination{ std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>{} };

	ONEC_ASSERT(!stream.fail() && !stream.bad(), "Failed to read " + file.string());

	return destination;
}

std::string& readShader(const std::regex& regex, const std::filesystem::path& file, std::unordered_map<std::filesystem::path, std::string>& includes)
{
	ONEC_ASSERT(file == file.lexically_normal(), "File must be lexically normal");

	std::ifstream stream{ file, std::ios::in };

	ONEC_ASSERT(stream.is_open(), "Failed to open file (" + file.string() + ")");

	auto iterator{ includes.find(file) };

	if (iterator != includes.end())
	{
		return iterator->second;
	}

	const std::string fileNumber{ std::to_string(includes.size()) };

	std::string source;
	std::string line;
	int lineCount{ 1 };

	while (std::getline(stream, line))
	{
		ONEC_ASSERT(!stream.fail() && !stream.bad(), "Failed to read file (" + file.string() + ")");

		++lineCount;

		std::smatch match;

		if (std::regex_match(line, match, regex))
		{
			const std::filesystem::path include{ (file.parent_path() / match[1].str()).lexically_normal() };
			iterator = includes.find(include);

			source += "#line 1 " + std::to_string(static_cast<int>(includes.size() - 1)) + "\n";

			if (iterator != includes.end())
			{
				source += iterator->second;
			}
			else
			{
				source += readShader(regex, include, includes);
			}

			source += "#line " + std::to_string(lineCount) + " " + fileNumber + "\n";
		}
		else
		{
			source += line + "\n";
		}
	}
	
	return includes.emplace(file, std::move(source)).first->second;
}

std::string readShader(const std::filesystem::path& file, GLenum& type)
{
	const std::filesystem::path extension{ file.extension() };

	if (extension == ".vert")
	{
		type = GL_VERTEX_SHADER;
	}
	else if (extension == ".tesc")
	{
		type = GL_TESS_CONTROL_SHADER;
	}
	else if (extension == ".tese")
	{
		type = GL_TESS_EVALUATION_SHADER;
	}
	else if (extension == ".geom")
	{
		type = GL_GEOMETRY_SHADER;
	}
	else if (extension == ".frag")
	{
		type = GL_FRAGMENT_SHADER;
	}
	else if (extension == ".comp")
	{
		type = GL_COMPUTE_SHADER;
	}
	else
	{
		ONEC_ERROR("File extension must either be .vert, .tesc, .tese, .geom, .frag or .comp");

		type = GL_NONE;
	}

	const std::regex regex{ " *# *include *\"(.*)\"" };
	std::unordered_map<std::filesystem::path, std::string> includes;
	
	return readShader(regex, file.lexically_normal(), includes);
}

std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, int& channelCount, const int requestedChannelCount)
{
	stbi_set_flip_vertically_on_load(1);
	std::byte* data;

	if (file.extension() == ".hdr")
	{
		data = reinterpret_cast<std::byte*>(stbi_loadf(file.string().c_str(), &size.x, &size.y, &channelCount, requestedChannelCount));
	}
	else
	{
		data = reinterpret_cast<std::byte*>(stbi_load(file.string().c_str(), &size.x, &size.y, &channelCount, requestedChannelCount));
	}

	ONEC_ASSERT(data != nullptr, "Failed to read file (" + file.string() + ")");

	return std::unique_ptr<std::byte, decltype(&free)>{ data, &free };
}

std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, GLenum& internalFormat, GLenum& pixelFormat, GLenum& pixelType)
{
	stbi_set_flip_vertically_on_load(1);
	std::byte* data;
	int channelCount;

	if (file.extension() == ".hdr")
	{
		data = reinterpret_cast<std::byte*>(stbi_loadf(file.string().c_str(), &size.x, &size.y, &channelCount, 0));

		ONEC_ASSERT(data != nullptr, "Failed to read file (" + file.string() + ")");

		switch (channelCount)
		{
		case 1:
			internalFormat = GL_R32F;
			pixelFormat = GL_RED;
			break;
		case 2:
			internalFormat = GL_RG32F;
			pixelFormat = GL_RG;
			break;
		case 3:
			internalFormat = GL_RGB32F;
			pixelFormat = GL_RGB;
			break;
		case 4:
			internalFormat = GL_RGBA32F;
			pixelFormat = GL_RGBA;
			break;
		default:
			ONEC_ERROR("Channel count must be between 1 and 4");

			break;
		}

		pixelType = GL_FLOAT;
	}
	else
	{
		data = reinterpret_cast<std::byte*>(stbi_load(file.string().c_str(), &size.x, &size.y, &channelCount, 0));

		ONEC_ASSERT(data != nullptr, "Failed to read file (" + file.string() + ")");

		switch (channelCount)
		{
		case 1:
			internalFormat = GL_R8;
			pixelFormat = GL_RED;
			break;
		case 2:
			internalFormat = GL_RG8;
			pixelFormat = GL_RG;
			break;
		case 3:
			internalFormat = GL_RGB8;
			pixelFormat = GL_RGB;
			break;
		case 4:
			internalFormat = GL_RGBA8;
			pixelFormat = GL_RGBA;
			break;
		default:
			ONEC_ERROR("Channel count must be between 1 and 4");

			break;
		}

		pixelType = GL_UNSIGNED_BYTE;
	}

	return std::unique_ptr<std::byte, decltype(&free)>{ data, &free };
}

std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, cudaChannelFormatDesc& format)
{
	stbi_set_flip_vertically_on_load(1);
	int channelCount;

	if (file.extension() == ".hdr")
	{
		float* data{ stbi_loadf(file.string().c_str(), &size.x, &size.y, &channelCount, 0) };

		ONEC_ASSERT(data != nullptr, "Failed to read file (" + file.string() + ")");

		switch (channelCount)
		{
		case 1:
			format = cudaCreateChannelDesc<float>();
			break;
		case 2:
			format = cudaCreateChannelDesc<float2>();
			break;
		case 3:
		{
			const size_t count{ static_cast<size_t>(size.x) * static_cast<size_t>(size.y) };
			const size_t floatCount{ 4 * count };
			float* const paddedData{ static_cast<float*>(malloc(floatCount * sizeof(float))) };

			ONEC_ASSERT(paddedData != nullptr, "Failed to allocate memory");

			size_t i{ 0 };
			size_t j{ 0 };

			while (i < floatCount)
			{
				paddedData[i++] = data[j++];
				paddedData[i++] = data[j++];
				paddedData[i++] = data[j++];
				paddedData[i++] = 1.0f;
			}

			free(data);
			data = paddedData;

			[[fallthrough]];
		}
		case 4:
			format = cudaCreateChannelDesc<float4>();
			break;
		default:
			ONEC_ERROR("Channel count must be between 1 and 4");

			break;
		}

		return std::unique_ptr<std::byte, decltype(&free)>{ reinterpret_cast<std::byte*>(data), & free };
	}
	else
	{
		unsigned char* data{ stbi_load(file.string().c_str(), &size.x, &size.y, &channelCount, 0) };

		ONEC_ASSERT(data != nullptr, "Failed to read file (" + file.string() + ")");

		switch (channelCount)
		{
		case 1:
			format = cudaCreateChannelDesc<unsigned char>();
			break;
		case 2:
			format = cudaCreateChannelDesc<uchar2>();
			break;
		case 3:
		{
			const size_t count{ static_cast<size_t>(size.x) * static_cast<size_t>(size.y) };
			const size_t byteCount{ count * sizeof(uchar4) };
			unsigned char* const paddedData{ static_cast<unsigned char*>(malloc(byteCount)) };

			ONEC_ASSERT(paddedData != nullptr, "Failed to allocate memory");

			size_t i{ 0 };
			size_t j{ 0 };

			while (i < byteCount)
			{
				paddedData[i++] = data[j++];
				paddedData[i++] = data[j++];
				paddedData[i++] = data[j++];
				paddedData[i++] = 255;
			}

			free(data);
			data = paddedData;

			[[fallthrough]];
		}
		case 4:
			format = cudaCreateChannelDesc<uchar4>();
			break;
		default:
			ONEC_ERROR("Channel count must be between 1 and 4");

			break;
		}

		return std::unique_ptr<std::byte, decltype(&free)>{ reinterpret_cast<std::byte*>(data), &free };
	}
}

void writeFile(const std::filesystem::path& file, std::string_view source) noexcept
{
	std::ofstream stream{ file, std::ios::out | std::ios::binary };

	ONEC_ASSERT(stream.is_open(), "Failed to open " + file.string());

	stream << source;

	ONEC_ASSERT(!stream.fail() && !stream.bad(), "Failed to write " + file.string());
}

void writeImage(const std::filesystem::path& file, const Span<const std::byte>&& source, const glm::ivec2 size, const int channelCount)
{
	stbi_flip_vertically_on_write(1);
	[[maybe_unused]] int status;

	const std::filesystem::path extension{ file.extension() };

	if (extension == ".png")
	{
		status = stbi_write_png(file.string().c_str(), size.x, size.y, channelCount, source.getData(), 0);
	}
	else if (extension == ".jpg")
	{
		status = stbi_write_jpg(file.string().c_str(), size.x, size.y, channelCount, source.getData(), 0);
	}
	else if (extension == ".bmp")
	{
		status = stbi_write_bmp(file.string().c_str(), size.x, size.y, channelCount, source.getData());
	}
	else if (extension == ".tga")
	{
		status = stbi_write_tga(file.string().c_str(), size.x, size.y, channelCount, source.getData());
	}
	else if (extension == ".hdr")
	{
		status = stbi_write_hdr(file.string().c_str(), size.x, size.y, channelCount, reinterpret_cast<const float*>(source.getData()));
	}
	else
	{
		ONEC_ERROR("File extension must either be .png, .jpg, .bmp, .tga or .hdr");
	}

	ONEC_ASSERT(status == 1, "Failed to write file (" + file.string() + ")");
}

}
