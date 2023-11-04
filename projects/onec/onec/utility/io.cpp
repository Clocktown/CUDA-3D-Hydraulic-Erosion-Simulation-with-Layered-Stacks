#include "io.hpp"
#include "span.hpp"
#include "../config/config.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <ctime>
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
	std::ifstream stream{ file, std::ios::in };

	ONEC_ASSERT(stream.is_open(), "Failed to open file (" + file.string() + ")");

	std::string data;
	std::string line;

	while (std::getline(stream, line))
	{
		ONEC_ASSERT(!stream.fail() && !stream.bad(), "Failed to read file (" + file.string() + ")");

		std::getline(stream, line);
		data += line + "\n";
	}

	return data;
}

std::string& readShader(const std::filesystem::path& file, std::unordered_map<std::filesystem::path, std::string>& includes, const std::regex& regex)
{
	ONEC_ASSERT(file == file.lexically_normal(), "File must be lexically normal");

	std::ifstream stream{ file, std::ios::in };

	ONEC_ASSERT(stream.is_open(), "Failed to open file (" + file.string() + ")");

	auto it{ includes.find(file) };

	if (it != includes.end())
	{
		return it->second;
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
			it = includes.find(include);

			source += "#line 1 " + std::to_string(static_cast<int>(includes.size() - 1)) + "\n";

			if (it != includes.end())
			{
				source += it->second;
			}
			else
			{
				source += readShader(include, includes, regex);
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

std::string readShader(const std::filesystem::path& file)
{
	std::unordered_map<std::filesystem::path, std::string> includes;
	const std::regex regex{ " *# *include *\"(.*)\"" };

	return readShader(file.lexically_normal(), includes, regex);
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
		ONEC_IF_RELEASE(type = GL_NONE);
	}

	return readShader(file);
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
			const size_t pixelCount{ static_cast<size_t>(size.x) * static_cast<size_t>(size.y) };
			float* const modifiedData{ static_cast<float*>(malloc(pixelCount * sizeof(float4))) };

			for (size_t i{ 0 }, j{ 0 }; i < 4 * pixelCount; i += 4)
			{
				modifiedData[i] = data[j++];
				modifiedData[i + 1] = data[j++];
				modifiedData[i + 2] = data[j++];
				modifiedData[i + 3] = 1.0f;
			}

			free(data);
			data = modifiedData;
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
			const size_t byteCount{ 4 * count };
			unsigned char* const modifiedData{ static_cast<unsigned char*>(malloc(byteCount)) };

			ONEC_ASSERT(modifiedData != nullptr, "Failed to allocate memory");

			size_t i{ 0 };
			size_t j{ 0 };

			while (i < byteCount)
			{
				modifiedData[i++] = data[j++];
				modifiedData[i++] = data[j++];
				modifiedData[i++] = data[j++];
				modifiedData[i++] = 255;
			}

			free(data);
			data = modifiedData;
		}
		case 4:
			format = cudaCreateChannelDesc<uchar4>();
			break;
		default:
			ONEC_ERROR("Channel count must be between 1 and 4");

			break;
		}

		return std::unique_ptr<std::byte, decltype(&free)>{ reinterpret_cast<std::byte*>(data), & free };
	}
}

void writeImage(const std::filesystem::path& file, const Span<const std::byte>&& data, const glm::ivec2& size, const int channelCount)
{
	ONEC_ASSERT(file.extension() == ".png", "File extension must be .png");

	stbi_flip_vertically_on_write(1);
	[[maybe_unused]] int status;

	const std::filesystem::path extension{ file.extension() };

	if (extension == ".png")
	{
		status = stbi_write_png(file.string().c_str(), size.x, size.y, channelCount, data.getData(), 0);
	}
	else if (extension == ".jpg")
	{
		status = stbi_write_jpg(file.string().c_str(), size.x, size.y, channelCount, data.getData(), 0);
	}
	else if (extension == ".bmp")
	{
		status = stbi_write_bmp(file.string().c_str(), size.x, size.y, channelCount, data.getData());
	}
	else if (extension == ".tga")
	{
		status = stbi_write_tga(file.string().c_str(), size.x, size.y, channelCount, data.getData());
	}
	else if (extension == ".hdr")
	{
		status = stbi_write_hdr(file.string().c_str(), size.x, size.y, channelCount, reinterpret_cast<const float*>(data.getData()));
	}
	else
	{
		ONEC_ERROR("File extension must either be .png, .jpg, .bmp, .tga or .hdr");
	}

	ONEC_ASSERT(status == 1, "Failed to write file (" + file.string() + ")");
}

std::string getDateTime(const std::string_view& format)
{
	if (format.empty())
	{
		return std::string{};
	}

	tm tm;
	const time_t time{ std::time(nullptr) };
	localtime_s(&tm, &time);

	std::vector<char> dateTime(32);
	size_t count;

	do
	{
		count = strftime(dateTime.data(), dateTime.size(), format.data(), &tm);

		if (count == 0)
		{
			dateTime.resize(2 * dateTime.size());
		}
		else
		{
			break;
		}
	}
	while (true);

	return std::string{ dateTime.data(), count };
}

}
