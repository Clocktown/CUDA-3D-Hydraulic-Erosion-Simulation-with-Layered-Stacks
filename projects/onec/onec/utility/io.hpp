#pragma once

#include "span.hpp"
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <memory>
#include <string>

namespace onec
{

std::string readFile(const std::filesystem::path& file);
std::string readShader(const std::filesystem::path& file);
std::string readShader(const std::filesystem::path& file, GLenum& type);
std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, int& channelCount, const int requestedChannelCount = 0);
std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, GLenum& internalFormat, GLenum& pixelFormat, GLenum& pixelType);
std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, cudaChannelFormatDesc& format);
void writeImage(const std::filesystem::path& file, const Span<const std::byte>&& data, const glm::ivec2& size, const int channelCount);
std::string getDateTime(const std::string_view& format = "%Y-%m-%d %H-%M-%S");

}
