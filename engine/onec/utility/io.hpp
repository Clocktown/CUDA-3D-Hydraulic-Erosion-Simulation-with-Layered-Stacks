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
std::string readShader(const std::filesystem::path& file, GLenum& type);
std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, int& channelCount, int requestedChannelCount = 0);
std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, GLenum& internalFormat, GLenum& pixelFormat, GLenum& pixelType);
std::unique_ptr<std::byte, decltype(&free)> readImage(const std::filesystem::path& file, glm::ivec2& size, cudaChannelFormatDesc& format);
void writeFile(const std::filesystem::path& file, std::string_view source) noexcept;
void writeImage(const std::filesystem::path& file, const Span<const std::byte>&& source, glm::ivec2 size, int channelCount);

}
