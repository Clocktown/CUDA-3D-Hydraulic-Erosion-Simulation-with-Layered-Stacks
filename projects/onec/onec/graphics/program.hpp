#pragma once

#include "../utility/span.hpp"
#include <glad/glad.h>
#include <filesystem>

namespace onec
{

class Program
{
public:
	explicit Program();
	explicit Program(const Span<const std::filesystem::path>&& files);
	Program(const Program& other) = delete;
	Program(Program&& other) noexcept;

	~Program();

	void initialize(const Span<const std::filesystem::path>&& files);
	void release();

	Program& operator=(const Program& other) = delete;
	Program& operator=(Program&& other) noexcept;

	GLuint getHandle();
	bool isEmpty() const;
private:
	void create(const Span<const std::filesystem::path>&& files);
	void destroy();

	GLuint m_handle;
};

}
