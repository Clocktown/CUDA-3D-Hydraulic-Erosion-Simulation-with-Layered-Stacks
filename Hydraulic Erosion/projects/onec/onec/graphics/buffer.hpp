#pragma once

#include "../utility/span.hpp"
#include <glad/glad.h>
#include <string>

namespace onec
{

class Buffer
{
public:
	explicit Buffer();
	explicit Buffer(const int count);
	explicit Buffer(const Span<const std::byte>&& data);
	Buffer(const Buffer& other);
	Buffer(Buffer&& other) noexcept;

	~Buffer();

	Buffer& operator=(const Buffer& other);
	Buffer& operator=(Buffer&& other) noexcept;
	
	void bind(const GLenum target, const GLuint location);
	void unbind(const GLenum target, const GLuint location);
	void initialize(const int count);
	void initialize(const Span<const std::byte>&& data);
	void release();
	void upload(const Span<const std::byte>&& data);
	void upload(const Span<const std::byte>&& data, const int count);
	void upload(const Span<const std::byte>&& data, const int offset, const int count);
	void download(const Span<std::byte>&& data) const;
	void download(const Span<std::byte>&& data, const int count) const;
	void download(const Span<std::byte>&& data, const int offset, const int count) const;

	void setName(const std::string_view& name);

	GLuint getHandle();
	int getCount() const;
	bool isEmpty() const;
private:
	GLuint m_handle;
	int m_count;
};

}	
