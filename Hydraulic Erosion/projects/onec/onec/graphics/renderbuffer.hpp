#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

namespace onec
{

class Renderbuffer
{
public:
	explicit Renderbuffer();
	explicit Renderbuffer(const glm::ivec2& size, const GLenum format);
	Renderbuffer(const Renderbuffer& other);
	Renderbuffer(Renderbuffer&& other) noexcept;

	~Renderbuffer();

	Renderbuffer& operator=(const Renderbuffer& other);
	Renderbuffer& operator=(Renderbuffer&& other) noexcept;

	void initialize(const glm::ivec2& size, const GLenum format);
	void release();

	void setName(const std::string_view& name);

	GLuint getHandle();
	GLenum getTarget() const;
	const glm::ivec2& getSize() const;
	GLenum getFormat() const;
	bool isEmpty() const;
private:
	GLuint m_handle;
	glm::ivec2 m_size;
	GLenum m_format;
};

}
