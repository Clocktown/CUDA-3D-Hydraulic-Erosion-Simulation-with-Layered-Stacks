#pragma once

#include <glad/glad.h>
#include <filesystem>
#include <string>

namespace onec
{

class Shader
{
public:
	explicit Shader(const GLenum type);
	explicit Shader(const std::filesystem::path& file);
	Shader(const Shader& other) = delete;
	Shader(Shader&& other) noexcept;

	~Shader();

	Shader& operator=(const Shader& other) = delete;
	Shader& operator=(Shader&& other) noexcept;

	void compile();

	void setName(const std::string_view& name);
	void setSource(const std::string_view& source);

	GLuint getHandle();
	GLenum getType() const;
private:
	GLuint m_handle;
	GLenum m_type;
};

}
