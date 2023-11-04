#pragma once

#include "shader.hpp"
#include <glad/glad.h>
#include <memory>
#include <string>

namespace onec
{

class Program
{
public:
	explicit Program();
	Program(const Program& other) = delete;
	Program(Program&& other) noexcept;

	~Program();

	Program& operator=(const Program& other) = delete;
	Program& operator=(Program&& other) noexcept;

	void use() const;
	void disuse() const;
	void link();
	void attachShader(const Shader& shader);
	void detachShader(const Shader& shader);

	void setName(const std::string_view& name);

	GLuint getHandle();
private:
	GLuint m_handle;
};

}
