#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

namespace onec
{

class Sampler
{
public:
	explicit Sampler();
	Sampler(const Sampler& other) = delete;
	Sampler(Sampler&& other) noexcept;

	~Sampler();

	Sampler& operator=(const Sampler& other) = delete;
	Sampler& operator=(Sampler&& other) noexcept;

	void bind(const GLuint unit) const;
	void unbind(const GLuint unit) const;

	void setName(const std::string_view& name);
	void setMinFilter(const GLenum minFilter);
	void setMagFilter(const GLenum magFilter);
	void setWrapModeS(const GLenum wrapModeS);
	void setWrapModeT(const GLenum wrapModeT);
	void setWrapModeR(const GLenum wrapModeR);
	void setBorderColor(const glm::vec4& borderColor);
	void setLODBias(const float lodBias);
	void setMinLOD(const float minLOD);
	void setMaxLOD(const float maxLOD);
	void setMaxAnisotropy(const float maxAnisotropy);

	GLuint getHandle();
private:
	GLuint m_handle;
};

}
