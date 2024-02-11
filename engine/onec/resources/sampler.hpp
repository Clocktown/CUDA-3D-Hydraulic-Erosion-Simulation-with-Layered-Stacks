#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace onec
{

struct SamplerState
{
	GLenum minFilter{ GL_NEAREST_MIPMAP_LINEAR };
	GLenum magFilter{ GL_LINEAR };
	glm::vec<3, GLenum> wrapMode{ GL_REPEAT };
	glm::vec4 borderColor{ 0.0f };
	float levelOfDetailBias{ 0.0f };
	float minLevelOfDetail{ -1000.0f };
	float maxLevelOfDetail{ 1000.0f };
	float maxAnisotropy{ 1.0f };
};

class Sampler
{
public:
	explicit Sampler();
	explicit Sampler(const SamplerState& state);

	Sampler(const Sampler& other) = delete;
	Sampler(Sampler&& other) noexcept;

	~Sampler();

	Sampler& operator=(const Sampler& other) = delete;
	Sampler& operator=(Sampler&& other) noexcept;

	void initialize(const SamplerState& state);
	void release();

	GLuint getHandle();
	bool isEmpty() const;
private:
	void create(const SamplerState& state);
	void destroy();

	GLuint m_handle;
};

}
