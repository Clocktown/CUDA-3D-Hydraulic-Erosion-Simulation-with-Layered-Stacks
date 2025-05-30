#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <limits>

namespace onec
{

struct RenderState
{
	void use();
	void disuse();

	glm::bvec4 colorMask{ true };
	bool depthTest{ true };
	GLenum depthFunction{ GL_LESS };
	bool depthMask{ true };
	bool stencilTest{ false };
	GLenum stencilFunction{ GL_ALWAYS };
	int stencilReference{ 0 };
	unsigned int stencilMask{ std::numeric_limits<unsigned int>::max() };
	GLenum stencilPassOperation{ GL_KEEP };
	GLenum stencilFailOperation{ GL_KEEP };
	GLenum stencilDepthFailOperation{ GL_KEEP };
	bool blending{ false };
	GLenum blendEquation{ GL_FUNC_ADD };
	GLenum blendSourceFactor{ GL_ONE };
	GLenum blendDestinationFactor{ GL_ZERO };
	glm::vec4 blendColor{ 0.0f };
	bool culling{ true };
	GLenum cullFace{ GL_BACK };
	GLenum polygonMode{ GL_FILL };
};

}
