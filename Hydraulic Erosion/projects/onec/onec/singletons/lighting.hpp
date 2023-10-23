#pragma once

#include "../components/light.hpp"
#include "../device/lighting.hpp"
#include "../graphics/buffer.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>

namespace onec 
{

struct Lighting
{
	static constexpr GLuint uniformBufferLocation{ 1 };

	device::Lighting uniforms;
	Buffer uniformBuffer;
};

}
