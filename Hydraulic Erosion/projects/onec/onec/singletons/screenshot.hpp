#pragma once

#include "../core/application.hpp"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace onec
{

struct Screenshot
{
	std::string name{ "Screenshot" };
	std::filesystem::path folder{ std::filesystem::current_path() };
};

}
