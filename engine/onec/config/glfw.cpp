#include "glfw.hpp"
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <iostream>
#include <string>

namespace onec
{
namespace internal
{

void glfwCheckError(const char* const file, const int line)
{
	const char* description;
	const int error{ glfwGetError(&description) };
	
	if (error != GLFW_NO_ERROR)
	{
		std::string name;

		switch (error)
		{
		case GLFW_NOT_INITIALIZED:
		    name = "GLFW_NOT_INITIALIZED";
		    break;
		case GLFW_NO_CURRENT_CONTEXT:
			name = "GLFW_NO_CURRENT_CONTEXT";
			break;
		case GLFW_INVALID_ENUM:
			name = "GLFW_INVALID_ENUM";
			break;
		case GLFW_INVALID_VALUE:
			name = "GLFW_INVALID_VALUE";
			break;
		case GLFW_OUT_OF_MEMORY:
			name = "GLFW_OUT_OF_MEMORY";
			break;
		case GLFW_API_UNAVAILABLE:
			name = "GLFW_API_UNAVAILABLE";
			break;
		case GLFW_VERSION_UNAVAILABLE:
			name = "GLFW_VERSION_UNAVAILABLE";
			break;
		case GLFW_PLATFORM_ERROR:
			name = "GLFW_PLATFORM_ERROR";
			break;
		case GLFW_FORMAT_UNAVAILABLE:
			name = "GLFW_FORMAT_UNAVAILABLE";
			break;
		case GLFW_NO_WINDOW_CONTEXT:
			name = "GLFW_NO_WINDOW_CONTEXT";
			break;
		default:
			name = std::to_string(error);
		}

		std::cerr << "GLFW Error: " << name << "\n"
			      << "Description: " << description << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";
		
		std::exit(EXIT_FAILURE);
	}
}

}
}
