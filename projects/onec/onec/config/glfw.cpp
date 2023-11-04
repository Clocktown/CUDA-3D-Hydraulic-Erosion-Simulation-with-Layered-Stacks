#include "glfw.hpp"
#include <GLFW/glfw3.h>
#include <iostream>

namespace onec
{
namespace internal
{

void glfwCheckError(const char* const file, const int line)
{
	const char* description;
	
	if (glfwGetError(&description) != GLFW_NO_ERROR)
	{
		std::cerr << "GLFW Error: " << description << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";
		
		std::exit(EXIT_FAILURE);
	}
}

}
}
