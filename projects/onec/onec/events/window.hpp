#pragma once

#include <glm/glm.hpp>

namespace onec 
{

struct OnWindowClose
{

};

struct OnWindowFocus
{
	bool isWindowFocused;
};

struct OnWindowMaximize
{
	bool isWindowMaximized;
};

struct OnWindowMinimize
{
	bool isWindowMinimized;
};

struct OnWindowMove
{
	glm::ivec2 windowPosition;
};

struct OnWindowResize
{
	glm::ivec2 windowSize;
};

struct OnFramebufferResize
{
	glm::ivec2 framebufferSize;
};

struct OnMouseEnter
{
	bool hasMouseEntered;
};

struct OnMouseMove
{
	glm::vec2 mousePosition;
};

struct OnMouseScroll
{
	glm::vec2 mouseScrollDelta;
};

struct OnKeyInput
{
	int key;
	int action;
};

struct OnCharInput
{
	unsigned int unicode;
};

}
