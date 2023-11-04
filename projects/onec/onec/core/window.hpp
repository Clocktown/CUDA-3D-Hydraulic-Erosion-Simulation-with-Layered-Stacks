#pragma once

#include <glm/glm.hpp>
#include <string>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace onec
{

class Window
{
public:
	Window(const Window& other) = delete;
	Window(Window&& other) = delete;

	~Window();

	Window& operator=(const Window& other) = delete;
	Window& operator=(Window&& other) = delete;

	void setTitle(const std::string_view& title);
	
	GLFWwindow* getHandle();
	const std::string& getTitle() const;
	const glm::ivec2& getPosition() const;
	const glm::ivec2& getSize() const;
	const glm::ivec2& getFramebufferSize() const;
	const int getSampleCount() const;
	bool isFocused() const;
	bool isHovered() const;
	bool isMinimized() const;
	bool isMaximized() const;
	bool shouldClose() const;
private:
	static void onFocus(GLFWwindow* const window, int isFocused);
	static void onMinimize(GLFWwindow* const window, int isMinimized);
	static void onMaximize(GLFWwindow* const window, int isMaximized);
	static void onDrag(GLFWwindow* const window, const int x, const int y);
	static void onResize(GLFWwindow* const window, const int width, const int height);
	static void onFramebufferResize(GLFWwindow* const window, const int width, const int height);
	static Window& get(const std::string_view* const title, const glm::ivec2& size, const int sampleCount);

	explicit Window(const std::string_view* const title, const glm::ivec2& size, const int sampleCount);

	GLFWwindow* m_handle;
	std::string m_title;
	glm::ivec2 m_position;
	glm::ivec2 m_size;
	glm::ivec2 m_framebufferSize;
	int m_sampleCount;

	friend Window& createWindow(const std::string_view& title, const glm::ivec2& size, const int sampleCount);
	friend Window& getWindow();
};

struct OnWindowFocus
{
	bool isFocused;
};

struct OnWindowMinimize
{
	bool isMinimized;
};

struct OnWindowMaximize
{
	bool isMaximized;
};

struct OnWindowDrag
{
	glm::ivec2 position;
};

struct OnWindowResize
{
	glm::ivec2 size;
};

struct OnFramebufferResize
{
	glm::ivec2 size;
};

Window& createWindow(const std::string_view& title = "Window", const glm::ivec2& size = glm::ivec2{ 1280, 720 }, const int sampleCount = 0);
Window& getWindow();

}
