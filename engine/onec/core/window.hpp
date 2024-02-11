#pragma once

#include <glm/glm.hpp>
#include <string>
#include <array>

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

	void open();
	void close();
	void maximize();
	void minimize();
	void restore();
	void pollEvents();
	void swapBuffers();

	void setTitle(std::string_view title);
	void setPosition(glm::ivec2 position);
	void setSize(glm::ivec2 size);
	void setSwapInterval(int swapInterval);
	void setCursorLocked(bool cursorLocked);

	GLFWwindow* getHandle();
	const std::string& getTitle() const;
	glm::ivec2 getPosition() const;
	glm::ivec2 getSize() const;
	glm::ivec2 getFramebufferSize() const;
	int getSampleCount() const;
	int getSwapInterval() const;
	glm::vec2 getMousePosition() const;
	glm::vec2 getMouseDelta() const;
	glm::vec2 getMouseScrollDelta() const;
	const std::u32string& getText() const;
	bool isOpen() const;
	bool isFocused() const;
	bool isHovered() const;
	bool isMaximized() const;
	bool isMinimized() const;
	bool isKeyPressed(int key) const;
	bool isKeyReleased(int key) const;
	bool isKeyDown(int key) const;
	bool isKeyUp(int key) const;
	bool isCursorLocked() const;
private:
	static void onMove(GLFWwindow* window, int x, int y);
	static void onResize(GLFWwindow* window, int width, int height);
	static void onFramebufferResize(GLFWwindow* window, int width, int height);
	static void onMouseEnter(GLFWwindow* window, int hasMouseEntered);
	static void onMouseMove(GLFWwindow* window, double x, double y);
	static void onMouseScroll(GLFWwindow* window, double x, double y);
	static void onMouseButtonInput(GLFWwindow* window, int mouseButton, int action, int modifier);
	static void onKeyInput(GLFWwindow* window, int key, int scancode, int action, int modifier);
	static void onCharInput(GLFWwindow* window, unsigned int unicode);
	static Window& get(const std::string_view* title, glm::ivec2 size, int sampleCount);

	explicit Window(const std::string_view* title, glm::ivec2 size, int sampleCount);

	void initializeOpenGL();
	void initializeImGui();
	void initializeCUDA();
	void updateMousePosition();

	GLFWwindow* m_handle;
	std::string m_title;
	glm::ivec2 m_position;
	glm::ivec2 m_size;
	glm::ivec2 m_framebufferSize;
	int m_sampleCount;
	int m_swapInterval{ 1 };
	glm::vec2 m_mousePosition;
	glm::vec2 m_mouseDelta{ 0.0f };
	glm::vec2 m_mouseScrollDelta{ 0.0f };
	std::array<char, GLFW_KEY_LAST + 1> m_actions;
	std::array<bool, GLFW_KEY_LAST + 1> m_keysDown;
	std::u32string m_text;
	
	friend Window& createWindow(std::string_view title, glm::ivec2 size, int sampleCount);
	friend Window& getWindow();
};

Window& createWindow(std::string_view title = "Window", glm::ivec2 size = glm::ivec2{ 1280, 720 }, int sampleCount = 0);
Window& getWindow();

}
