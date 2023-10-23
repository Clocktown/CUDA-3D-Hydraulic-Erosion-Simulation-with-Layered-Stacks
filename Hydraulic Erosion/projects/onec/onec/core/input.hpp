#pragma once

#include <glm/glm.hpp>
#include <array>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace onec
{

class Input
{
public:
	Input(const Input& other) = delete;
	Input(Input&& other) = delete;

	~Input() = default;

	Input& operator=(const Input& other) = delete;
	Input& operator=(Input&& other) = delete;

	void poll();

	void lockCursor();
	void unlockCursor();
	
	const glm::vec2& getMousePosition() const;
	const glm::vec2& getMouseDelta() const;
	const glm::vec2& getMouseScrollDelta() const;
	bool isKeyPressed(const int key) const;
	bool isKeyReleased(const int key) const;
	bool isKeyDown(const int key) const;
	bool isKeyUp(const int key) const;
	bool isCursorLocked() const;
private:
	static void onMouseEnter(GLFWwindow* const window, const int hasEntered);
	static void onMouseDrag(GLFWwindow* const window, const double x, const double y);
	static void onMouseScroll(GLFWwindow* const window, const double x, const double y);
	static void onMouseButton(GLFWwindow* const window, const int mouseButton, const int action, const int mods);
	static void onKey(GLFWwindow* const window, const int key, const int scancode, const int action, const int mods);

	explicit Input();

	void updateMousePosition();
	
	glm::vec2 m_mousePosition;
	glm::vec2 m_mouseDelta{ 0.0f };
	glm::vec2 m_mouseScrollDelta{ 0.0f };
	std::array<char, GLFW_KEY_LAST + 2> m_actions;
	std::array<bool, GLFW_KEY_LAST + 2> m_isKeyDown;
	bool m_isCursorLocked{ false };

	friend Input& getInput();
};

struct OnMouseEnter
{
	bool hasEntered;
};

struct OnMouseDrag
{
	glm::vec2 position;
	glm::vec2 delta;
};

struct OnMouseScroll
{
	glm::vec2 delta;
};

struct OnKey
{
	int key;
	int action;
};

Input& getInput();

}
