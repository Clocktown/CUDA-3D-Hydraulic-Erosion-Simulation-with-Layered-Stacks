#include "input.hpp"
#include "window.hpp"
#include "world.hpp"
#include "../config/glfw.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <array>

namespace onec
{

void Input::onMouseEnter([[maybe_unused]] GLFWwindow* const window, const int hasEntered)
{
	OnMouseEnter event{ hasEntered == GLFW_TRUE };

	if (event.hasEntered)
	{
		Input& input{ getInput() };
		input.updateMousePosition();
	}

	World& world{ getWorld() };
	world.dispatch(event);
}

void Input::onMouseDrag([[maybe_unused]] GLFWwindow* const window, const double x, const double y)
{
	Input& input{ getInput() };
	const glm::vec2 position{ x, y };
	
	if (!ImGui::GetIO().WantCaptureMouse)
	{
		const glm::vec2 delta{ position - input.m_mousePosition };

		input.m_mouseDelta += delta;

		World& world{ getWorld() };
		world.dispatch(OnMouseDrag{ position, delta });
	}

	input.m_mousePosition = position;
}

void Input::onMouseScroll([[maybe_unused]] GLFWwindow* const window, const double x, const double y)
{
	if (!ImGui::GetIO().WantCaptureMouse)
	{
		Input& input{ getInput() };

		const glm::vec2 scroll{ x, y };
		input.m_mouseScrollDelta += scroll;

		World& world{ getWorld() };
		world.dispatch(OnMouseScroll{ scroll });
	}
}

void Input::onMouseButton([[maybe_unused]] GLFWwindow* const window, const int mouseButton, const int action, [[maybe_unused]] const int mods)
{
	if (!ImGui::GetIO().WantCaptureMouse)
	{
		if (mouseButton != GLFW_KEY_UNKNOWN)
		{
			Input& input{ getInput() };

			const size_t index{ static_cast<size_t>(mouseButton) + 1 };
			input.m_actions[index] = static_cast<char>(action);
			input.m_isKeyDown[index] = action == GLFW_PRESS;
		}

		World& world{ getWorld() };
		world.dispatch(OnKey{ mouseButton, action });
	}
}

void Input::onKey([[maybe_unused]] GLFWwindow* const window, const int key, [[maybe_unused]] const int scancode, const int action, [[maybe_unused]] const int mods)
{
	if (!ImGui::GetIO().WantCaptureKeyboard)
	{
		if (key != GLFW_KEY_UNKNOWN && action != GLFW_REPEAT)
		{
			Input& input{ getInput() };

			const size_t index{ static_cast<size_t>(key) + 1 };
			input.m_actions[index] = static_cast<char>(action);	
			input.m_isKeyDown[index] = action == GLFW_PRESS;
		}

		World& world{ getWorld() };
		world.dispatch(OnKey{ key, action });
	}
}

Input::Input()
{
	GLFWwindow* const window{ getWindow().getHandle() };
	
	GLFW_CHECK_ERROR(glfwSetCursorEnterCallback(window, &onMouseEnter));
	GLFW_CHECK_ERROR(glfwSetCursorPosCallback(window, &onMouseDrag));
	GLFW_CHECK_ERROR(glfwSetScrollCallback(window, &onMouseScroll));
	GLFW_CHECK_ERROR(glfwSetMouseButtonCallback(window, &onMouseButton));
	GLFW_CHECK_ERROR(glfwSetKeyCallback(window, &onKey));

	updateMousePosition();

	memset(m_actions.data(), -1, m_actions.size() * sizeof(char));
	memset(m_isKeyDown.data(), 0, m_isKeyDown.size() * sizeof(char));
}

void Input::poll()
{
	m_mouseDelta = glm::vec2{ 0.0f };
	m_mouseScrollDelta = glm::vec2{ 0.0f };

	memset(m_actions.data(), -1, m_actions.size() * sizeof(char));

	GLFW_CHECK_ERROR(glfwPollEvents());
}

void Input::updateMousePosition()
{
	glm::dvec2 mousePosition;
	GLFW_CHECK_ERROR(glfwGetCursorPos(getWindow().getHandle(), &mousePosition.x, &mousePosition.y));
	m_mousePosition = mousePosition;
}

void Input::lockCursor()
{
	m_isCursorLocked = true;
	GLFW_CHECK_ERROR(glfwSetInputMode(getWindow().getHandle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED));
}

void Input::unlockCursor()
{
	m_isCursorLocked = false;
	GLFW_CHECK_ERROR(glfwSetInputMode(getWindow().getHandle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL));
}

const glm::vec2& Input::getMousePosition() const
{
	return m_mousePosition;
}

const glm::vec2& Input::getMouseDelta() const
{
	return m_mouseDelta;
}

const glm::vec2& Input::getMouseScrollDelta() const
{
	return m_mouseScrollDelta;
}

bool Input::isKeyPressed(const int key) const
{
	ONEC_ASSERT(key >= GLFW_KEY_UNKNOWN && key <= GLFW_KEY_LAST, "Key must be valid");

	return m_actions[static_cast<size_t>(key) + 1] == GLFW_PRESS;
}

bool Input::isKeyReleased(const int key) const
{
	ONEC_ASSERT(key >= GLFW_KEY_UNKNOWN && key <= GLFW_KEY_LAST, "Key must be valid");

    return m_actions[static_cast<size_t>(key) + 1] == GLFW_RELEASE;
}

bool Input::isKeyDown(const int key) const
{
	ONEC_ASSERT(key >= GLFW_KEY_UNKNOWN && key <= GLFW_KEY_LAST, "Key must be valid");

	return m_isKeyDown[static_cast<size_t>(key) + 1];
}

bool Input::isKeyUp(const int key) const
{
	return !isKeyDown(key);
}

bool Input::isCursorLocked() const
{
	return m_isCursorLocked;
}

Input& getInput()
{
	static Input input;
	return input;
}

}
