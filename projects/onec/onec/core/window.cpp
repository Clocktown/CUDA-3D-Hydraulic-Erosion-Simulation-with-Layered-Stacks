#include "window.hpp"
#include "world.hpp"
#include "../config/config.hpp"
#include "../config/glfw.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include <cstdlib>
#include <string>
#include <array>

namespace onec
{

void Window::onClose([[maybe_unused]] GLFWwindow* const window)
{
    World& world{ getWorld() };
    world.dispatch<OnWindowClose>();
}

void Window::onFocus([[maybe_unused]] GLFWwindow* const window, const int isFocused)
{
    World& world{ getWorld() };
    world.dispatch(OnWindowFocus{ isFocused == GLFW_TRUE });
}

void Window::onMaximize([[maybe_unused]] GLFWwindow* const window, const int isMaximized)
{
    World& world{ getWorld() };
    world.dispatch(OnWindowMaximize{ isMaximized == GLFW_TRUE });
}

void Window::onMinimize([[maybe_unused]] GLFWwindow* const window, const int isMinimized)
{
    World& world{ getWorld() };
    world.dispatch(OnWindowMinimize{ isMinimized == GLFW_TRUE });
}

void Window::onMove([[maybe_unused]] GLFWwindow* const, const int x, const int y)
{
    Window& window{ getWindow() };

    const OnWindowMove event{ glm::ivec2{ x, y } };
    window.m_position = event.windowPosition;

    World& world{ getWorld() };
    world.dispatch(event);
}

void Window::onResize([[maybe_unused]] GLFWwindow* const, const int width, const int height)
{
    const OnWindowResize event{ glm::ivec2{ width, height } };

    Window& window{ getWindow() };
    window.m_size = event.windowSize;

    World& world{ getWorld() };
    world.dispatch(event);
}

void Window::onFramebufferResize([[maybe_unused]] GLFWwindow* const, const int width, const int height)
{
    const OnFramebufferResize event{ glm::ivec2{ width, height } };

    Window& window{ getWindow() };
    window.m_framebufferSize = event.framebufferSize;

    World& world{ getWorld() };
    world.dispatch(event);
}

void Window::onMouseEnter([[maybe_unused]] GLFWwindow* const, const int hasMouseEntered)
{
    const OnMouseEnter event{ hasMouseEntered == GLFW_TRUE };

    if (event.hasMouseEntered)
    {
        Window& window{ getWindow() };
        window.updateMousePosition();
    }

    World& world{ getWorld() };
    world.dispatch(event);
}

void Window::onMouseMove([[maybe_unused]] GLFWwindow* const, const double x, const double y)
{
    Window& window{ getWindow() };

    const OnMouseMove event{ glm::vec2{ x, y } };

    if (!ImGui::GetIO().WantCaptureMouse)
    {
        window.m_mouseDelta += event.mousePosition - window.m_mousePosition;

        World& world{ getWorld() };
        world.dispatch(event);
    }

    window.m_mousePosition = event.mousePosition;
}

void Window::onMouseScroll([[maybe_unused]] GLFWwindow* const, const double x, const double y)
{
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        const OnMouseScroll event{ glm::vec2{ x, y } };

        Window& window{ getWindow() };
        window.m_mouseScrollDelta += event.mouseScrollDelta;

        World& world{ getWorld() };
        world.dispatch(event);
    }
}

void Window::onMouseButtonInput([[maybe_unused]] GLFWwindow* const window, const int mouseButton, const int action, const int modifier)
{
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        if (mouseButton != GLFW_KEY_UNKNOWN)
        {
            Window& window{ getWindow() };

            const std::size_t index{ static_cast<std::size_t>(mouseButton) };
            window.m_actions[index] = static_cast<char>(action);
            window.m_keysDown[index] = action == GLFW_PRESS;
        }

        World& world{ getWorld() };
        world.dispatch(OnKeyInput{ mouseButton, action, modifier });
    }
}

void Window::onKeyInput([[maybe_unused]] GLFWwindow* const, const int key, [[maybe_unused]] const int scancode, const int action, const int modifier)
{
    if (!ImGui::GetIO().WantCaptureKeyboard)
    {
        if (key != GLFW_KEY_UNKNOWN && action != GLFW_REPEAT)
        {
            Window& window{ getWindow() };

            const std::size_t index{ static_cast<std::size_t>(key) };
            window.m_actions[index] = static_cast<char>(action);
            window.m_keysDown[index] = action == GLFW_PRESS;
        }

        World& world{ getWorld() };
        world.dispatch(OnKeyInput{ key, action, modifier });
    }
}

void Window::onCharInput([[maybe_unused]] GLFWwindow* const, const unsigned int unicode)
{
    if (!ImGui::GetIO().WantTextInput)
    {
        Window& window{ getWindow() };
        window.m_inputText.push_back(unicode);
        
        World& world{ getWorld() };
        world.dispatch(OnCharInput{ unicode });
    }
}

Window& Window::get(const std::string_view* const title, const glm::ivec2 size, const int sampleCount)
{
	static Window window{ title, size, sampleCount };
	return window;
}

Window::Window(const std::string_view* const title, const glm::ivec2 size, const int sampleCount) :
    m_size{ size },
    m_sampleCount{ sampleCount }
{
    ONEC_ASSERT(title != nullptr, "Failed to get window");
    ONEC_ASSERT(sampleCount >= 0, "Sample count must be greater than or equal to 0");

    [[maybe_unused]] const int status{ glfwInit() };

    ONEC_ASSERT(status == GLFW_TRUE, "Failed to initialize GLFW");

    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_SAMPLES, sampleCount));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE));

    ONEC_IF_DEBUG(GLFW_CHECK_ERROR(glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE));)
 
    GLFWwindow* const handle{ glfwCreateWindow(size.x, size.y, title->data(), nullptr, nullptr) };

    ONEC_ASSERT(handle != nullptr, "Failed to create window");

    m_handle = handle;
    m_title = *title;
    m_inputText.reserve(32);

    updateMousePosition();
    memset(m_actions.data(), -1, m_actions.size() * sizeof(char));
    memset(m_keysDown.data(), 0, m_keysDown.size() * sizeof(bool));

    GLFW_CHECK_ERROR(glfwSetWindowUserPointer(handle, this));
    GLFW_CHECK_ERROR(glfwSetWindowCloseCallback(handle, &onClose));
    GLFW_CHECK_ERROR(glfwSetWindowFocusCallback(handle, &onFocus));
    GLFW_CHECK_ERROR(glfwSetWindowMaximizeCallback(handle, &onMaximize));
    GLFW_CHECK_ERROR(glfwSetWindowIconifyCallback(handle, &onMinimize));
    GLFW_CHECK_ERROR(glfwSetWindowPosCallback(handle, &onMove));
    GLFW_CHECK_ERROR(glfwSetWindowSizeCallback(handle, &onResize));
    GLFW_CHECK_ERROR(glfwSetWindowSizeCallback(handle, &onFramebufferResize));
    GLFW_CHECK_ERROR(glfwSetCursorEnterCallback(handle, &onMouseEnter));
    GLFW_CHECK_ERROR(glfwSetCursorPosCallback(handle, &onMouseMove));
    GLFW_CHECK_ERROR(glfwSetScrollCallback(handle, &onMouseScroll));
    GLFW_CHECK_ERROR(glfwSetMouseButtonCallback(handle, &onMouseButtonInput));
    GLFW_CHECK_ERROR(glfwSetKeyCallback(handle, &onKeyInput));
    GLFW_CHECK_ERROR(glfwSetCharCallback(handle, &onCharInput));

    GLFW_CHECK_ERROR(glfwGetWindowPos(handle, &m_position.x, &m_position.y));
    GLFW_CHECK_ERROR(glfwGetWindowSize(handle, &m_size.x, &m_size.y));
    GLFW_CHECK_ERROR(glfwGetFramebufferSize(handle, &m_framebufferSize.x, &m_framebufferSize.y));

    initializeOpenGL();
    initializeImGui();
}

Window::~Window()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    GLFW_CHECK_ERROR(glfwTerminate());
}

void Window::open()
{
    GLFW_CHECK_ERROR(glfwSetWindowShouldClose(m_handle, GLFW_FALSE));
}

void Window::close()
{
    GLFW_CHECK_ERROR(glfwSetWindowShouldClose(m_handle, GLFW_TRUE));
}

void Window::maximize()
{
    GLFW_CHECK_ERROR(glfwMaximizeWindow(m_handle));
}

void Window::minimize()
{
    GLFW_CHECK_ERROR(glfwIconifyWindow(m_handle));
}

void Window::restore()
{
    GLFW_CHECK_ERROR(glfwRestoreWindow(m_handle));
}

void Window::pollEvents()
{
    m_mouseDelta = glm::vec2{ 0.0f };
    m_mouseScrollDelta = glm::vec2{ 0.0f };
    m_inputText.clear();

    memset(m_actions.data(), -1, m_actions.size() * sizeof(char));

    GLFW_CHECK_ERROR(glfwPollEvents());
}

void Window::swapBuffers()
{
    GLFW_CHECK_ERROR(glfwSwapBuffers(m_handle));
}

void Window::setTitle(const std::string_view title)
{
	m_title = title;
	GLFW_CHECK_ERROR(glfwSetWindowTitle(m_handle, title.data()));
}

void Window::setPosition(const glm::ivec2 position)
{
    GLFW_CHECK_ERROR(glfwSetWindowPos(m_handle, position.x, position.y));
}

void Window::setSize(const glm::ivec2 size)
{
    GLFW_CHECK_ERROR(glfwSetWindowSize(m_handle, size.x, size.y));
}

void Window::setSwapInterval(const int swapInterval)
{
    m_swapInterval = swapInterval;
    GLFW_CHECK_ERROR(glfwSwapInterval(swapInterval));
}

GLFWwindow* Window::getHandle()
{
	return m_handle;
}

const std::string& Window::getTitle() const
{
	return m_title;
}

glm::ivec2 Window::getPosition() const
{
    return m_position;
}

glm::ivec2 Window::getSize() const
{
	return m_size;
}

glm::ivec2 Window::getFramebufferSize() const
{
    return m_framebufferSize;
}

int Window::getSampleCount() const
{
    return m_sampleCount;
}

int Window::getSwapInterval() const
{
    return m_swapInterval;
}

glm::vec2 Window::getMousePosition() const
{
    return m_mousePosition;
}

glm::vec2 Window::getMouseDelta() const
{
    return m_mouseDelta;
}

glm::vec2 Window::getMouseScrollDelta() const
{
    return m_mouseScrollDelta;
}

const std::vector<unsigned int>& Window::getInputText() const
{
    return m_inputText;
}

bool Window::isOpen() const
{
    return glfwWindowShouldClose(m_handle) == GLFW_FALSE;
}

bool Window::isFocused() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_FOCUSED) == GLFW_TRUE;
}

bool Window::isHovered() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_HOVERED) == GLFW_TRUE;
}

bool Window::isMaximized() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_MAXIMIZED) == GLFW_TRUE;
}

bool Window::isMinimized() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_ICONIFIED) == GLFW_TRUE;
}

bool Window::isKeyPressed(const int key) const
{
    return key >= GLFW_MOUSE_BUTTON_1 && key <= GLFW_KEY_LAST && m_actions[static_cast<std::size_t>(key)] == GLFW_PRESS;
}

bool Window::isKeyReleased(const int key) const
{
    return key >= GLFW_MOUSE_BUTTON_1 && key <= GLFW_KEY_LAST && m_actions[static_cast<std::size_t>(key)] == GLFW_RELEASE;
}

bool Window::isKeyDown(const int key) const
{
    return key >= GLFW_MOUSE_BUTTON_1 && key <= GLFW_KEY_LAST && m_keysDown[static_cast<std::size_t>(key)];
}

bool Window::isKeyUp(const int key) const
{
    return !isKeyDown(key);
}

void Window::initializeOpenGL()
{
    GLFW_CHECK_ERROR(glfwMakeContextCurrent(m_handle));
    GLFW_CHECK_ERROR(glfwSwapInterval(m_swapInterval));

    [[maybe_unused]] const int status{ gladLoadGLLoader(reinterpret_cast<GLADloadproc>(&glfwGetProcAddress)) };

    ONEC_ASSERT(status == GL_TRUE, "Failed to initialize OpenGL");
}

void Window::initializeImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    [[maybe_unused]] bool status{ ImGui_ImplGlfw_InitForOpenGL(m_handle, true) };

    ONEC_ASSERT(status, "Failed to initialize ImGui for GLFW");

    status = ImGui_ImplOpenGL3_Init();

    ONEC_ASSERT(status, "Failed to initialize ImGui for OpenGL");
}

void Window::updateMousePosition()
{
    glm::dvec2 mousePosition;
    GLFW_CHECK_ERROR(glfwGetCursorPos(m_handle, &mousePosition.x, &mousePosition.y));
    m_mousePosition = mousePosition;
}

Window& createWindow(const std::string_view title, const glm::ivec2 size, const int sampleCount)
{
    return Window::get(&title, size, sampleCount);
}

Window& getWindow()
{
    return Window::get(nullptr, glm::ivec2{ 0 }, 0);
}

}
