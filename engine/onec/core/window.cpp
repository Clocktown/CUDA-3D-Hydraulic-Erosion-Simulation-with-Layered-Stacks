#include "window.hpp"
#include "../config/config.hpp"
#include "../config/glfw.hpp"
#include "../config/gl.hpp"
#include "../config/cu.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <string>
#include <array>

namespace onec
{

void Window::onMove([[maybe_unused]] GLFWwindow* const, const int x, const int y)
{
    Window& window{ getWindow() };
    window.m_position = glm::ivec2{ x, y };
}

void Window::onResize([[maybe_unused]] GLFWwindow* const, const int width, const int height)
{
    Window& window{ getWindow() };
    window.m_size = glm::ivec2{ width, height };
}

void Window::onFramebufferResize([[maybe_unused]] GLFWwindow* const, const int width, const int height)
{
    Window& window{ getWindow() };
    window.m_framebufferSize = glm::ivec2{ width, height };
}

void Window::onMouseEnter([[maybe_unused]] GLFWwindow* const, const int hasMouseEntered)
{
    if (hasMouseEntered == GLFW_TRUE)
    {
        Window& window{ getWindow() };
        window.updateMousePosition();
    }
}

void Window::onMouseMove([[maybe_unused]] GLFWwindow* const, const double x, const double y)
{
    Window& window{ getWindow() };
    const glm::vec2 mousePosition{ x, y };

    if (!ImGui::GetIO().WantCaptureMouse)
    {
        window.m_mouseDelta += mousePosition - window.m_mousePosition;
    }

    window.m_mousePosition = mousePosition;
}

void Window::onMouseScroll([[maybe_unused]] GLFWwindow* const, const double x, const double y)
{
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        Window& window{ getWindow() };
        window.m_mouseScrollDelta += glm::vec2{ x, y };
    }
}

void Window::onMouseButtonInput([[maybe_unused]] GLFWwindow* const window, const int mouseButton, const int action, [[maybe_unused]] const int modifier)
{
    if (!ImGui::GetIO().WantCaptureMouse && mouseButton != GLFW_KEY_UNKNOWN)
    {
        Window& window{ getWindow() };

        const std::size_t index{ static_cast<std::size_t>(mouseButton) };
        window.m_actions[index] = static_cast<char>(action);
        window.m_keysDown[index] = action == GLFW_PRESS;
    }
}

void Window::onKeyInput([[maybe_unused]] GLFWwindow* const, const int key, [[maybe_unused]] const int scancode, const int action, [[maybe_unused]] const int modifier)
{
    if (!ImGui::GetIO().WantCaptureKeyboard && key != GLFW_KEY_UNKNOWN && action != GLFW_REPEAT)
    {
        Window& window{ getWindow() };

        const std::size_t index{ static_cast<std::size_t>(key) };
        window.m_actions[index] = static_cast<char>(action);
        window.m_keysDown[index] = action == GLFW_PRESS;
    }
}

void Window::onCharInput([[maybe_unused]] GLFWwindow* const, const unsigned int unicode)
{
    if (!ImGui::GetIO().WantTextInput)
    {
        Window& window{ getWindow() };
        window.m_text.push_back(static_cast<char32_t>(unicode));
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
    ONEC_IF_RELEASE(GLFW_CHECK_ERROR(glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_TRUE));)
 
    GLFWwindow* const handle{ glfwCreateWindow(size.x, size.y, title->data(), nullptr, nullptr) };

    ONEC_ASSERT(handle != nullptr, "Failed to create window");

    m_handle = handle;
    m_title = *title;
    m_text.reserve(32);

    updateMousePosition();
    memset(m_actions.data(), -1, m_actions.size() * sizeof(char));
    memset(m_keysDown.data(), 0, m_keysDown.size() * sizeof(bool));

    GLFW_CHECK_ERROR(glfwSetWindowUserPointer(handle, this));
    GLFW_CHECK_ERROR(glfwSetWindowPosCallback(handle, &onMove));
    GLFW_CHECK_ERROR(glfwSetWindowSizeCallback(handle, &onResize));
    GLFW_CHECK_ERROR(glfwSetWindowSizeCallback(handle, &onFramebufferResize));
    GLFW_CHECK_ERROR(glfwSetCursorEnterCallback(handle, &onMouseEnter));
    GLFW_CHECK_ERROR(glfwSetCursorPosCallback(handle, &onMouseMove));
    GLFW_CHECK_ERROR(glfwSetScrollCallback(handle, &onMouseScroll));
    GLFW_CHECK_ERROR(glfwSetMouseButtonCallback(handle, &onMouseButtonInput));
    GLFW_CHECK_ERROR(glfwSetKeyCallback(handle, &onKeyInput));
    GLFW_CHECK_ERROR(glfwSetCharCallback(handle, &onCharInput));
    GLFW_CHECK_ERROR(glfwSetInputMode(handle, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE));
    
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
    m_text.clear();

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

void Window::setCursorLocked(bool cursorLocked)
{
    GLFW_CHECK_ERROR(glfwSetInputMode(m_handle, GLFW_CURSOR, cursorLocked ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL));
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

const std::u32string& Window::getText() const
{
    return m_text;
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

bool Window::isCursorLocked() const
{
    return glfwGetInputMode(m_handle, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
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

void Window::initializeCUDA()
{
    int device;
    CU_CHECK_ERROR(cudaGLGetDevices(nullptr, &device, 1, cudaGLDeviceListAll));
    CU_CHECK_ERROR(cudaSetDevice(device));
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
