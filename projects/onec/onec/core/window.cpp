#include "window.hpp"
#include "input.hpp"
#include "world.hpp"
#include "../config/config.hpp"
#include "../config/glfw.hpp"
#include "../config/gl.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>

namespace onec
{

void Window::onFocus([[maybe_unused]] GLFWwindow* const window, int isFocused)
{
    World& world{ getWorld() };
    world.dispatch(OnWindowFocus{ isFocused == GLFW_TRUE });
}

void Window::onMinimize([[maybe_unused]] GLFWwindow* const window, int isMinimized)
{
    World& world{ getWorld() };
    world.dispatch(OnWindowMinimize{ isMinimized == GLFW_TRUE });
}

void Window::onMaximize([[maybe_unused]] GLFWwindow* const window, int isMaximized)
{
    World& world{ getWorld() };
    world.dispatch(OnWindowMaximize{ isMaximized == GLFW_TRUE });
}

void Window::onDrag([[maybe_unused]] GLFWwindow* const window, const int x, const int y)
{
    glm::ivec2& position{ getWindow().m_position };
    position.x = x;
    position.y = y;
    
    World& world{ getWorld() };
    world.dispatch(OnWindowDrag{ position });
}

void Window::onResize([[maybe_unused]] GLFWwindow* const window, const int width, const int height)
{
    glm::ivec2& size{ getWindow().m_size };
    size.x = width;
    size.y = height;

    World& world{ getWorld() };
    world.dispatch(OnWindowResize{ size });
}

void Window::onFramebufferResize([[maybe_unused]] GLFWwindow* const window, const int width, const int height)
{
    glm::ivec2& framebufferSize{ getWindow().m_framebufferSize };
    framebufferSize.x = width;
    framebufferSize.y = height;

    World& world{ getWorld() };
    world.dispatch(OnFramebufferResize{ framebufferSize });
}

Window& Window::get(const std::string_view* const title, const glm::ivec2& size, const int sampleCount)
{
	static Window window{ title, size, sampleCount };
	return window;
}

Window::Window(const std::string_view* const title, const glm::ivec2& size, const int sampleCount) :
    m_size{ size },
    m_sampleCount{ sampleCount }
{
    ONEC_ASSERT(title != nullptr, "Failed to get window");
    ONEC_ASSERT(sampleCount >= 0, "Sample count must be greater than or equal to 0");

    m_title = *title;
    
    [[maybe_unused]] int status{ glfwInit() };

    ONEC_ASSERT(status == GLFW_TRUE, "Failed to initialize GLFW");

    ONEC_IF_DEBUG(GLFW_CHECK_ERROR(glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE)));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6));
    GLFW_CHECK_ERROR(glfwWindowHint(GLFW_SAMPLES, sampleCount));
    
    m_handle = glfwCreateWindow(size.x, size.y, title->data(), nullptr, nullptr);

    ONEC_ASSERT(m_handle != nullptr, "Failed to create window");

    GLFW_CHECK_ERROR(glfwSetWindowUserPointer(m_handle, this));
    GLFW_CHECK_ERROR(glfwSetWindowFocusCallback(m_handle, &onFocus));
    GLFW_CHECK_ERROR(glfwSetWindowIconifyCallback(m_handle, &onMinimize));
    GLFW_CHECK_ERROR(glfwSetWindowMaximizeCallback(m_handle, &onMaximize));
    GLFW_CHECK_ERROR(glfwSetWindowPosCallback(m_handle, &onDrag));
    GLFW_CHECK_ERROR(glfwSetWindowSizeCallback(m_handle, &onResize));
    GLFW_CHECK_ERROR(glfwSetWindowSizeCallback(m_handle, &onFramebufferResize));

    GLFW_CHECK_ERROR(glfwGetWindowPos(m_handle, &m_position.x, &m_position.y));
    GLFW_CHECK_ERROR(glfwGetWindowSize(m_handle, &m_size.x, &m_size.y));
    GLFW_CHECK_ERROR(glfwGetFramebufferSize(m_handle, &m_framebufferSize.x, &m_framebufferSize.y));

    GLFW_CHECK_ERROR(glfwMakeContextCurrent(m_handle));
    
    status = gladLoadGLLoader(reinterpret_cast<GLADloadproc>(&glfwGetProcAddress));

    ONEC_ASSERT(status == GL_TRUE, "Failed to initialize OpenGL");
}

Window::~Window()
{
    GLFW_CHECK_ERROR(glfwTerminate());
}

void Window::setTitle(const std::string_view& title)
{
	m_title = title;
	GLFW_CHECK_ERROR(glfwSetWindowTitle(m_handle, title.data()));
}

GLFWwindow* Window::getHandle()
{
	return m_handle;
}

const std::string& Window::getTitle() const
{
	return m_title;
}

const glm::ivec2& Window::getSize() const
{
	return m_size;
}

const glm::ivec2& Window::getFramebufferSize() const
{
    return m_framebufferSize;
}

const int Window::getSampleCount() const
{
    return m_sampleCount;
}

bool Window::isFocused() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_FOCUSED) == GLFW_TRUE;
}

bool Window::isHovered() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_HOVERED) == GLFW_TRUE;
}

bool Window::isMinimized() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_ICONIFIED) == GLFW_TRUE;
}

bool Window::isMaximized() const
{
    return glfwGetWindowAttrib(m_handle, GLFW_MAXIMIZED) == GLFW_TRUE;
}

const glm::ivec2& Window::getPosition() const
{
    return m_position;
}

bool Window::shouldClose() const
{
    return glfwWindowShouldClose(m_handle) == GLFW_TRUE;
}

Window& createWindow(const std::string_view& title, const glm::ivec2& size, const int sampleCount)
{
    Window& window{ Window::get(&title, size, sampleCount) };
    getInput();

    return window;
}

Window& getWindow()
{
    return Window::get(nullptr, glm::ivec2{ 0 }, 0);
}

}
