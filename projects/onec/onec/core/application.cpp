#include "application.hpp"
#include "window.hpp"
#include "input.hpp"
#include "world.hpp"
#include "../config/config.hpp"
#include "../config/glfw.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include <filesystem>
#include <string>

namespace onec
{

Application& Application::get(const std::string_view* const name)
{
	static Application application{ name };
	return application;
}

Application::Application(const std::string_view* const name)
{
	ONEC_ASSERT(name != nullptr, "Failed to get application");
	
	m_name = *name;

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	[[maybe_unused]] bool status{ ImGui_ImplGlfw_InitForOpenGL(getWindow().getHandle(), true) };

	ONEC_ASSERT(status, "Failed to initialize ImGui for GLFW");

	status = ImGui_ImplOpenGL3_Init("#version 460");

	ONEC_ASSERT(status, "Failed to initialize ImGui for OpenGL");
}

Application::~Application()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Application::run()
{
	ONEC_ASSERT(!m_isRunning, "Application is already running");

	m_isRunning = true;
	m_shouldExit = false;
	
	Window& window{ getWindow() };
	Input& input{ getInput() };
	World& world{ getWorld() };

	GLFW_CHECK_ERROR(glfwSetTime(0.0));

	world.dispatch<OnStart>();
	
	const double delay{ glfwGetTime() };

	while (!m_shouldExit && !window.shouldClose())
	{
		const double unscaledDeltaTime{ glfwGetTime() - delay - m_unscaledTime };
		
		if (isVSyncEnabled() || m_targetFrameRate <= 0 || unscaledDeltaTime * m_targetFrameRate >= 1.0)
		{
			ONEC_ASSERT(unscaledDeltaTime != 0.0, "Unscaled delta time cannot be equal to 0");

			m_frameRate = 1.0 / unscaledDeltaTime;
			m_deltaTime = glm::min(m_timeScale * unscaledDeltaTime, m_maxDeltaTime);
			m_time += m_deltaTime;
			m_unscaledTime += unscaledDeltaTime;
			m_unscaledDeltaTime = unscaledDeltaTime;

			input.poll();

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			const double maxFixedTime{ m_time - m_fixedDeltaTime };

			while (m_fixedTime <= maxFixedTime)
			{
				m_fixedTime += m_fixedDeltaTime;
				world.dispatch<OnFixedUpdate>();
			}
			
			world.dispatch<OnUpdate>();

			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			
			GLFW_CHECK_ERROR(glfwSwapBuffers(window.getHandle()));

			++m_frameCount;
		}
	}

	GLFW_CHECK_ERROR(glfwSetWindowShouldClose(window.getHandle(), GLFW_TRUE));

	world.dispatch<OnExit>();

	m_frameCount = 0;
	m_frameRate = 0.0;
	m_time = 0.0;
	m_deltaTime = 0.0;
	m_fixedTime = 0.0;
	m_unscaledTime = 0.0;
	m_unscaledDeltaTime = 0.0;
	m_isRunning = false;
}

void Application::exit()
{
	m_shouldExit = true;
}

void Application::setName(const std::string_view& name)
{
	m_name = name;
}

void Application::setDirectory(const std::filesystem::path& directory)
{
	m_directory = directory;
}

void Application::setVSyncCount(const int vSyncCount)
{
	m_vSyncCount = vSyncCount;
	GLFW_CHECK_ERROR(glfwSwapInterval(m_vSyncCount));
}

void Application::setTargetFrameRate(const int targetFrameRate)
{
	m_targetFrameRate = static_cast<double>(targetFrameRate);
}

void Application::setMaxDeltaTime(const double maxDeltaTime)
{
	m_maxDeltaTime = maxDeltaTime;
}

void Application::setFixedDeltaTime(const double fixedDeltaTime)
{
	m_fixedDeltaTime = fixedDeltaTime;
}

const std::string& Application::getName() const
{
	return m_name;
}

const std::filesystem::path& Application::getDirectory() const
{
	return m_directory;
}

int Application::getFrameCount() const
{
	return m_frameCount;
}

int Application::getVSyncCount() const
{
	return m_vSyncCount;
}

int Application::getTargetFrameRate() const
{
	return static_cast<int>(m_targetFrameRate);
}

double Application::getFrameRate() const
{
	return m_frameRate;
}

double Application::getTime() const
{
	return m_time;
}

double Application::getDeltaTime() const
{
	return m_deltaTime;
}

double Application::getMaxDeltaTime() const
{
	return m_maxDeltaTime;
}

double Application::getFixedTime() const
{
	return m_fixedTime;
}

double Application::getFixedDeltaTime() const
{
	return m_fixedDeltaTime;
}

double Application::getUnscaledTime() const
{
	return m_unscaledTime;
}

double Application::getUnscaledDeltaTime() const
{
	return m_unscaledDeltaTime;
}

bool Application::isRunning() const
{
	return m_isRunning;
}

bool Application::isVSyncEnabled() const
{
	return m_vSyncCount != 0;
}

Application& createApplication(const std::string_view& name, const glm::ivec2& size, const int sampleCount)
{
	createWindow(name, size, sampleCount);
	return Application::get(&name);
}

Application& getApplication()
{
	return Application::get(nullptr);
}

}
