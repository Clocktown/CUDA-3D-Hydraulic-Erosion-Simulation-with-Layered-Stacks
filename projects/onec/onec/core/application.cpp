#include "application.hpp"
#include "window.hpp"
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
}

void Application::run()
{
	ONEC_ASSERT(!isRunning(), "Application must not be running");

	m_running = true;

	Window& window{ getWindow() };
	window.open();

	World& world{ getWorld() };
	world.dispatch<OnStart>();
	
	const double delay{ glfwGetTime() };
	double time{ 0.0 };
	double fixedTime{ 0.0 };
	double unscaledTime{ 0.0 };

	while (window.isOpen())
	{
		const double unscaledDeltaTime{ glfwGetTime() - unscaledTime - delay };
		
		if (window.getSwapInterval() != 0 || m_targetFrameRate <= 0 || unscaledDeltaTime * m_targetFrameRate >= 1.0)
		{
			window.pollEvents();

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			const double deltaTime{ glm::min(m_timeScale * unscaledDeltaTime, m_maxDeltaTime) };
			time += deltaTime;

			unscaledTime += unscaledDeltaTime;

			ONEC_ASSERT(unscaledDeltaTime != 0.0, "Unscaled delta time must not be equal to 0");

			m_frameRate = 1.0 / unscaledDeltaTime;
			m_time = time;
			m_deltaTime = deltaTime;
			m_unscaledTime = unscaledTime;
			m_unscaledDeltaTime = unscaledDeltaTime;

			double fixedDeltaTime{ m_fixedDeltaTime };

			while (fixedTime <= time - fixedDeltaTime)
			{
				fixedTime += fixedDeltaTime;
				m_fixedTime = fixedTime;

				world.dispatch(OnFixedUpdate{ static_cast<float>(fixedDeltaTime) });

				fixedDeltaTime = m_fixedDeltaTime;
			}
			
			world.dispatch(OnUpdate{ static_cast<float>(deltaTime) });

			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			
			window.swapBuffers();

			++m_frameCount;
		}
	}

	m_frameRate = 0.0;
	m_deltaTime = 0.0;
	m_unscaledDeltaTime = 0.0;

	world.dispatch<OnExit>();

	m_frameCount = 0;
	m_time = 0.0;
	m_fixedTime = 0.0;
	m_unscaledTime = 0.0;
	m_running = false;
}

void Application::exit()
{
	Window& window{ getWindow() };
	window.close();
}

void Application::setName(const std::string_view name)
{
	m_name = name;
}

void Application::setDirectory(const std::filesystem::path& directory)
{
	m_directory = directory;
}

void Application::setTargetFrameRate(const int targetFrameRate)
{
	m_targetFrameRate = static_cast<double>(targetFrameRate);
}

void Application::setTimeScale(const double timeScale)
{
	m_timeScale = timeScale;
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

int Application::getTargetFrameRate() const
{
	return static_cast<int>(m_targetFrameRate);
}

double Application::getFrameRate() const
{
	return m_frameRate;
}

double Application::getTimeScale() const
{
	return m_timeScale;
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

double Application::getRealTime() const
{
	return glfwGetTime();
}

bool Application::isRunning() const
{
	return m_running;
}

Application& createApplication(const std::string_view name, const glm::ivec2 size, const int sampleCount)
{
	createWindow(name, size, sampleCount);
	return Application::get(&name);
}

Application& getApplication()
{
	return Application::get(nullptr);
}

}
