#pragma once

#include <glm/glm.hpp>
#include <filesystem>
#include <string>

namespace onec
{

class Application
{
public:
	Application(const Application& other) = delete;
	Application(Application&& other) = delete;
	
	~Application();

	Application& operator=(const Application& other) = delete;
	Application& operator=(Application&& other) = delete;

	void run();
	void exit();

	void setName(const std::string_view& name);
	void setDirectory(const std::filesystem::path& directory);
	void setVSyncCount(const int vSyncCount);
	void setTargetFrameRate(const int targetFrameRate);
	void setMaxDeltaTime(const double maxDeltaTime);
	void setFixedDeltaTime(const double fixedDeltaTime);

	const std::string& getName() const;
	const std::filesystem::path& getDirectory() const;
	int getFrameCount() const;
	int getVSyncCount() const;
	int getTargetFrameRate() const;
	double getFrameRate() const;
	double getTime() const;
	double getDeltaTime() const;
	double getMaxDeltaTime() const;
	double getFixedTime() const;
	double getFixedDeltaTime() const;
	double getUnscaledTime() const;
	double getUnscaledDeltaTime() const;
	bool isRunning() const;
	bool isVSyncEnabled() const;
private:
	static Application& get(const std::string_view* const name);

	explicit Application(const std::string_view* const name);

	std::string m_name;
	std::filesystem::path m_directory{ std::filesystem::current_path() };
	int m_frameCount{ 0 };
	int m_vSyncCount{ 1 };
	double m_targetFrameRate{ 60.0 };
	double m_frameRate{ 0.0 };
	double m_timeScale{ 1.0 };
	double m_time{ 0.0 };
	double m_deltaTime{ 0.0f };
	double m_maxDeltaTime{ 1.0 / 3.0 };
	double m_fixedTime{ 0.0 };
	double m_fixedDeltaTime{ 1.0 / 60.0 };
	double m_unscaledTime{ 0.0 };
	double m_unscaledDeltaTime{ 0.0 };
	bool m_isRunning{ false };
	bool m_shouldExit{ false };

	friend Application& createApplication(const std::string_view& name, const glm::ivec2& size, const int sampleCount);
	friend Application& getApplication();
};

struct OnStart
{

};

struct OnUpdate
{

};

struct OnFixedUpdate
{

};

struct OnExit
{

};

Application& createApplication(const std::string_view& name = "Application", const glm::ivec2& size = glm::vec2{ 1280, 720 }, const int sampleCount = 0);
Application& getApplication();

}
