#pragma once

#include <glm/glm.hpp>
#include <filesystem>
#include <string>

namespace onec
{

struct OnStart
{

};

struct OnUpdate
{

};

struct OnExit
{

};

class Application
{
public:
	Application(const Application& other) = delete;
	Application(Application&& other) = delete;

	~Application() = default;

	Application& operator=(const Application& other) = delete;
	Application& operator=(Application&& other) = delete;

	void run();
	void exit();

	void setName(std::string_view name);
	void setDirectory(const std::filesystem::path& directory);
	void setTargetFrameRate(int targetFrameRate);
	void setTimeScale(double timeScale);
	void setMaxDeltaTime(double maxDeltaTime);

	const std::string& getName() const;
	const std::filesystem::path& getDirectory() const;
	int getFrameCount() const;
	int getTargetFrameRate() const;
	double getFrameRate() const;
	double getTimeScale() const;
	double getTime() const;
	double getDeltaTime() const;
	double getMaxDeltaTime() const;
	double getUnscaledTime() const;
	double getUnscaledDeltaTime() const;
	double getRealTime() const;
	bool isRunning() const;
private:
	static Application& get(const std::string_view* name);

	explicit Application(const std::string_view* name);

	std::string m_name;
	std::filesystem::path m_directory{ std::filesystem::current_path() };
	int m_frameCount{ 0 };
	double m_targetFrameRate{ 60.0 };
	double m_frameRate{ 0.0 };
	double m_timeScale{ 1.0 };
	double m_time{ 0.0 };
	double m_deltaTime{ 0.0f };
	double m_maxDeltaTime{ 1.0 / 3.0 };
	double m_unscaledTime{ 0.0 };
	double m_unscaledDeltaTime{ 0.0 };
	bool m_running{ false };

	friend Application& createApplication(std::string_view name, glm::ivec2 size, int sampleCount);
	friend Application& getApplication();
};

Application& createApplication(std::string_view name = "Application", glm::ivec2 size = glm::vec2{ 1280, 720 }, int sampleCount = 0);
Application& getApplication();

}
