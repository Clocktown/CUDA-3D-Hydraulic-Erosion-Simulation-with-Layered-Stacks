#pragma once

#include <chrono>

namespace onec
{

template<typename Period = std::ratio<1>>
class Stopwatch
{
public:
	void start();
	void resume();
	void pause();
	void stop();

	double getTime() const;
	bool isStopped() const;
	bool isPaused() const;
	bool isRunning() const;
private:
	enum class State : unsigned char
	{
		Stopped,
		Paused,
		Running
	};

	State m_state{ State::Stopped };
	double m_time{ 0.0 };
	std::chrono::steady_clock::time_point m_start;
};

}

#include "stopwatch.inl"
