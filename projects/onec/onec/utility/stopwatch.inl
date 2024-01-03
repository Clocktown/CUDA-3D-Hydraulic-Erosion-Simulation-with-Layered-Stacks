#include "stopwatch.hpp"
#include "../config/config.hpp"
#include <chrono>

namespace onec
{

template<typename Period>
inline void Stopwatch<Period>::start()
{
	ONEC_ASSERT(m_state == State::Stopped, "Stopwatch must be stopped");

	m_state = State::Running;
	m_time = 0.0;
	m_start = std::chrono::steady_clock::now();
}

template<typename Period>
inline void Stopwatch<Period>::resume()
{
	ONEC_ASSERT(m_state == State::Paused, "Stopwatch must be paused");

	m_state = State::Running;
	m_start = std::chrono::steady_clock::now();
}

template<typename Period>
inline void Stopwatch<Period>::pause()
{
	ONEC_ASSERT(m_state == State::Running, "Stopwatch must be running");

	const std::chrono::duration<double, Period> duration{ std::chrono::steady_clock::now() - m_start };

	m_state = State::Paused;
	m_time += duration.count();
}

template<typename Period>
inline void Stopwatch<Period>::stop()
{
	ONEC_ASSERT(m_state != State::Stopped, "Stopwatch must not be stopped");

	if (m_state == State::Running)
	{
		const std::chrono::duration<double, Period> duration{ std::chrono::steady_clock::now() - m_start };
		m_time += duration.count();
	}

	m_state = State::Stopped;
}

template<typename Period>
inline double Stopwatch<Period>::getTime() const
{
	if (m_state == State::Running)
	{
		const std::chrono::duration<double, Period> duration{ std::chrono::steady_clock::now() - m_start };
		return m_time + duration.count();
	}

	return m_time;
}

template<typename Period>
inline bool Stopwatch<Period>::isStopped() const
{
	return m_state == State::Stopped;
}

template<typename Period>
inline bool Stopwatch<Period>::isPaused() const
{
	return m_state == State::Paused;
}

template<typename Period>
inline bool Stopwatch<Period>::isRunning() const
{
	return m_state == State::Running;
}

}
