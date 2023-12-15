#include "time.hpp"
#include "../config/config.hpp"
#include <chrono>

namespace onec
{

template<typename Period>
inline void Timer<Period>::start()
{
	ONEC_ASSERT(m_state == State::Stopped, "Timer must be stopped");

	m_state = State::Running;
	m_time = 0.0;
	m_start = std::chrono::steady_clock::now();
}

template<typename Period>
inline void Timer<Period>::resume()
{
	ONEC_ASSERT(m_state == State::Paused, "Timer must be paused");

	m_state = State::Running;
	m_start = std::chrono::steady_clock::now();
}

template<typename Period>
inline void Timer<Period>::pause()
{
	ONEC_ASSERT(m_state == State::Running, "Timer must be running");

	const std::chrono::duration<double, Period> duration{ std::chrono::steady_clock::now() - m_start };

	m_state = State::Paused;
	m_time += duration.count();
}

template<typename Period>
inline void Timer<Period>::stop()
{
	ONEC_ASSERT(m_state != State::Stopped, "Timer must not be stopped");

	if (m_state == State::Running)
	{
		const std::chrono::duration<double, Period> duration{ std::chrono::steady_clock::now() - m_start };
		m_time += duration.count();
	}

	m_state = State::Stopped;
}

template<typename Period>
inline double Timer<Period>::getTime() const
{
	if (m_state == State::Running)
	{
		const std::chrono::duration<double, Period> duration{ std::chrono::steady_clock::now() - m_start };
		return m_time + duration.count();
	}

	return m_time;
}

template<typename Period>
inline bool Timer<Period>::isStopped() const
{
	return m_state == State::Stopped;
}

template<typename Period>
inline bool Timer<Period>::isPaused() const
{
	return m_state == State::Paused;
}

template<typename Period>
inline bool Timer<Period>::isRunning() const
{
	return m_state == State::Running;
}

}
