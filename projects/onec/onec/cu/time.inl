#include "time.hpp"
#include "../config/config.hpp"
#include "../config/cu.hpp"
#include <cuda_runtime.h>
#include <utility>
#include <chrono>

namespace onec
{
namespace cu 
{

template<typename Period>
inline Timer<Period>::Timer() :
	m_state{ State::Stopped },
	m_time{ 0.0 },
	m_hasChanged{ false }
{
	CU_CHECK_ERROR(cudaEventCreate(&m_start));
	CU_CHECK_ERROR(cudaEventCreate(&m_end));
}

template<typename Period>
inline Timer<Period>::Timer(Timer&& other) noexcept :
	m_state{ std::exchange(other.m_state, State::Stopped) },
	m_time{ std::exchange(other.m_time, 0.0) },
	m_start{ std::exchange(other.m_start, cudaEvent_t{}) },
	m_end{ std::exchange(other.m_end, cudaEvent_t{}) },
	m_hasChanged{ std::exchange(other.m_hasChanged, false) }
{

}

template<typename Period>
inline Timer<Period>::~Timer()
{
	CU_CHECK_ERROR(cudaEventDestroy(m_start));
	CU_CHECK_ERROR(cudaEventDestroy(m_end));
}

template<typename Period>
inline Timer<Period>& Timer<Period>::operator=(Timer<Period>&& other) noexcept
{
	if (this != &other)
	{
		CU_CHECK_ERROR(cudaEventDestroy(m_start));
		CU_CHECK_ERROR(cudaEventDestroy(m_end));

		m_state = std::exchange(other.m_state, State::Stopped);
		m_time = std::exchange(other.m_time, 0.0);
		m_start = std::exchange(other.m_start, cudaEvent_t{});
		m_end = std::exchange(other.m_end, cudaEvent_t{});
		m_hasChanged = std::exchange(other.m_hasChanged, false);
	}

	return *this;
}

template<typename Period>
inline void Timer<Period>::start()
{
	ONEC_ASSERT(m_state == State::Stopped, "Timer must be stopped");

	m_state = State::Running;
	m_time = 0.0;
	m_hasChanged = false;
	CU_CHECK_ERROR(cudaEventRecord(m_start));
}

template<typename Period>
inline void Timer<Period>::resume()
{
	ONEC_ASSERT(m_state == State::Paused, "Timer must be paused");

	synchronize();

	m_state = State::Running;
	CU_CHECK_ERROR(cudaEventRecord(m_start));
}

template<typename Period>
inline void Timer<Period>::pause()
{
	ONEC_ASSERT(m_state == State::Running, "Timer must be running");

	CU_CHECK_ERROR(cudaEventRecord(m_end));
	m_state = State::Paused;
	m_hasChanged = true;
}

template<typename Period>
inline void Timer<Period>::stop()
{
	ONEC_ASSERT(m_state != State::Stopped, "Timer must be running or be paused");

	if (m_state == State::Running)
	{
		CU_CHECK_ERROR(cudaEventRecord(m_end));
		m_hasChanged = true;
	}

	m_state = State::Stopped;
}

template<typename Period>
inline double Timer<Period>::getTime() const
{
	if (m_state == State::Running)
	{
		float duration;
		CU_CHECK_ERROR(cudaEventRecord(m_end));
		CU_CHECK_ERROR(cudaEventSynchronize(m_end));
		CU_CHECK_ERROR(cudaEventElapsedTime(&duration, m_start, m_end));

		return m_time + duration * static_cast<double>(Period::den) / (1000.0 * static_cast<double>(Period::num));
	}
	else
	{
		synchronize();
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

template<typename Period>
inline void Timer<Period>::synchronize() const
{
	if (m_hasChanged)
	{
		float duration;
		CU_CHECK_ERROR(cudaEventSynchronize(m_end));
		CU_CHECK_ERROR(cudaEventElapsedTime(&duration, m_start, m_end));

		m_time += duration * static_cast<double>(Period::den) / (1000.0 * static_cast<double>(Period::num));
		m_hasChanged = false;
	}
}

}
}
