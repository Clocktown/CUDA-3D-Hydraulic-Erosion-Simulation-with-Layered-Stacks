#include "stopwatch.hpp"
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
inline Stopwatch<Period>::Stopwatch() :
	m_state{ State::Stopped },
	m_time{ 0.0 },
	m_stream{ cudaStream_t{} },
	m_hasChanged{ false }
{
	CU_CHECK_ERROR(cudaEventCreate(&m_start));
	CU_CHECK_ERROR(cudaEventCreate(&m_end));
}

template<typename Period>
inline Stopwatch<Period>::Stopwatch(Stopwatch&& other) noexcept :
	m_state{ std::exchange(other.m_state, State::Stopped) },
	m_time{ std::exchange(other.m_time, 0.0) },
	m_stream{ std::exchange(other.m_stream, cudaStream_t{}) },
	m_start{ std::exchange(other.m_start, cudaEvent_t{}) },
	m_end{ std::exchange(other.m_end, cudaEvent_t{}) },
	m_hasChanged{ std::exchange(other.m_hasChanged, false) }
{

}

template<typename Period>
inline Stopwatch<Period>::~Stopwatch()
{
	CU_CHECK_ERROR(cudaEventDestroy(m_start));
	CU_CHECK_ERROR(cudaEventDestroy(m_end));
}

template<typename Period>
inline Stopwatch<Period>& Stopwatch<Period>::operator=(Stopwatch<Period>&& other) noexcept
{
	if (this != &other)
	{
		CU_CHECK_ERROR(cudaEventDestroy(m_start));
		CU_CHECK_ERROR(cudaEventDestroy(m_end));

		m_state = std::exchange(other.m_state, State::Stopped);
		m_time = std::exchange(other.m_time, 0.0);
		m_stream = std::exchange(other.m_stream, cudaStream_t{});
		m_start = std::exchange(other.m_start, cudaEvent_t{});
		m_end = std::exchange(other.m_end, cudaEvent_t{});
		m_hasChanged = std::exchange(other.m_hasChanged, false);
	}

	return *this;
}

template<typename Period>
inline void Stopwatch<Period>::start(const cudaStream_t stream)
{
	ONEC_ASSERT(m_state == State::Stopped, "Stopwatch must be stopped");

	m_state = State::Running;
	m_time = 0.0;
	m_stream = stream;
	m_hasChanged = false;
	CU_CHECK_ERROR(cudaEventRecord(m_start, m_stream));
}

template<typename Period>
inline void Stopwatch<Period>::resume()
{
	ONEC_ASSERT(m_state == State::Paused, "Stopwatch must be paused");

	update();

	m_state = State::Running;
	CU_CHECK_ERROR(cudaEventRecord(m_start, m_stream));
}

template<typename Period>
inline void Stopwatch<Period>::pause()
{
	ONEC_ASSERT(m_state == State::Running, "Stopwatch must be running");

	CU_CHECK_ERROR(cudaEventRecord(m_end, m_stream));
	m_state = State::Paused;
	m_hasChanged = true;
}

template<typename Period>
inline void Stopwatch<Period>::stop()
{
	ONEC_ASSERT(m_state != State::Stopped, "Stopwatch must be running or be paused");

	if (m_state == State::Running)
	{
		CU_CHECK_ERROR(cudaEventRecord(m_end, m_stream));
		m_hasChanged = true;
	}

	m_state = State::Stopped;
}

template<typename Period>
inline void Stopwatch<Period>::update() const
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

template<typename Period>
inline double Stopwatch<Period>::getTime() const
{
	if (m_state == State::Running)
	{
		float duration;
		CU_CHECK_ERROR(cudaEventRecord(m_end, m_stream));
		CU_CHECK_ERROR(cudaEventSynchronize(m_end));
		CU_CHECK_ERROR(cudaEventElapsedTime(&duration, m_start, m_end));

		return m_time + duration * static_cast<double>(Period::den) / (1000.0 * static_cast<double>(Period::num));
	}
	else
	{
		update();
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
}
