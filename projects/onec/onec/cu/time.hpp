#pragma once

#include <cuda_runtime.h>
#include <chrono>

namespace onec
{
namespace cu
{

template<typename Period = std::ratio<1>>
class Timer
{
public:
	Timer();
	Timer(const Timer<Period>& other) = delete;
	Timer(Timer<Period>&& other) noexcept;

	~Timer();

	Timer<Period>& operator=(const Timer<Period>& other) = delete;
	Timer<Period>& operator=(Timer<Period>&& other) noexcept;
	
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

	void synchronize() const;

	State m_state;
	mutable double m_time;
	cudaEvent_t m_start;
	cudaEvent_t m_end;
	mutable bool m_hasChanged;
};

}
}

#include "time.inl"
