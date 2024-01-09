#pragma once

#include <cuda_runtime.h>
#include <chrono>

namespace onec
{
namespace cu
{

template<typename Period = std::ratio<1>>
class Stopwatch
{
public:
	Stopwatch();
	Stopwatch(const Stopwatch<Period>& other) = delete;
	Stopwatch(Stopwatch<Period>&& other) noexcept;

	~Stopwatch();

	Stopwatch<Period>& operator=(const Stopwatch<Period>& other) = delete;
	Stopwatch<Period>& operator=(Stopwatch<Period>&& other) noexcept;
	
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

#include "stopwatch.inl"
