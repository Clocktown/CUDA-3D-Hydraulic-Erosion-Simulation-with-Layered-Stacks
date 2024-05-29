#include <onec/onec.hpp>
#include "performance.hpp"

namespace geo {
	Performance::Performance() {
		cudaEventCreate(&globalStart);
		cudaEventCreate(&globalStop);
		cudaEventCreate(&localStart);
		cudaEventCreate(&localStop);
		cudaEventCreate(&kernelStart);
		cudaEventCreate(&kernelStop);
	}

	Performance::~Performance() {
		cudaEventDestroy(globalStart);
		cudaEventDestroy(globalStop);
		cudaEventDestroy(localStart);
		cudaEventDestroy(localStop);
		cudaEventDestroy(kernelStart);
		cudaEventDestroy(kernelStop);
	}

	void Performance::measure(const std::string& name, cudaEvent_t& start, cudaEvent_t& stop) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		measurements[name].update(milliseconds);
	}

	void Performance::reset() {
		for (auto& measurement : measurements) {
			measurement.second = {};
		}
	}
}