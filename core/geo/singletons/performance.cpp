#include <onec/onec.hpp>
#include "performance.hpp"

namespace geo {
	measurement::measurement() {
		CU_CHECK_ERROR(cudaEventCreate(&startE));
		CU_CHECK_ERROR(cudaEventCreate(&stopE));
	}

	measurement::~measurement() {
		CU_CHECK_ERROR(cudaEventDestroy(startE));
		CU_CHECK_ERROR(cudaEventDestroy(stopE));
	}

	void measurement::start() {
		cudaEventRecord(startE);
		wasMeasured = true;
	}

	void measurement::stop() {
		cudaEventRecord(stopE);
		wasMeasured = true;
	}

	void measurement::measure() {
		if (wasMeasured) {
			cudaEventSynchronize(stopE);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, startE, stopE);
			update(milliseconds);
			wasMeasured = false;
		}
	}

	void performance::resetAll() {
		for (auto& measurement : measurements) {
			measurement.second.reset();
		}
	}

	void performance::measureAll() {
		for (auto& measurement : measurements) {
			measurement.second.measure();
		}
	}
}