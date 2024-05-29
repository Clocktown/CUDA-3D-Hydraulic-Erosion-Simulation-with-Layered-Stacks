#pragma once
#include <string>
#include <map>

#include <utility>


namespace geo {
	struct measurement {
		measurement();
		~measurement();

		measurement(const measurement& o) {
			mean = o.mean;
			count = o.count;
			cudaEventCreate(&startE);
			cudaEventCreate(&stopE);
			wasMeasured = o.wasMeasured;
		}

		float mean{ 0.f };
		int count{ 0 };
		cudaEvent_t startE = nullptr, stopE = nullptr;
		bool wasMeasured{ false };
		void reset() { mean = 0.f; count = 0.f; wasMeasured = false; }
		void update(float duration) { mean = (duration + mean * count) / (count + 1); count++; }
		void measure();
		void start();
		void stop();
	};

	struct performance
	{
		bool measurePerformance{ false };
		bool measureRendering{ false };
		bool measureParts{ false };
		bool measureIndividualKernels{ false };
		bool pauseAfterStepCount{ 0 };

		std::map<std::string, measurement> measurements {
			{"Global Simulation", {}},
			{"Rendering", {}},
			{"Build Draw List", {}},
			{"Rain", {}},
			{"Transport", {}},
			{"Erosion", {}},
			{"Support", {}},
			{"Setup Pipes", {}},
			{"Resolve Pipes", {}},
			{"Horizontal Erosion", {}},
			{"Split Kernel", {}},
			{"Vertical Erosion", {}},
			{"Start Support Check", {}},
			{"Step Support Check", {}},
			{"End Support Check", {}}
		};

		void measureAll();
		void resetAll();
	};
};