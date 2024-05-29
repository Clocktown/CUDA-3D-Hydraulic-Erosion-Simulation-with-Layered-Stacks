#pragma once
#include <string>
#include <map>


namespace geo {
	struct measurement {
		float mean{ 0.f };
		int count{ 0 };
		void reset() { mean = 0.f; count = 0.f; }
		void update(float duration) { mean = (duration + mean * count) / (count + 1); count++; }
	};

	struct Performance
	{
		Performance();
		~Performance();
		bool measurePerformance{ false };
		bool measureRendering{ false };
		bool measureParts{ false };
		bool measureIndividualKernels{ false };
		bool pauseAfterStepCount{ 0 };
		cudaEvent_t globalStart, globalStop;
		cudaEvent_t localStart, localStop;
		cudaEvent_t kernelStart, kernelStop;
		std::map<std::string, measurement> measurements{
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

		void measure(const std::string& name, cudaEvent_t& start, cudaEvent_t& stop);
		void reset();
	};
};