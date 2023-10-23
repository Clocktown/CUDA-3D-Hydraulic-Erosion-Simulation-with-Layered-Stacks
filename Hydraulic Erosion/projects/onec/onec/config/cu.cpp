#include "cu.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace onec
{
namespace internal
{

void cuCheckError(const cudaError_t error, const char* const file, const int line)
{
	if (error != cudaSuccess)
	{
		std::cerr << "CUDA Error: " << cudaGetErrorString(error) << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";
		
		std::exit(EXIT_FAILURE);
	}
}

}
}
