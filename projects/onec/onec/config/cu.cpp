#include "cu.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

namespace onec
{
namespace internal
{

void cuCheckError(const char* const file, const int line)
{
	const cudaError_t error{ cudaGetLastError() };
	
	if (error != cudaSuccess)
	{
		std::cerr << "CUDA Error: " << cudaGetErrorName(error) << "\n"
			      << "Description: " << cudaGetErrorString(error) << "\n"
			      << "File: " << file << "\n"
			      << "Line: " << line << "\n";
		
		std::exit(EXIT_FAILURE);
	}
}

}
}
