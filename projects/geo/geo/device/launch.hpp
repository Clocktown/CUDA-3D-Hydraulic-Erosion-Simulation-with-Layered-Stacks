#pragma once

#include <cuda_runtime.h>

namespace geo
{
namespace device
{

struct Launch
{
	dim3 gridSize;
	dim3 blockSize{ 8, 8, 1 };
};

}
}
