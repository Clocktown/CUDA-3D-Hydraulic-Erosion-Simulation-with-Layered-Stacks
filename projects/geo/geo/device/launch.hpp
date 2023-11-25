#pragma once

#include <cuda_runtime.h>

namespace geo
{
namespace device
{

struct Launch
{
	struct Standard1D
	{
		unsigned int gridSize;
		unsigned int blockSize{ 256 };
	};

	struct Standard2D
	{
		dim3 gridSize;
		dim3 blockSize{ 8, 8, 1 };
	};

	struct Standard3D
	{
		dim3 gridSize;
		dim3 blockSize{ 8, 8, 8 };
	};

	struct GridStride1D
	{
		unsigned int gridSize;
		unsigned int blockSize{ 512 };
	};

	struct GridStride2D
	{
		dim3 gridSize;
		dim3 blockSize{ 16, 16, 1 };
	};

	struct GridStride3D
	{
		dim3 gridSize;
		dim3 blockSize{ 8, 8, 8 };
	};

	Standard1D standard1D;
	Standard2D standard2D;
	Standard3D standard3D;
	GridStride1D gridStride1D;
	GridStride2D gridStride2D;
	GridStride3D gridStride3D;
};

}
}
