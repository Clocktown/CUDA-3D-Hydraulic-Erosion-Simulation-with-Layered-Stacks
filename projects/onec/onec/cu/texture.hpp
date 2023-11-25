 #pragma once

#include "array.hpp"
#include "array_view.hpp"
#include <cuda_runtime.h>

namespace onec
{
namespace cu
{

class Texture
{
public:
	explicit Texture();
	explicit Texture(const Array& array, const cudaTextureDesc& desc = cudaTextureDesc{});
	explicit Texture(const ArrayView& arrayView, const cudaTextureDesc& desc = cudaTextureDesc{});
	Texture(const Texture& other) = delete;
	Texture(Texture&& other) noexcept;

	~Texture();

	Texture& operator=(const Texture& other) = delete;
	Texture& operator=(Texture&& other) noexcept;

	void initialize(const Array& array, const cudaTextureDesc& desc = cudaTextureDesc{});
	void initialize(const ArrayView& arrayView, const cudaTextureDesc& desc = cudaTextureDesc{});
	void release();

	cudaTextureObject_t getHandle();
	bool isEmpty() const;
private:
	cudaTextureObject_t m_handle;
};

}
}
