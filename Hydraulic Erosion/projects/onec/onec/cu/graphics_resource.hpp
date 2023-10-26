#pragma once

#include "buffer_view.hpp"
#include "array_view.hpp"
#include "../graphics/buffer.hpp"
#include "../graphics/texture.hpp"
#include "../graphics/renderbuffer.hpp"
#include <cuda_runtime.h>

namespace onec
{
namespace cu
{

class GraphicsResource
{
	explicit GraphicsResource();
	explicit GraphicsResource(onec::Buffer& buffer, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	explicit GraphicsResource(onec::Texture& texture, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	explicit GraphicsResource(onec::Renderbuffer& renderbuffer, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	GraphicsResource(const GraphicsResource& other) = delete;
	GraphicsResource(GraphicsResource&& other) noexcept;

	~GraphicsResource();

	GraphicsResource& operator=(const GraphicsResource& other) = delete;
	GraphicsResource& operator=(GraphicsResource&& other) noexcept;

	void initialize(onec::Buffer& buffer, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	void initialize(onec::Texture& texture, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	void initialize(onec::Renderbuffer& renderbuffer, const unsigned int flags = cudaGraphicsRegisterFlagsNone);
	void release();
	void map(cudaStream_t stream = cudaStream_t{});
	void unmap();

	cudaGraphicsResource_t getHandle();
	BufferView getBufferView() const;
	ArrayView getArrayView(const int layer = 0, const int mipLevel = 0) const;
	bool isEmpty() const;
private:
	cudaGraphicsResource_t m_handle;
	cudaStream_t m_stream;
};

}
}
