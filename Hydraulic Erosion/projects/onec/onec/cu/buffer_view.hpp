#pragma once

#include "buffer.hpp"
#include "../utility/span.hpp"
#include <cuda_runtime.h>

namespace onec
{
namespace cu
{

class GraphicsResource;

class BufferView
{
public:
	BufferView();
	BufferView(Buffer& buffer);
	BufferView(Buffer& buffer, const int count);

	BufferView& operator=(Buffer& buffer);

	void upload(const Span<const std::byte>&& data);
	void upload(const Span<const std::byte>&& data, const int count);
	void upload(const Span<const std::byte>&& data, const int offset, const int count);
	void download(const Span<std::byte>&& data) const;
	void download(const Span<std::byte>&& data, const int count) const;
	void download(const Span<std::byte>&& data, const int offset, const int count) const;

	const std::byte* getData() const;
	std::byte* getData();
	int getCount() const;
	bool isEmpty() const;
private:
	BufferView(std::byte* const data, const int count);

	std::byte* m_data;
	int m_count;

	friend class GraphicsResource;
};

}
}
