#include "simulation.hpp"
#include <cuda_fp16.h>

namespace geo{
namespace device{

__forceinline__ __device__  half4 toHalf4(const float4& v) {
	auto v2 = reinterpret_cast<const float2*>(&v);
	half4 res;
	res.a = __float22half2_rn(v2[0]);
	res.b = __float22half2_rn(v2[1]);
	return res;
}

__forceinline__ __device__  half4 toHalf4(const glm::vec4& v) {
	return toHalf4(glm::cuda_cast(v));
}

__forceinline__ __device__  float4 half4toFloat4(const half4& v) {
	float4 res;
	auto res2 = reinterpret_cast<float2*>(&res);
	res2[0] = __half22float2(v.a);
	res2[1] = __half22float2(v.b);
	return res;
}

__forceinline__ __device__  glm::vec4 half4toVec4(const half4& v) {
	return glm::cuda_cast(half4toFloat4(v));
}

}
}