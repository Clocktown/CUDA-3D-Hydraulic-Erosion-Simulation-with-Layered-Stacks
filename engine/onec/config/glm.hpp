#pragma once

#include "cu.hpp"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace glm
{

CU_HOST_DEVICE char1 cuda_cast(vec<1, char> x);
CU_HOST_DEVICE char2 cuda_cast(vec<2, char> x);
CU_HOST_DEVICE char3 cuda_cast(vec<3, char> x);
CU_HOST_DEVICE char4 cuda_cast(vec<4, char> x);
CU_HOST_DEVICE uchar1 cuda_cast(vec<1, unsigned char> x);
CU_HOST_DEVICE uchar2 cuda_cast(vec<2, unsigned char> x);
CU_HOST_DEVICE uchar3 cuda_cast(vec<3, unsigned char> x);
CU_HOST_DEVICE uchar4 cuda_cast(vec<4, unsigned char> x);
CU_HOST_DEVICE short1 cuda_cast(vec<1, short> x);
CU_HOST_DEVICE short2 cuda_cast(vec<2, short> x);
CU_HOST_DEVICE short3 cuda_cast(vec<3, short> x);
CU_HOST_DEVICE short4 cuda_cast(vec<4, short> x);
CU_HOST_DEVICE ushort1 cuda_cast(vec<1, unsigned short> x);
CU_HOST_DEVICE ushort2 cuda_cast(vec<2, unsigned short> x);
CU_HOST_DEVICE ushort3 cuda_cast(vec<3, unsigned short> x);
CU_HOST_DEVICE ushort4 cuda_cast(vec<4, unsigned short> x);
CU_HOST_DEVICE int1 cuda_cast(ivec1 x);
CU_HOST_DEVICE int2 cuda_cast(ivec2 x);
CU_HOST_DEVICE int3 cuda_cast(ivec3 x);
CU_HOST_DEVICE int4 cuda_cast(ivec4 x);
CU_HOST_DEVICE uint1 cuda_cast(uvec1 x);
CU_HOST_DEVICE uint2 cuda_cast(uvec2 x);
CU_HOST_DEVICE uint3 cuda_cast(uvec3 x);
CU_HOST_DEVICE uint4 cuda_cast(uvec4 x);
CU_HOST_DEVICE long1 cuda_cast(vec<1, long> x);
CU_HOST_DEVICE long2 cuda_cast(vec<2, long> x);
CU_HOST_DEVICE long3 cuda_cast(vec<3, long> x);
CU_HOST_DEVICE long4 cuda_cast(vec<4, long> x);
CU_HOST_DEVICE ulong1 cuda_cast(vec<1, unsigned long> x);
CU_HOST_DEVICE ulong2 cuda_cast(vec<2, unsigned long> x);
CU_HOST_DEVICE ulong3 cuda_cast(vec<3, unsigned long> x);
CU_HOST_DEVICE ulong4 cuda_cast(vec<4, unsigned long> x);
CU_HOST_DEVICE float1 cuda_cast(vec1 x);
CU_HOST_DEVICE float2 cuda_cast(vec2 x);
CU_HOST_DEVICE float3 cuda_cast(vec3 x);
CU_HOST_DEVICE float4 cuda_cast(vec4 x);
CU_HOST_DEVICE double1 cuda_cast(dvec1 x);
CU_HOST_DEVICE double2 cuda_cast(dvec2 x);
CU_HOST_DEVICE double3 cuda_cast(dvec3 x);
CU_HOST_DEVICE double4 cuda_cast(dvec4 x);
CU_HOST_DEVICE vec<1, char> cuda_cast(char1 x);
CU_HOST_DEVICE vec<2, char> cuda_cast(char2 x);
CU_HOST_DEVICE vec<3, char> cuda_cast(char3 x);
CU_HOST_DEVICE vec<4, char> cuda_cast(char4 x);
CU_HOST_DEVICE vec<1, unsigned char> cuda_cast(uchar1 x);
CU_HOST_DEVICE vec<2, unsigned char> cuda_cast(uchar2 x);
CU_HOST_DEVICE vec<3, unsigned char> cuda_cast(uchar3 x);
CU_HOST_DEVICE vec<4, unsigned char> cuda_cast(uchar4 x);
CU_HOST_DEVICE vec<1, short> cuda_cast(short1 x);
CU_HOST_DEVICE vec<2, short> cuda_cast(short2 x);
CU_HOST_DEVICE vec<3, short> cuda_cast(short3 x);
CU_HOST_DEVICE vec<4, short> cuda_cast(short4 x);
CU_HOST_DEVICE vec<1, unsigned short> cuda_cast(ushort1 x);
CU_HOST_DEVICE vec<2, unsigned short> cuda_cast(ushort2 x);
CU_HOST_DEVICE vec<3, unsigned short> cuda_cast(ushort3 x);
CU_HOST_DEVICE vec<4, unsigned short> cuda_cast(ushort4 x);
CU_HOST_DEVICE ivec1 cuda_cast(int1 x);
CU_HOST_DEVICE ivec2 cuda_cast(int2 x);
CU_HOST_DEVICE ivec3 cuda_cast(int3 x);
CU_HOST_DEVICE ivec4 cuda_cast(int4 x);
CU_HOST_DEVICE uvec1 cuda_cast(uint1 x);
CU_HOST_DEVICE uvec2 cuda_cast(uint2 x);
CU_HOST_DEVICE uvec3 cuda_cast(uint3 x);
CU_HOST_DEVICE uvec4 cuda_cast(uint4 x);
CU_HOST_DEVICE vec<1, long> cuda_cast(long1 x);
CU_HOST_DEVICE vec<2, long> cuda_cast(long2 x);
CU_HOST_DEVICE vec<3, long> cuda_cast(long3 x);
CU_HOST_DEVICE vec<4, long> cuda_cast(long4 x);
CU_HOST_DEVICE vec<1, unsigned long> cuda_cast(ulong1 x);
CU_HOST_DEVICE vec<2, unsigned long> cuda_cast(ulong2 x);
CU_HOST_DEVICE vec<3, unsigned long> cuda_cast(ulong3 x);
CU_HOST_DEVICE vec<4, unsigned long> cuda_cast(ulong4 x);
CU_HOST_DEVICE vec1 cuda_cast(float1 x);
CU_HOST_DEVICE vec2 cuda_cast(float2 x);
CU_HOST_DEVICE vec3 cuda_cast(float3 x);
CU_HOST_DEVICE vec4 cuda_cast(float4 x);
CU_HOST_DEVICE dvec1 cuda_cast(double1 x);
CU_HOST_DEVICE dvec2 cuda_cast(double2 x);
CU_HOST_DEVICE dvec3 cuda_cast(double3 x);
CU_HOST_DEVICE dvec4 cuda_cast(double4 x);

}

#include "glm.inl"
