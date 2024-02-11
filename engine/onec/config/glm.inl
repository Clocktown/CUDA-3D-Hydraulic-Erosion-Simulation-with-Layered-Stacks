#include "glm.hpp"
#include <glm/glm.hpp>

namespace glm
{

CU_INLINE CU_HOST_DEVICE char1 cuda_cast(const vec<1, char> x)
{
    return char1{ x.x };
}

CU_INLINE CU_HOST_DEVICE char2 cuda_cast(const vec<2, char> x)
{
    return char2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE char3 cuda_cast(const vec<3, char> x)
{
    return char3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE char4 cuda_cast(const vec<4, char> x)
{
    return char4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE uchar1 cuda_cast(const vec<1, unsigned char> x)
{
    return uchar1{ x.x };
}

CU_INLINE CU_HOST_DEVICE uchar2 cuda_cast(const vec<2, unsigned char> x)
{
    return uchar2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE uchar3 cuda_cast(const vec<3, unsigned char> x)
{
    return uchar3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE uchar4 cuda_cast(const vec<4, unsigned char> x)
{
    return uchar4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE short1 cuda_cast(const vec<1, short> x)
{
    return short1{ x.x };
}

CU_INLINE CU_HOST_DEVICE short2 cuda_cast(const vec<2, short> x)
{
    return short2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE short3 cuda_cast(const vec<3, short> x)
{
    return short3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE short4 cuda_cast(const vec<4, short> x)
{
    return short4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE ushort1 cuda_cast(const vec<1, unsigned short> x)
{
    return ushort1{ x.x };
}

CU_INLINE CU_HOST_DEVICE ushort2 cuda_cast(const vec<2, unsigned short> x)
{
    return ushort2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE ushort3 cuda_cast(const vec<3, unsigned short> x)
{
    return ushort3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE ushort4 cuda_cast(const vec<4, unsigned short> x)
{
    return ushort4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE int1 cuda_cast(const ivec1 x)
{
    return int1{ x.x };
}

CU_INLINE CU_HOST_DEVICE int2 cuda_cast(const ivec2 x)
{
    return int2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE int3 cuda_cast(const ivec3 x)
{
    return int3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE int4 cuda_cast(const ivec4 x)
{
    return int4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE uint1 cuda_cast(const uvec1 x)
{
    return uint1{ x.x };
}

CU_INLINE CU_HOST_DEVICE uint2 cuda_cast(const uvec2 x)
{
    return uint2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE uint3 cuda_cast(const uvec3 x)
{
    return uint3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE uint4 cuda_cast(const uvec4 x)
{
    return uint4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE long1 cuda_cast(const vec<1, long> x)
{
    return long1{ x.x };
}

CU_INLINE CU_HOST_DEVICE long2 cuda_cast(const vec<2, long> x)
{
    return long2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE long3 cuda_cast(const vec<3, long> x)
{
    return long3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE long4 cuda_cast(const vec<4, long> x)
{
    return long4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE ulong1 cuda_cast(const vec<1, unsigned long> x)
{
    return ulong1{ x.x };
}

CU_INLINE CU_HOST_DEVICE ulong2 cuda_cast(const vec<2, unsigned long> x)
{
    return ulong2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE ulong3 cuda_cast(const vec<3, unsigned long> x)
{
    return ulong3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE ulong4 cuda_cast(const vec<4, unsigned long> x)
{
    return ulong4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE float1 cuda_cast(const vec1 x)
{
    return float1{ x.x };
}

CU_INLINE CU_HOST_DEVICE float2 cuda_cast(const vec2 x)
{
    return float2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE float3 cuda_cast(const vec3 x)
{
    return float3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE float4 cuda_cast(const vec4 x)
{
    return float4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE double1 cuda_cast(const dvec1 x)
{
    return double1{ x.x };
}

CU_INLINE CU_HOST_DEVICE double2 cuda_cast(const dvec2 x)
{
    return double2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE double3 cuda_cast(const dvec3 x)
{
    return double3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE double4 cuda_cast(const dvec4 x)
{
    return double4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE vec<1, char> cuda_cast(const char1 x)
{
    return vec<1, char>{ x.x };
}

CU_INLINE CU_HOST_DEVICE vec<2, char> cuda_cast(const char2 x)
{
    return vec<2, char>{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE vec<3, char> cuda_cast(const char3 x)
{
    return vec<3, char>{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE vec<4, char> cuda_cast(const char4 x)
{
    return vec<4, char>{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE vec<1, unsigned char> cuda_cast(const uchar1 x)
{
    return vec<1, unsigned char>{ x.x };
}

CU_INLINE CU_HOST_DEVICE vec<2, unsigned char> cuda_cast(const uchar2 x)
{
    return vec<2, unsigned char>{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE vec<3, unsigned char> cuda_cast(const uchar3 x)
{
    return vec<3, unsigned char>{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE vec<4, unsigned char> cuda_cast(const uchar4 x)
{
    return vec<4, unsigned char>{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE vec<1, short> cuda_cast(const short1 x)
{
    return vec<1, short>{ x.x };
}

CU_INLINE CU_HOST_DEVICE vec<2, short> cuda_cast(const short2 x)
{
    return vec<2, short>{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE vec<3, short> cuda_cast(const short3 x)
{
    return vec<3, short>{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE vec<4, short> cuda_cast(const short4 x)
{
    return vec<4, short>{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE vec<1, unsigned short> cuda_cast(const ushort1 x)
{
    return vec<1, unsigned short>{ x.x };
}

CU_INLINE CU_HOST_DEVICE vec<2, unsigned short> cuda_cast(const ushort2 x)
{
    return vec<2, unsigned short>{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE vec<3, unsigned short> cuda_cast(const ushort3 x)
{
    return vec<3, unsigned short>{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE vec<4, unsigned short> cuda_cast(const ushort4 x)
{
    return vec<4, unsigned short>{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE ivec1 cuda_cast(const int1 x)
{
    return ivec1{ x.x };
}

CU_INLINE CU_HOST_DEVICE ivec2 cuda_cast(const int2 x)
{
    return ivec2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE ivec3 cuda_cast(const int3 x)
{
    return ivec3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE ivec4 cuda_cast(const int4 x)
{
    return ivec4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE uvec1 cuda_cast(const uint1 x)
{
    return uvec1{ x.x };
}

CU_INLINE CU_HOST_DEVICE uvec2 cuda_cast(const uint2 x)
{
    return uvec2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE uvec3 cuda_cast(const uint3 x)
{
    return uvec3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE uvec4 cuda_cast(const uint4 x)
{
    return uvec4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE vec<1, long> cuda_cast(const long1 x)
{
    return vec<1, long>{ x.x };
}

CU_INLINE CU_HOST_DEVICE vec<2, long> cuda_cast(const long2 x)
{
    return vec<2, long>{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE vec<3, long> cuda_cast(const long3 x)
{
    return vec<3, long>{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE vec<4, long> cuda_cast(const long4 x)
{
    return vec<4, long>{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE vec<1, unsigned long> cuda_cast(const ulong1 x)
{
    return vec<1, unsigned long>{ x.x };
}

CU_INLINE CU_HOST_DEVICE vec<2, unsigned long> cuda_cast(const ulong2 x)
{
    return vec<2, unsigned long>{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE vec<3, unsigned long> cuda_cast(const ulong3 x)
{
    return vec<3, unsigned long>{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE vec<4, unsigned long> cuda_cast(const ulong4 x)
{
    return vec<4, unsigned long>{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE vec1 cuda_cast(const float1 x)
{
    return vec1{ x.x };
}

CU_INLINE CU_HOST_DEVICE vec2 cuda_cast(const float2 x)
{
    return vec2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE vec3 cuda_cast(const float3 x)
{
    return vec3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE vec4 cuda_cast(const float4 x)
{
    return vec4{ x.x, x.y, x.z, x.w };
}

CU_INLINE CU_HOST_DEVICE dvec1 cuda_cast(const double1 x)
{
    return dvec1{ x.x };
}

CU_INLINE CU_HOST_DEVICE dvec2 cuda_cast(const double2 x)
{
    return dvec2{ x.x, x.y };
}

CU_INLINE CU_HOST_DEVICE dvec3 cuda_cast(const double3 x)
{
    return dvec3{ x.x, x.y, x.z };
}

CU_INLINE CU_HOST_DEVICE dvec4 cuda_cast(const double4 x)
{
    return dvec4{ x.x, x.y, x.z, x.w };
}

}
