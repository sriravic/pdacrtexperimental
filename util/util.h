#ifndef __UTIL_H__
#define __UTIL_H__

#pragma once
#include <global.h>

typedef unsigned int uint32;
typedef int			 int32;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// morton codes generation
static __device__ __host__ unsigned int SeparateBy1(unsigned int x) {
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}

static __device__ __host__ unsigned int CompactBy1(unsigned int x) {
	x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

static __device__ __host__ unsigned int MortonCode2(unsigned int x, unsigned int y) {
	return SeparateBy1(x) | (SeparateBy1(y) << 1);
}

static __device__ __host__ void MortonDecode2(unsigned int c, unsigned int &x, unsigned int &y) {
	x = CompactBy1(c);
	y = CompactBy1(c >> 1);
}

__host__ __device__ inline unsigned int encodeMorton3(unsigned int x, unsigned int y, unsigned int z) {

	x &= 0x000003ff;
	x = (x ^ (x << 16)) & 0xff0000ff;
	x = (x ^ (x <<  8)) & 0x0300f00f;
	x = (x ^ (x <<  4)) & 0x030c30c3;
	x = (x ^ (x <<  2)) & 0x09249249;

	y &= 0x000003ff;
	y = (y ^ (y << 16)) & 0xff0000ff;
	y = (y ^ (y <<  8)) & 0x0300f00f;
	y = (y ^ (y <<  4)) & 0x030c30c3;
	y = (y ^ (y <<  2)) & 0x09249249;

	z &= 0x000003ff;
	z = (z ^ (z << 16)) & 0xff0000ff;
	z = (z ^ (z <<  8)) & 0x0300f00f;
	z = (z ^ (z <<  4)) & 0x030c30c3;
	z = (z ^ (z <<  2)) & 0x09249249;

	return (z << 2) | (y << 1) | x;
}

struct MortonCode
{
	unsigned int code;
	int x;
	int y;
	__device__ __host__ MortonCode() {};
	__host__ __host__ MortonCode(unsigned int _code, int _x, int _y):code(_code), x(_x), y(_y) {}
};

struct MortonCompare : public std::binary_function<MortonCode, MortonCode, bool>
{
	bool operator()(MortonCode x, MortonCode y) const
	{   
		return x.code < y.code;
	}
};

inline __host__ __device__ uint32 quantize(const float x, const uint32 n)
{
	return (uint32)max( min( int32( x * float(n) ), int32(n-1) ), int32(0) );
}

/// a convenience functor to compute the Morton code of a point sequences
/// relative to a given bounding box
struct MortonFunctor
{
    /// constructor
    ///
    /// \param bbox     global bounding box
    __host__ __device__ MortonFunctor(const float3& bmin, const float3& bmax) :
        m_base( bmin ),
        m_inv(make_float3((1.0f / (bmax.x - bmin.x)), (1.0f / (bmax.y - bmin.y)), (1.0f / (bmax.z - bmin.z))))
    {}


	template<typename Tuple>
	inline __host__ __device__ void operator() (Tuple vertices) {
		float3 v0 = (thrust::get<0>(vertices));
		float3 v1 = (thrust::get<1>(vertices));
		float3 v2 = (thrust::get<2>(vertices));
		float3 centroid = (v0 + v1 + v2)/3.0f;
		
		uint32 x = quantize( (centroid.x - m_base.x) * m_inv.x, 1024u );
        uint32 y = quantize( (centroid.y - m_base.y) * m_inv.y, 1024u );
        uint32 z = quantize( (centroid.z - m_base.z) * m_inv.z, 1024u );

		thrust::get<3>(vertices) = centroid;
        thrust::get<4>(vertices) = encodeMorton3( x, y, z );
	}

	inline __host__ __device__ uint32 operator() (const float3& point) const
    {
        uint32 x = quantize( (point.x - m_base.x) * m_inv.x, 1024u );
        uint32 y = quantize( (point.y - m_base.y) * m_inv.y, 1024u );
        uint32 z = quantize( (point.z - m_base.z) * m_inv.z, 1024u );

        return encodeMorton3( x, y, z );
    }
	const float3 m_base;
    const float3 m_inv;
};

struct AabbFunctor
{
	template<typename Tuple>
	__host__ __device__ void operator() (Tuple vertices) {
		float3 v0 = thrust::get<0>(vertices);
		float3 v1 = thrust::get<1>(vertices);
		float3 v2 = thrust::get<2>(vertices);
		float3 bmin = fminf(v0, fminf(v1, v2));
		float3 bmax = fmaxf(v0, fmaxf(v1, v2));
		thrust::get<3>(vertices) = bmin;
		thrust::get<4>(vertices) = bmax;
		thrust::get<5>(vertices) = (bmin + bmax) * 0.5f;
	}
};

#endif