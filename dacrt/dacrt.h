#ifndef __DACRT_H__
#define __DACRT_H__

/*
This file contains all the divide and conquer related code..
*/

#pragma once
#include <global.h>
#include <util/util.h>
#include <dacrt/dacrt_util.h>
#include <dacrt/dacrt_object_partitioning.h>
#include <dacrt/dacrt_spatial_partitioning.h>
#include <dacrt/dacrt_soa.h>
#include <util/logger.h>


// Method types
// Note: When writing a new way of rendering, just add the method's flag name here and modify the dacrt.cpp file with appropriate wrapper call
//       This should ease the way in which more complex stuff is added to the codebase.
enum Method {CPU_OBJECT_PRIMARY, CPU_OBJECT_SHADOW, CPU_OBJECT_SECONDARY,			// different passes of the scene
			 CPU_SPATIAL_PRIMARY, CPU_SPATIAL_SHADOW, CPU_SPATIAL_SECONDARY,
			 // this contains the filtering code done on the cpu and brute force done on the gpu
			 CPU_GPU_OBJECT_PRIMARY, CPU_GPU_OBJECT_SHADOW, CPU_GPU_OBJECT_SECONDARY,
			 CPU_GPU_SPATIAL_PRIMARY, CPU_GPU_SPATIAL_SHADOW, CPU_GPU_SPATIAL_SECONDARY,
			 CPU_GPU_SPATIAL_MODIFIED,
			 CPU_GPU_DBUFFER_SPATIAL_PRIMARY,
			 CPU_GPU_TWIN_TREES,
			 // these flags for all the work done on the gpu
			 GPU_OBJECT_PRIMARY, GPU_OBJECT_SHADOW, GPU_OBJECT_SECONDARY,
			 GPU_SPATIAL_PRIMARY, GPU_SPATIAL_SHADOW, GPU_SPATIAL_SECONDARY,
			 GPU_SPATIAL_PRIMARY_SEGMENTED, GPU_SPATIAL_SHADOW_SEGMENTED, GPU_SPATIAL_SECONDARY_SEGMENTED,
			 GPU_SPATIAL_CELL, GPU_SPATIAL_JUMP, GPU_SPATIAL_FULLY_PARALLEL, GPU_SPATIAL_FULLY_PARALLEL_MODIFIED,
			 GPU_DACRT_FULLY_CUDA, GPU_DACRT_FULLY_CUDA_SHADOW, GPU_DACRT_FULLY_CUDA_SECONDARY,
			 GPU_DACRT_FULLY_CUDA_OBJECT, GPU_DACRT_FULLY_CUDA_SHADOW_OBJECT, GPU_DACRT_FULLY_CUDA_SECONDARY_OBJECT,
			 GPU_CONES_CUDA, GPU_CUDA_AO,
			 
			 /* AoS Format methods come here..!*/
			 CPU_OBJECT_PRIMARY_AOS, CPU_OBJECT_SHADOW_AOS, CPU_OBJECT_SECONDARY_AOS,
			 CPU_SPATIAL_PRIMARY_AOS, CPU_SPATIAL_SHADOW_AOS, CPU_SPATIAL_SECONDARY_AOS,
			 CPU_GPU_OBJECT_PRIMARY_AOS, CPU_GPU_OBJECT_SHADOW_AOS, CPU_GPU_OBJECT_SECONDARY_AOS,
			 CPU_GPU_SPATIAL_PRIMARY_AOS, CPU_GPU_SPATIAL_SHADOW_AOS, CPU_GPU_SPATIAL_SECONDARY_AOS,
			 CPU_GPU_SPATIAL_MODIFIED_AOS,
			 CPU_GPU_DBUFFER_SPATIAL_PRIMARY_AOS,
			 CPU_GPU_TWIN_TREES_AOS,
			 GPU_OBJECT_PRIMARY_AOS, GPU_OBJECT_SHADOW_AOS, GPU_OBJECT_SECONDARY_AOS,
			 GPU_SPATIAL_PRIMARY_AOS, GPU_SPATIAL_SHADOW_AOS, GPU_SPATIAL_SECONDARY_AOS,
			 GPU_SPATIAL_PRIMARY_SEGMENTED_AOS, GPU_SPATIAL_SHADOW_SEGMENTED_AOS, GPU_SPATIAL_SECONDARY_SEGMENTED_AOS,
			 GPU_SPATIAL_CELL_AOS, GPU_SPATIAL_JUMP_AOS
			};								


// custom functors that we use for dacrt come here
// they are not utility classes because they form the core of the algorithms that we use. Hence we declare them here
template<typename T>
struct MinHitFunctor:public thrust::binary_function<T, T, T>
{
	__host__ __device__ const T operator() (const T& lhs, const T& rhs) const {
		return thrust::get<0>(lhs) < thrust::get<0>(rhs) ? lhs : rhs;
	}
};


// Note: This is the only method that is visible to other parts of the application. This makes the source code cleaner also.
//       Passing in void type and using it in the appropriate manner at the function call side.
void dacrt(void* renderdata, Method method, DacrtRunTimeParameters& rtparams, Counters& ctr, Logger& logger);


// SHADING METHODS

// This method shades only primary + one shadow (cpu only)
void cpuShade(const PointLight& pl, RayArray& rays, int width, int height, int raycnt, TriangleArray& triangles, int tricnt, float* maxts, int* hitids, float3* buffer,
	bool* shadow_hits, int num_shadow_rays, int* shadow_ray_idx, int* pray_association, MortonCode* mcodes, bool enable_shadows = false, 
	bool use_morton_codes = false);

void cpuShadeAos(const PointLight& pl, RayArrayAos& rays, int width, int height, int raycnt, TriangleArrayAos& triangles, int tricnt, float* maxts, int* hitids, float3* buffer,
	bool* shadow_hits, int num_shadow_rays, int* shadow_ray_idx, int* pray_association, MortonCode* mcodes, bool enable_shadows = false, 
	bool use_morton_codes = false);

#endif
