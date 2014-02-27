#ifndef __RENDER_DATA_H__
#define __RENDER_DATA_H__

#pragma once
#include <primitives/primitives.h>

/**
Render Data
***********

Render data is a custom struct for packing up stuff from the scene to be passed to the dacrt method. We will have an abstract structure that
we can use to pack data in a very generic way. Data will be of the format <space, triangles, triangleid, rays, rayid>

Since this is the most common form, we use this. Primary rays and other passes can use this base class data as well as pass their own data
*/

struct RenderData 
{
		// base default for all classes/CPU/GPU alike
};

struct CpuRenderData
{
	// very basic data required by all
	AABB& space;
	int* tri_idx_array;
	int* ray_idx_array;
	TriangleArray& triangles;
	RayArray& rays;
	AabbArray& tri_aabbs;
	int num_rays;
	int num_triangles;

	CpuRenderData(AABB& _space, TriangleArray& tris, int* triids, AabbArray& aabbs, int ntris, RayArray& _rays, int* rids, int nrays):space(_space), triangles(tris), rays(_rays), tri_aabbs(aabbs) {
		tri_idx_array = triids;
		ray_idx_array = rids;
		num_rays = nrays;
		num_triangles = ntris;
	}
};

// Primary pass render data is totally different in the way it handles data. It also uses a hitid + maxts value array to do stuff.
// assuming callee has already allocated memory and passes only valid references/pointers.
// WARNING: No checking of validity is performed.!
struct CpuPrimaryPassRenderData : public CpuRenderData
{
	int* hitids;
	float* maxts;
	CpuPrimaryPassRenderData(AABB& space, TriangleArray& tris, int* triids, AabbArray& aabbs, int ntris, RayArray& _rays, int* rids, int nrays, float* _maxts, int* _hitids):CpuRenderData(space,
	tris, triids, aabbs, ntris, _rays, rids, nrays) {
		hitids = _hitids;
		maxts = _maxts;
	}
};

struct CpuShadowPassData {
	
	AABB& space;
	int* tri_idx_array;
	int* sray_idx_array;
	int* pray_idx_array;
	TriangleArray& triangles;
	RayArray& rays;
	int num_rays;
	int num_triangles;
	int tpivot;
	int rpivot;
	bool* shadows;
	CpuShadowPassData(AABB& _space, TriangleArray& _tris, int* triids, int ntris, int _tpivot, RayArray& _rays, int* shadow_rids, int* pray_ids, int nrays, int _rpivot, bool* _shadow):space(_space),
		triangles(_tris), rays(_rays) {
			tri_idx_array = triids;
			sray_idx_array = shadow_rids;
			pray_idx_array = pray_ids;
			num_rays = nrays;
			num_triangles = ntris;
			tpivot = _tpivot;
			rpivot = _rpivot;
			shadows = _shadow;
	}
};

struct CpuGpuPrimaryPassRenderData : public CpuPrimaryPassRenderData
{
	// we just have to add the dev triangles and dev rays
	TriangleArray& dev_triangles;
	RayArray& dev_rays;
	CpuGpuPrimaryPassRenderData(AABB& space, TriangleArray& tris, TriangleArray& devtris, int* triids, AabbArray& aabbs, int ntris, 
		RayArray& _rays, RayArray& _devrays, int* rids, int nrays, float* _maxts, int* _hitids):CpuPrimaryPassRenderData(space, tris, triids, aabbs, ntris, _rays, rids, nrays, _maxts, _hitids), dev_triangles(devtris),
		dev_rays(_devrays) {
	}
};

struct GpuPrimaryPassRenderData 
{
	AABB& space;
	TriangleArray& d_triangles;
	RayArray& d_rays;
	int* dtri_idx_array;
	int* dray_idx_array;
	int* d_hitids;
	float* d_maxts;
	//int* h_hitids;
	//float* h_maxts;
	int num_triangles;
	int num_rays;
	GpuPrimaryPassRenderData(AABB& _space, TriangleArray& dtris, int* dtriids, int ntris, RayArray& drays, int* drayids, int nrays, float* _d_maxts,
		int* _d_hitids /*,float* _h_maxts, int* _h_hitids*/):space(_space), d_triangles(dtris), d_rays(drays) {
			dtri_idx_array = dtriids;
			dray_idx_array = drayids;
			d_maxts = _d_maxts;
			d_hitids = _d_hitids;
			//h_maxts = _h_maxts;
			//h_hitids = _h_hitids;
			num_triangles = ntris;
			num_rays = nrays;
	}
};

// data for cell
struct GpuCellData
{
	AABB& scene_box;
	TriangleArray& d_triangles;
	RayArray& d_rays;
	int* dtri_idx_array;
	int* dray_idx_array;
	float* h_maxts;
	int* h_hitids;
	int num_triangles;
	int num_rays;
	int tpivot;
	int rpivot;
	GpuCellData(AABB& _scene_box, TriangleArray& _dtris, RayArray& _drays, int* _dtri_ids, int* _dray_ids, int _num_tris, int _num_rays, int _tpivot,
		int _rpivot, float* _h_maxts, int* _h_hitids):scene_box(_scene_box), d_triangles(_dtris), d_rays(_drays) {
			dtri_idx_array = _dtri_ids;
			dray_idx_array = _dray_ids;
			h_hitids = _h_hitids;
			h_maxts = _h_maxts;
			num_triangles = _num_tris;
			num_rays = _num_rays;
			tpivot = _tpivot;
			rpivot = _rpivot;
	}
};

struct GpuFullyParallelData
{
	
	AABB& scene_box;
	TriangleArray& d_triangles;
	RayArray& d_rays;
	int* dtri_idx_array;
	int* dray_idx_array;
	float* h_maxts;
	int* h_hitids;
	int num_triangles;
	int num_rays;
	int tpivot;
	int rpivot;
	GpuFullyParallelData(AABB& _scene_box, TriangleArray& _dtris, RayArray& _drays, int* _dtri_ids, int* _dray_ids, int _num_tris, int _num_rays, int _tpivot,
		int _rpivot, float* _h_maxts, int* _h_hitids):scene_box(_scene_box), d_triangles(_dtris), d_rays(_drays) {
			dtri_idx_array = _dtri_ids;
			dray_idx_array = _dray_ids;
			h_hitids = _h_hitids;
			h_maxts = _h_maxts;
			num_triangles = _num_tris;
			num_rays = _num_rays;
			tpivot = _tpivot;
			rpivot = _rpivot;
	}
};

struct CpuGpuTwinData
{
	AABB& scene_box;
	TriangleArray& d_triangles;
	RayArray& d_rays;
	TriangleArray& h_triangles;
	RayArray& h_rays;
	int* dtri_idx_array;
	int* dray_idx_array;
	int* htri_idx_array;
	int* hray_idx_array;
	float* h_maxts;
	int* h_hitids;
	int num_triangles;
	int num_rays;
	int tpivot;
	int rpivot;
	CpuGpuTwinData(AABB& _scene_box, TriangleArray& _dtris, TriangleArray& _htris, RayArray& _drays, RayArray& _hrays, int* _dtri_idx_array, int* _dray_idx_array,
		int* _htri_idx_array, int* _hray_idx_array, float* _h_maxts, int* _h_hitids, int _ntris, int _nrays, int _tpivot, int _rpivot):scene_box(_scene_box),
		d_triangles(_dtris), h_triangles(_htris), d_rays(_drays), h_rays(_hrays) {

			dtri_idx_array = _dtri_idx_array;
			dray_idx_array = _dray_idx_array;
			htri_idx_array = _htri_idx_array;
			hray_idx_array = _hray_idx_array;
			h_maxts = _h_maxts;
			h_hitids = _h_hitids;
			num_triangles = _ntris;
			num_rays = _nrays;
			tpivot = _tpivot;
			rpivot = _rpivot;
	}
};

// AoS data 
struct GpuPrimaryPassRenderDataAos 
{
	AABB4& space;
	TriangleArrayAos& d_triangles;
	RayArrayAos& d_rays;
	int* dtri_idx_array;
	int* dray_idx_array;
	int* d_hitids;
	float* d_maxts;
	//int* h_hitids;
	//float* h_maxts;
	int num_triangles;
	int num_rays;
	int tpivot;
	int rpivot;
	GpuPrimaryPassRenderDataAos(AABB4& _space, TriangleArrayAos& dtris, int* dtriids, int ntris, int _tpivot, RayArrayAos& drays, int* drayids, int nrays, int _rpivot, float* _d_maxts,
		int* _d_hitids /*,float* _h_maxts, int* _h_hitids*/):space(_space), d_triangles(dtris), d_rays(drays), tpivot(_tpivot), rpivot(_rpivot) {
			dtri_idx_array = dtriids;
			dray_idx_array = drayids;
			d_maxts = _d_maxts;
			d_hitids = _d_hitids;
			//h_maxts = _h_maxts;
			//h_hitids = _h_hitids;
			num_triangles = ntris;
			num_rays = nrays;
	}
};



#endif