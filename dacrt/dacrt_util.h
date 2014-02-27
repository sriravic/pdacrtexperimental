#ifndef __DACRT_UTIL_H__
#define __DACRT_UTIL_H__

#pragma once
#include <global.h>

/// Contains some custom utility structures that we normally use throughout the dacrt program

struct Counters
{
	float	tri_filter_time;
	float	ray_filter_time;
	float	kernel_overall_time;
	float	kernel_time;
	float	mem_cpy_time;
	float	brute_force_time;
	float	seg_sort_time;
	float	reduction_time;
	float	update_min_time;
	float	trisortbykey_time;						
	float   trireduction_time;
	float   raysortbykey_time;
	float   rayreduction_time;
	float   misc_time;
	float   other_time1;			// for future use.!
	float   other_time2;
	float   other_time3;
	// the below are actual counters. 
	int		raybox;
	int		tribox;
	int		raytri;
	int		trifilter_sort_cnt;		// number of times the sorting operation is carried out.
	int		rayfilter_sort_cnt;
	std::vector<std::pair<int, float> > tri_sort_times;		// keep a track of all sort times.
	std::vector<std::pair<int, float> > ray_sort_times;
	Counters() {
		tri_filter_time = ray_filter_time = kernel_overall_time = kernel_time = mem_cpy_time = brute_force_time = seg_sort_time = reduction_time = update_min_time = 0.0f;
		trisortbykey_time = trireduction_time = raysortbykey_time = rayreduction_time = 0.0f;
		raybox = tribox = raytri = 0;
		trifilter_sort_cnt = 0, rayfilter_sort_cnt = 0;
		misc_time = other_time1 = other_time2 = other_time3 = 0;
	}
};

struct DacrtRunTimeParameters
{
	int BUFFER_SIZE;				
	int MAX_SEGMENTS;				
	int PARALLEL_TRI_THRESHOLD;		
	int PARALLEL_RAY_THRESHOLD;
	int NUM_RAYS_PER_BLOCK;
	int TRI_SHARED_MEMORY_SPACE;			// this parameter is given so that we can allocate shared memory dynamically
	int RAY_BUFFER_THRESHOLD;
	int TRI_BUFFER_THRESHOLD;
	int MAX_SEGMENT_THRESHOLD;
	size_t GRID_DIM_X;							// use a 2d grid for doing ray and triangle filtering operations.!
	size_t GRID_DIM_Y;			
	size_t GRID_DIM_Z;
};

struct ParallelPack
{
	thrust::device_vector<int>& buffered_ray_idx;
	thrust::device_vector<int>& buffered_tri_idx;
	thrust::device_vector<int>& segment_ids;
	thrust::device_vector<int>& ray_segment_sizes;
	thrust::device_vector<int>& tri_segment_sizes;
	thrust::device_vector<float>& buffered_ray_maxts;
	thrust::device_vector<int>& buffered_ray_hitids;
	thrust::device_vector<float>& dev_ray_maxts;
	thrust::device_vector<int>& dev_hitids;
	int& ray_buffer_occupied;
	int& tri_buffer_occupied;
	int& num_segments;
	
	// ctor
	ParallelPack(
		thrust::device_vector<int>& _buffered_ray_idx, 
		thrust::device_vector<int>& _buffered_tri_idx, 
		thrust::device_vector<int>& _segment_ids,
		thrust::device_vector<int>& _ray_segment_sizes, 
		thrust::device_vector<int>& _tri_segment_sizes, 
		thrust::device_vector<float>& _buffered_ray_maxts,
		thrust::device_vector<int>& _buffered_ray_hitids, 
		thrust::device_vector<float>& _dev_ray_maxts, 
		thrust::device_vector<int>& _dev_hitids, 
		int& _ray_buffered_occupied,
		int& _tri_buffer_occupied, 
		int& _num_segments): buffered_ray_idx(_buffered_ray_idx), 

		buffered_tri_idx(_buffered_tri_idx), 
		segment_ids(_segment_ids),
		ray_segment_sizes(_ray_segment_sizes), 
		tri_segment_sizes(_tri_segment_sizes), 
		buffered_ray_maxts(_buffered_ray_maxts), 
		buffered_ray_hitids(_buffered_ray_hitids),
		dev_ray_maxts(_dev_ray_maxts), 
		dev_hitids(_dev_hitids), 
		ray_buffer_occupied(_ray_buffered_occupied), 
		tri_buffer_occupied(_tri_buffer_occupied),
		num_segments(_num_segments) 
	{}	
};

// the parallel pack for the new modified version of dacrt in soa format.
// contains data and not the references for both the host and device vectors
struct ParallelPackModified
{
	thrust::host_vector<int> htri_segment_sizes;
	thrust::host_vector<int> hray_segment_sizes;
	thrust::host_vector<int> hsegment_ids;
	thrust::host_vector<int> blockNos;
	thrust::host_vector<int> blockStart;			// tells which block am I am within the segment
	// device data
	thrust::device_vector<int> buffered_ray_idx;
	thrust::device_vector<int> buffered_tri_idx;
	thrust::device_vector<int> segment_ids;
	thrust::device_vector<int> ray_segment_sizes;
	thrust::device_vector<int> tri_segment_sizes;
	thrust::device_vector<float> buffered_ray_maxts;
	thrust::device_vector<int> buffered_ray_hitids;
	thrust::device_vector<float> dev_ray_maxts;
	thrust::device_vector<int> dev_hitids;
	thrust::device_vector<int> dblockNos;
	thrust::device_vector<int> dblockStart;
	// common data
	size_t blockCnt;
	size_t bstart;
	size_t ray_buffer_occupied;
	size_t tri_buffer_occupied;
	size_t num_segments;

	ParallelPackModified();
	ParallelPackModified(DacrtRunTimeParameters& rtparams, int num_rays);
	// reset the block variables and the corresponding device data
	void clearBlockData();
	// clears all data related to device memory and the counter values also.
	void clearDeviceData();
	void clearHostData();
};

struct DoubleBuffer
{
public:
	DoubleBuffer(DacrtRunTimeParameters& rtparams, int num_rays);
	ParallelPackModified& getBuffer();
	void resetBuffer(int buffer_num);
private:
	ParallelPackModified buffer[2];
	bool flags[2];
	int to_use_buffer;						// which buffer should i fill?
};

struct PointLight
{
	float3 color;
	float3 position;
};

#endif