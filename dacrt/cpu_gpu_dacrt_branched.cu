#include <dacrt/dacrt.h>
#include <util/logger.h>
#include <util/util.h>
#include <util/cutimer.h>
#include <pthread.h>

struct CpuData
{
	AABB& root;
	TriangleArray& triangles;
	RayArray& rays;
	int* h_tri_idx_array;
	int* h_ray_idx_array;
	float* h_maxts;
	int* h_hitds;
	int num_triangles;
	int num_rays;
	int tpivot;
	int rpivot;
	Counters& cpuctr;
	// add logger not for now. cpu thread safe?
	CpuData(AABB& _root, TriangleArray& hTriangles, RayArray& hRays, int* tri_idx_array, int* ray_idx_array, float* maxts, int* hitids, int ntris, int nrays, int _tpivot, int _rpivot, Counters& _cpuctr):root(_root),
		triangles(hTriangles), rays(hRays), cpuctr(_cpuctr) {
			h_tri_idx_array = tri_idx_array;
			h_ray_idx_array = ray_idx_array;
			h_maxts = maxts;
			h_hitds = hitids;
			num_triangles = ntris;
			num_rays = nrays;
			tpivot = _tpivot;
			rpivot = _rpivot;
	}
};

struct GpuData
{
	AABB& root;
	TriangleArray& triangles;
	RayArray& rays;
	int* d_tri_idx_array;
	int* d_ray_idx_array;
	float* h_maxts;
	int* h_hitids;
	int num_triangles;
	int num_rays;
	int tpivot;
	int rpivot;
	DacrtRunTimeParameters& rtparams;
	Counters& ctr;
	Logger& logger;
	GpuData(AABB& _root, TriangleArray& dTriangles, RayArray& dRays, int* tri_idx_array, int* ray_idx_array, float* maxts, int* hitids, int ntris, int nrays, int _tpivot, int _rpivot, DacrtRunTimeParameters& _rtparams,
		Counters& _ctr, Logger& _logger):root(_root), triangles(dTriangles), rays(dRays), rtparams(_rtparams), ctr(_ctr), logger(_logger) {
			d_tri_idx_array = tri_idx_array;
			d_ray_idx_array = ray_idx_array;
			h_maxts = maxts;
			h_hitids = hitids;
			num_triangles = ntris;
			num_rays = nrays;
			tpivot = _tpivot;
			rpivot = _rpivot;
		}
};


void* cpuDacrtThreadMethod(void* pdata) {
	double start, end;
	start = omp_get_wtime();
	CpuData* data = (CpuData*)pdata;
	// call the method
	std::cout<<"CPU thread starting cpu dacrt spatial partitioning\n"<<std::endl;
	cpuDacrtSpatialPartitioning(data->root, data->triangles, data->num_triangles, data->h_tri_idx_array, data->tpivot, data->rays, data->num_rays, data->h_ray_idx_array, data->rpivot, data->h_maxts, data->h_hitds, data->cpuctr);
	end = omp_get_wtime();
	std::cout<<"CPU thread completed in :"<<end-start<<" seconds\n"<<std::endl;
	pthread_exit(NULL);
	return NULL;
}

void* gpuDacrtThreadMethod(void* pdata) {
	double start, end;
	GpuData* data = (GpuData*)pdata;
	start = omp_get_wtime();
	//gpuDacrtCell(data->root, data->triangles, data->d_ray_idx_array, data->tpivot, data->rays, data->d_ray_idx_array, data->rpivot, data->h_maxts, data->h_hitids,
	//	data->rtparams, data->ctr, data->logger);
	gpuDacrtSpatialSegmented(data->root, data->triangles, data->d_tri_idx_array, data->num_triangles, data->tpivot, data->rays, data->d_ray_idx_array, data->num_rays, data->rpivot,
		data->h_maxts, data->h_hitids, data->rtparams, data->ctr, data->logger);
	end = omp_get_wtime();
	std::cout<<"GPU thread completed in : "<<end-start<<" seconds\n"<<std::endl;
	pthread_exit(NULL);
	return NULL;
}


void cpuGpuDacrtBranched(/*Method cpu_method, Method gpu_method,*/ 
	AABB& root, 
	TriangleArray& hTriangles, TriangleArray& dTriangles,
	int* h_tri_idx_array, int* d_tri_idx_array,
	int num_triangles, int tpivot,
	RayArray& hRays, RayArray& dRays,
	int* h_ray_idx_array, int* d_ray_idx_array,
	int num_rays, int rpivot,
	float* h_maxts, int* h_hitids,
	DacrtRunTimeParameters& rtparams,
	Counters& ctr,
	Logger& logger) {

		// Left will be GPU, right will be CPU
		// we will launch two separate threads of cpu that call the work.
		// NOTE: make sure that CPU and GPU are pure methods.
		AABB right, left;
		float split_pos;
		splitSpatialMedian(root, left, right, split_pos);

		// create two work packets for both the cpu and gpu and launch their appropriate threads.
		// for now, we are assuming cpu is cpu_primary_spatial and gpu is gpu_cell approach
		pthread_t cpu_thread, gpu_thread;

		/// NOTE: cell method also employs h_maxts and h_hitids only. So we create a temporary buffer here, get that data, and then resolve this minimum and hitid values
		float* h_temp_maxts = new float[num_rays];
		int* h_temp_hitids = new int[num_rays];
		// assume data is created for now.
		Counters cpu_ctr, gpu_ctr;
		CpuData cdata(right, hTriangles, hRays, h_tri_idx_array, h_ray_idx_array, h_maxts, h_hitids, num_triangles, num_rays, tpivot, rpivot, cpu_ctr);
		GpuData gdata(left, dTriangles, dRays, d_tri_idx_array, d_ray_idx_array, h_temp_maxts, h_temp_hitids, num_triangles, num_rays, tpivot, rpivot, rtparams, gpu_ctr, logger);

		// create two threads and let them run
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		pthread_create(&cpu_thread, &attr, cpuDacrtThreadMethod, (void*)&cdata);
		pthread_create(&gpu_thread, &attr, gpuDacrtThreadMethod, (void*)&gdata);

		// wait for the two threads.
		// and then recombine the results
		pthread_attr_destroy(&attr);
		void* status;
		pthread_join(cpu_thread, &status);
		pthread_join(gpu_thread, &status);

		// now we have to do an update of all the min values
#pragma omp parallel for
		for(int i = 0; i < num_rays; i++) {
			if(h_temp_maxts[i] < h_maxts[i] && h_temp_maxts[i] > 0 && h_temp_hitids[i] != -1) {
				h_maxts[i] = h_temp_maxts[i];
				h_hitids[i] = h_temp_hitids[i];
			}
		}

		// free up the memory
		SAFE_RELEASE(h_temp_hitids);
		SAFE_RELEASE(h_temp_maxts);
		
}
	
