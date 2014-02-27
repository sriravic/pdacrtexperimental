#include <util/scene.h>
#include <renderer/renderer.h>

struct NumShadows
{
	__device__ __host__ int operator() (int val1, int val2) { return ((val1 > 0) + (val2 > 0)); }
};

struct IsHit
{
	__device__ __host__ bool operator() (int val) { return val != -1; }
};

struct ShadowRayGenerator
{
	__device__ __host__ ShadowRayGenerator(const float3& light_pos):pt_light_pos(light_pos) {}
	template<typename Tuple>
	__device__ __host__ void operator() (Tuple T) {
		// we pass the original ray's origin + direction
		// plus we pass the hit location
		// we get back with the origin + direction of returned ray
		float3 i_origin = thrust::get<0>(T);
		float3 i_direction = thrust::get<1>(T);
		float  i_t = thrust::get<2>(T);
		float3 xt_pt = i_origin + i_t * i_direction;
		float3 ret_dir = normalize(pt_light_pos - xt_pt);
		float3 ret_o = xt_pt + ret_dir * 0.00001f;
		thrust::get<3>(T) = ret_o;
		thrust::get<4>(T) = ret_dir;
	}
	float3 pt_light_pos;
};

/*
extern "C"
float distance(const float3& pt1, const float3& pt2)  {
	return sqrtf((pt1.x - pt2.x ) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) + (pt1.z - pt2.z) * (pt1.z - pt2.z));
}
*/


template<typename T>
__global__ void createIdxArray(T* in_array, size_t N) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N) {
		in_array[tid] = (T)tid;
	}
}


extern "C"
void renderCallGpuSpatialPrimary(Scene* scene, RayArray& d_rays, int** d_ray_idx_array, float* h_maxts, int* h_hitids, int num_primary_rays, DacrtRunTimeParameters& rtparams, Counters& primary_ray_ctr, Logger& logger) {

	thrust::device_vector<float> d_maxts(num_primary_rays, FLT_MAX);
	thrust::device_vector<int> d_hitids(num_primary_rays, -1);
	GpuPrimaryPassRenderData data(scene->scene_box, scene->d_triangles, scene->d_tri_idx_array, scene->num_triangles, d_rays,
		*d_ray_idx_array, num_primary_rays, thrust::raw_pointer_cast(&d_maxts[0]), thrust::raw_pointer_cast(&d_hitids[0])/*, h_maxts, h_hitids*/);
			// we have to do a copy of data here
	dacrt((void*)&data, GPU_SPATIAL_PRIMARY, rtparams, primary_ray_ctr, logger);
	
	// copy the data back
	thrust::copy(d_maxts.begin(), d_maxts.end(), h_maxts);
	thrust::copy(d_hitids.begin(), d_hitids.end(), h_hitids);
}

void Renderer::setupShadowRaysCuda(RayArray& d_primary_rays, RayArray& d_shadow_rays, int* d_hitids, float* d_maxts, int* d_pray_idx_array,
	int** d_sray_idx_array, int** d_spray_idx_array, int& num_shadow_rays) {
	
	// first we copy all the host data to device memory
	int num_primary_rays = scene->width * scene->height;
	int* hitcnt = NULL;
	checkCuda(cudaMalloc((void**)&hitcnt, sizeof(int) * num_primary_rays));
	thrust::transform(thrust::device_ptr<int>(d_hitids), thrust::device_ptr<int>(d_hitids) + num_primary_rays, thrust::device_ptr<int>(hitcnt),
		IsHit());
	num_shadow_rays = thrust::reduce(thrust::device_ptr<int>(hitcnt), thrust::device_ptr<int>(hitcnt) + num_primary_rays);
	
	// allocate enough memory and
	//d_spray_idx_array = NULL;					// shadow primary association list
	//d_sray_idx_array = NULL;					// shadow ray idx
	float*  d_pray_hits	     = NULL;			// we copy only those that have been hit to generate new rays
	float3* d_pray_origin    = NULL;
	float3* d_pray_dirs	     = NULL;

	checkCuda(cudaMalloc((void**)&(*d_spray_idx_array), sizeof(int) * num_shadow_rays));
	checkCuda(cudaMalloc((void**)&(*d_sray_idx_array), sizeof(int) * num_shadow_rays));
	checkCuda(cudaMalloc((void**)&d_pray_hits, sizeof(float) * num_shadow_rays));
	checkCuda(cudaMalloc((void**)&d_pray_origin, sizeof(float3) * num_shadow_rays));
	checkCuda(cudaMalloc((void**)&d_pray_dirs, sizeof(float3) * num_shadow_rays));
		
	thrust::sequence(thrust::device_ptr<int>(*d_sray_idx_array), thrust::device_ptr<int>(*d_sray_idx_array) + num_shadow_rays, 0, 1);
	thrust::copy_if(thrust::device_ptr<int>(d_pray_idx_array), thrust::device_ptr<int>(d_pray_idx_array) + num_primary_rays, 
		thrust::device_ptr<int>(d_hitids), thrust::device_ptr<int>(*d_spray_idx_array), IsHit());
	thrust::copy_if(thrust::device_ptr<float>(d_maxts), thrust::device_ptr<float>(d_maxts) + num_primary_rays,
		thrust::device_ptr<int>(d_hitids), thrust::device_ptr<float>(d_pray_hits), IsHit());
	thrust::copy_if(thrust::device_ptr<float3>(d_primary_rays.o), thrust::device_ptr<float3>(d_primary_rays.o) + num_primary_rays,
		thrust::device_ptr<int>(d_hitids), thrust::device_ptr<float3>(d_pray_origin), IsHit());
	thrust::copy_if(thrust::device_ptr<float3>(d_primary_rays.d), thrust::device_ptr<float3>(d_primary_rays.d) + num_primary_rays,
		thrust::device_ptr<int>(d_hitids), thrust::device_ptr<float3>(d_pray_dirs), IsHit());

	checkCuda(cudaMalloc((void**)&d_shadow_rays.o, sizeof(float3) * num_shadow_rays));
	checkCuda(cudaMalloc((void**)&d_shadow_rays.d, sizeof(float3) * num_shadow_rays));
			
	ShadowRayGenerator sgen(make_float3(-0.50f, 1.8f, -1.0f));
	
	// now we can generate rays
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<float3>(d_pray_origin),
																  thrust::device_ptr<float3>(d_pray_dirs),
																  thrust::device_ptr<float>(d_pray_hits),
																  thrust::device_ptr<float3>(d_shadow_rays.o),
																  thrust::device_ptr<float3>(d_shadow_rays.d))),
					 thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<float3>(d_pray_origin) + num_shadow_rays,
																  thrust::device_ptr<float3>(d_pray_dirs) + num_shadow_rays,
																  thrust::device_ptr<float>(d_pray_hits) + num_shadow_rays,
																  thrust::device_ptr<float3>(d_shadow_rays.o) + num_shadow_rays,
																  thrust::device_ptr<float3>(d_shadow_rays.d) + num_shadow_rays)),
					 sgen			
	);

	// free up space
	CUDA_SAFE_RELEASE(hitcnt);
	CUDA_SAFE_RELEASE(d_pray_hits);
	CUDA_SAFE_RELEASE(d_pray_origin);
	CUDA_SAFE_RELEASE(d_pray_dirs);
}


// NOTE: This method has separate values for rpivot and tpivot thereby enabling us to do pre filtering operations on the data.
extern "C"
void renderCallGpuSpatialPrimaryAos(Scene* scene, RayArrayAos& d_rays, int** d_ray_idx_array, float* h_maxts, int* h_hitids, int num_primary_rays, int rpivot, DacrtRunTimeParameters& rtparams, Counters& primary_ray_ctr, Logger& logger) {

	thrust::device_vector<float> d_maxts(num_primary_rays, FLT_MAX);
	thrust::device_vector<int> d_hitids(num_primary_rays, -1);
	GpuPrimaryPassRenderDataAos data(scene->scene_box_aos, scene->d_triangles_aos, scene->d_tri_idx_array, scene->num_triangles, scene->num_triangles, d_rays,
		*d_ray_idx_array, num_primary_rays, rpivot, thrust::raw_pointer_cast(&d_maxts[0]), thrust::raw_pointer_cast(&d_hitids[0])/*, h_maxts, h_hitids*/);
			// we have to do a copy of data here
	dacrt((void*)&data, GPU_SPATIAL_PRIMARY_AOS, rtparams, primary_ray_ctr, logger);
	
	// copy the data back
	thrust::copy(d_maxts.begin(), d_maxts.end(), h_maxts);
	thrust::copy(d_hitids.begin(), d_hitids.end(), h_hitids);
}

