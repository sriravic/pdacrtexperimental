#include <util/scene.h>
#include <renderer/renderer.h>
#include <io/io.h>
#include <util/logger.h>
#include <util/myrandom.h>

//#define USE_MORTON_CODES 1

// some extern functions for seamless cuda usage
extern "C" void renderCallGpuSpatialPrimary(Scene* scene, RayArray& d_rays, int** d_ray_idx_array, float* h_maxts, int* h_hitids, int num_primary_rays, DacrtRunTimeParameters& rtparams, Counters& primary_ray_ctr, Logger& logger);
extern "C" void renderCallGpuSpatialPrimaryAos(Scene* scene, RayArrayAos& d_rays, int** d_ray_idx_array, float* h_maxts, int* h_hitids, int num_primary_rays, int rpivot, DacrtRunTimeParameters& rtparams, Counters& primary_ray_ctr, Logger& logger);

// ao rays dump file 
//extern "C" void cpuShadeAmbientOcclusion(int width, int height, int raycnt, int* hitids, float3* buffer, int num_ao_rays, float* ao, int* ao_rays_id, int* pray_association);

extern unsigned int total_device_bytes;

inline float getDistance(const float3& pt1, const float3& pt2) {
	return sqrtf((pt1.x - pt2.x ) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y) + (pt1.z - pt2.z) * (pt1.z - pt2.z));
}


int Renderer::frame_no = 0;		

Renderer::Renderer(Scene* _scene, const std::string& data_fmt, const char* _output, const char* _ao_output, bool morton_enable, bool shadow_enable, bool secondary_enable, const float3& pt_light_pos, bool set_rtparams, bool _use_cuda) {
	scene            = _scene;
	data_format      = data_fmt;
	output           = _output;
	ao_output        = _ao_output;
	max_depth        = 0;				// for now
	use_morton_codes = morton_enable;
	enable_shadows   = shadow_enable;
	enable_secondary = secondary_enable;
	use_cuda         = _use_cuda;
	logfile          = new Logger("log.csv");
	primary_mcodes   = NULL;
	buffer           = NULL;
	shadow_buffer	 = NULL;
	secondary_buffer = NULL;
	point_light_pos  = pt_light_pos;
	if(!set_rtparams) {
		// set the default values
		std::cout<<"Setting Default RTPARAM values\n";
		rtparams.BUFFER_SIZE = 4 * 1024 * 1024; rtparams.MAX_SEGMENTS = 9000; rtparams.MAX_SEGMENT_THRESHOLD = 5; rtparams.NUM_RAYS_PER_BLOCK = 256;
		rtparams.PARALLEL_RAY_THRESHOLD = 256; rtparams.PARALLEL_TRI_THRESHOLD = 256; rtparams.TRI_SHARED_MEMORY_SPACE = 256;
	}
}

Renderer::Renderer(Scene* _scene, const std::string& data_fmt, const char* _output, const char* _ao_output, bool morton_enable, bool shadow_enable, bool secondary_enable, DacrtRunTimeParameters& _rtparams, const float3& pt_light_pos, bool _use_cuda) {
	scene            = _scene;
	data_format      = data_fmt;
	output           = _output;
	ao_output        = _ao_output;
	max_depth        = 0;				// for now
	use_morton_codes = morton_enable;
	enable_shadows   = shadow_enable;
	enable_secondary = secondary_enable;
	use_cuda         = _use_cuda;
	logfile	         = new Logger("log.csv");
	primary_mcodes   = NULL;
	buffer           = NULL;
	shadow_buffer    = NULL;
	secondary_buffer = NULL;
	point_light_pos  = pt_light_pos;
	rtparams.BUFFER_SIZE = _rtparams.BUFFER_SIZE; rtparams.MAX_SEGMENTS = _rtparams.MAX_SEGMENTS; rtparams.MAX_SEGMENT_THRESHOLD = _rtparams.MAX_SEGMENT_THRESHOLD; 
	rtparams.NUM_RAYS_PER_BLOCK = _rtparams.NUM_RAYS_PER_BLOCK; rtparams.PARALLEL_RAY_THRESHOLD = _rtparams.PARALLEL_RAY_THRESHOLD; 
	rtparams.PARALLEL_TRI_THRESHOLD = _rtparams.PARALLEL_TRI_THRESHOLD; rtparams.TRI_SHARED_MEMORY_SPACE = _rtparams.TRI_SHARED_MEMORY_SPACE;
	rtparams.RAY_BUFFER_THRESHOLD = _rtparams.RAY_BUFFER_THRESHOLD; rtparams.TRI_BUFFER_THRESHOLD = _rtparams.TRI_BUFFER_THRESHOLD;
	rtparams.GRID_DIM_X = _rtparams.GRID_DIM_X; rtparams.GRID_DIM_Y = _rtparams.GRID_DIM_Y; rtparams.GRID_DIM_Z = _rtparams.GRID_DIM_Z;
}

Renderer::~Renderer() { 
	delete logfile;
	output = NULL; 
	SAFE_RELEASE(primary_mcodes);
	SAFE_RELEASE(buffer);
	SAFE_RELEASE(shadow_buffer);
	SAFE_RELEASE(secondary_buffer);			// Warning .. Check this works or not.!
}

void Renderer::render(Method primary_ray_method, Method shadow_ray_method, Method secondary_ray_method, int maxdepth, int num_shadow_samples, int num_secondary_samples) {
	
	//DacrtRunTimeParameters rtparams;
	//rtparams.BUFFER_SIZE = 4 * 1024 * 1024; rtparams.MAX_SEGMENTS = 9000; rtparams.MAX_SEGMENT_THRESHOLD = 5; rtparams.NUM_RAYS_PER_BLOCK = 256;
	//rtparams.PARALLEL_RAY_THRESHOLD = 256; rtparams.PARALLEL_TRI_THRESHOLD = 256; rtparams.TRI_SHARED_MEMORY_SPACE = 256;

	// allocate all data here
	int*	 h_pray_idx_array = NULL;
	int*	 d_pray_idx_array = NULL;
	int*     d_hitids = NULL;			
	float*   d_maxts  = NULL;
	int*	 h_hitids = NULL;
	float*	 h_maxts = NULL;

	if(data_format == "soa") {
		RayArray h_primary_rays;
		RayArray d_primary_rays;
		renderPrimaryRaysSoa(primary_ray_method, rtparams, h_primary_rays, d_primary_rays, &h_pray_idx_array, &d_pray_idx_array, &h_hitids, &h_maxts, &d_hitids, &d_maxts);
		// create shadow rays and secondary rays from the hit points and material types
		if(enable_shadows) {
			renderShadowRaySoa(shadow_ray_method, rtparams, h_primary_rays, d_primary_rays, h_pray_idx_array, d_pray_idx_array, h_hitids, h_maxts, d_hitids, d_maxts);
		}
		if(enable_secondary) {
			renderSecondaryRaysSoa(secondary_ray_method, rtparams, h_primary_rays, d_primary_rays, h_pray_idx_array, d_pray_idx_array, h_hitids, h_maxts, d_hitids, d_maxts, max_depth, num_secondary_samples);
		}

		// RELEASE DATA
		SAFE_RELEASE(h_primary_rays.o);
		SAFE_RELEASE(h_primary_rays.d);
		CUDA_SAFE_RELEASE(d_primary_rays.o);
		CUDA_SAFE_RELEASE(d_primary_rays.d);

	} else if(data_format == "aos") {
		RayArrayAos h_primary_rays_aos;
		RayArrayAos d_primary_rays_aos;
		renderPrimaryRaysAos(primary_ray_method, rtparams, h_primary_rays_aos, d_primary_rays_aos, &h_pray_idx_array, &d_pray_idx_array, &h_hitids, &h_maxts);
		SAFE_RELEASE(h_primary_rays_aos.rays);
		CUDA_SAFE_RELEASE(d_primary_rays_aos.rays);
	}
	
	// draw the output
	writeToJpeg(scene->width, scene->height, buffer, output);
	
	// cleanup of other data independent data!! (pun.!! :P :P)
	SAFE_RELEASE(h_pray_idx_array);
	SAFE_RELEASE(h_maxts);
	SAFE_RELEASE(h_hitids);
	CUDA_SAFE_RELEASE(d_pray_idx_array);
	CUDA_SAFE_RELEASE(d_hitids);
	CUDA_SAFE_RELEASE(d_maxts);
}

/// NOTE: I still have no idea of how I am going to do the gpu part. So I guess I will fix with this architecture and use compilation flags?
///		  or possible have a new method.
void Renderer::renderPrimaryRaysSoa(Method method, DacrtRunTimeParameters& rtparams, RayArray& h_rays, RayArray& d_rays, int** h_ray_idx_array, int** d_ray_idx_array, int** h_hitids, float** h_maxts, int** d_hitids, float** d_maxts) {

	// we can generate rays
	double start, end;
	int num_primary_rays = scene->width * scene->height * scene->spp;
	
	printf("Generating Rays\n");
	start = omp_get_wtime();

	h_rays.d			= new float3[num_primary_rays];
	h_rays.o			= new float3[num_primary_rays];
	*h_ray_idx_array	= new int[num_primary_rays];
	*h_hitids			= new int[num_primary_rays];
	*h_maxts			= new float[num_primary_rays];
	
	checkCuda(cudaMalloc((void**)&d_rays.d, sizeof(float3) * num_primary_rays));
	checkCuda(cudaMalloc((void**)&d_rays.o, sizeof(float3) * num_primary_rays));
	checkCuda(cudaMalloc((void**)&(*d_ray_idx_array), sizeof(int) * num_primary_rays));
	checkCuda(cudaMalloc((void**)&(*d_hitids), sizeof(int) * num_primary_rays));
	checkCuda(cudaMalloc((void**)&(*d_maxts), sizeof(float) * num_primary_rays));
	
	if(use_morton_codes) {
		// check for 2d, and 5d morton codes
		primary_mcodes = new MortonCode[num_primary_rays];
#pragma omp parallel for
		for(int y = 0; y < scene->height; y++) {
			for(int x = 0; x < scene->width; x++) {
				int idx = x + y * scene->width;
				primary_mcodes[idx] = MortonCode(MortonCode2(x, y), x, y);
			}
		}
		
		// sort the codes
		std::sort(primary_mcodes, primary_mcodes + num_primary_rays, MortonCompare());

		// generate the rays based on the sorted values
#pragma omp parallel for
		for(int i = 0; i < num_primary_rays; i++) {
			h_rays.o[i] = scene->eye;
			h_rays.d[i] = normalize(scene->ef->genRayDir(primary_mcodes[i].x, primary_mcodes[i].y));
			(*h_ray_idx_array)[i] = i;
			(*h_hitids)[i] = -1;
			(*h_maxts)[i] = FLT_MAX;
		}
	} else {
#pragma omp parallel for
		for(int y = 0; y < scene->height; y++) {
			for(int x = 0; x < scene->width; x++) {
				int idx = y * scene->width + x;
				h_rays.o[idx] = scene->eye;
				h_rays.d[idx] = normalize(scene->ef->genRayDir(x, y));
				(*h_ray_idx_array)[idx] = idx;
				(*h_hitids)[idx] = -1;
				(*h_maxts)[idx] = FLT_MAX;
			}
		}
	}

	checkCuda(cudaMemcpy(d_rays.o, h_rays.o, sizeof(float3) * num_primary_rays, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_rays.d, h_rays.d, sizeof(float3) * num_primary_rays, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(*d_ray_idx_array, *h_ray_idx_array, sizeof(int) * num_primary_rays, cudaMemcpyHostToDevice));

	//calculate total device bytes
	total_device_bytes += 2 * sizeof(float3) * num_primary_rays + sizeof(int) * num_primary_rays;

	end = omp_get_wtime();
	printf("Ray Generation Time : %f seconds\n", end-start);
	printf("Total Device Memory usage so far : %u MB\n", total_device_bytes/(1024 * 1024));
	
	Counters primary_ray_ctr;

	// now we can do the actual rendering
	switch(method) {
	case CPU_OBJECT_PRIMARY:
		{
			printf("Starting CPU_OBJECT_PRIMARY\n");
			CpuPrimaryPassRenderData data(scene->scene_box, scene->h_triangles, scene->h_tri_idx_array, scene->h_aabbs, scene->num_triangles,
				h_rays, *h_ray_idx_array, num_primary_rays, *h_maxts, *h_hitids);
			start = omp_get_wtime();
			dacrt((void*)&data, CPU_OBJECT_PRIMARY, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("Primary Ray Pass - CPU_OBJECT_PRIMARY - method completed in : %f seconds \n", end-start);
		}
		break;
	case CPU_SPATIAL_PRIMARY:
		{
			printf("Starting CPU_SPATIAL_PRIMARY\n");
			CpuPrimaryPassRenderData data(scene->scene_box, scene->h_triangles, scene->h_tri_idx_array, scene->h_aabbs, scene->num_triangles,
				h_rays, *h_ray_idx_array, num_primary_rays, *h_maxts, *h_hitids);
			start = omp_get_wtime();
			dacrt((void*)&data, CPU_SPATIAL_PRIMARY, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - CPU_SPATIAL_PRIMARY - method completed in : %f seconds \n", end-start);
		}
		break;
	case CPU_GPU_OBJECT_PRIMARY:
		{
			printf("Starting CPU_GPU_OBJECT_PRIMARY\n");
			CpuGpuPrimaryPassRenderData data(scene->scene_box, scene->h_triangles, scene->d_triangles, scene->h_tri_idx_array, scene->h_aabbs,
				scene->num_triangles, h_rays, d_rays, *h_ray_idx_array, num_primary_rays, *h_maxts, *h_hitids);
			start = omp_get_wtime();
			dacrt((void*)&data, CPU_GPU_OBJECT_PRIMARY, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - CPU_GPU_OBJECT_PRIMARY - completed in : %f seconds \n", end-start);
		}
		break;
	case CPU_GPU_SPATIAL_PRIMARY:
		{
			printf("Starting CPU_GPU_SPATIAL_PRIMARY\n");
			CpuGpuPrimaryPassRenderData data(scene->scene_box, scene->h_triangles, scene->d_triangles, scene->h_tri_idx_array, scene->h_aabbs,
				scene->num_triangles, h_rays, d_rays, *h_ray_idx_array, num_primary_rays, *h_maxts, *h_hitids);
			start = omp_get_wtime();
			dacrt((void*)&data, CPU_GPU_SPATIAL_PRIMARY, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - CPU_GPU_SPATIAL_PRIMARY - completed in : %f seconds \n", end-start);
		}
		break;
	case CPU_GPU_SPATIAL_MODIFIED:
		{
			printf("Starting CPU_GPU_SPATIAL_MODIFIED\n");
			CpuGpuPrimaryPassRenderData data(scene->scene_box, scene->h_triangles, scene->d_triangles, scene->h_tri_idx_array, scene->h_aabbs,
				scene->num_triangles, h_rays, d_rays, *h_ray_idx_array, num_primary_rays, *h_maxts, *h_hitids);
			start = omp_get_wtime();
			dacrt((void*)&data, CPU_GPU_SPATIAL_MODIFIED, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - CPU_GPU_SPATIAL_MODIFIED - completed in : %f seconds \n", end-start);
		}
		break;
	case GPU_SPATIAL_PRIMARY:
		{
			/// NOTE: one way to get around cluttered code will be to extern this entire block in the renderer.cu
			///		  We will have to do that incase this way of coding does not work.!!
			printf("Starting GPU_SPATIAL_PRIMARY\n");
			start = omp_get_wtime();
			renderCallGpuSpatialPrimary(scene, d_rays, d_ray_idx_array, *h_maxts, *h_hitids, num_primary_rays, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - GPU_SPATIAL_PRIMARY - completed in : %f seconds \n", end-start);
		}
		break;
	case GPU_SPATIAL_PRIMARY_SEGMENTED:
		{
			/// NOTE: pass host pointers for maxts and hitids as device pointers (just the name convention and not actually device)
			/// VERY VERY IMPORTANT TO READ ALL COMMENTS in the same section in all other files also so that you dont get confused.
			printf("Starting GPU_SPATIAL_PRIMARY_SEGMENTED\n");
			GpuPrimaryPassRenderData data(scene->scene_box, scene->d_triangles, scene->d_tri_idx_array, scene->num_triangles, d_rays,
				*d_ray_idx_array, num_primary_rays, *h_maxts, *h_hitids);
			start = omp_get_wtime();
			dacrt((void*)&data, GPU_SPATIAL_PRIMARY_SEGMENTED, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - GPU_SPATIAL_PRIMARY_SEGMENTED - completed in : %f seconds\n", end-start);

		}
		break;
	case GPU_SPATIAL_CELL:
		{
			printf("Starting GPU_SPATIAL_CELL\n");
			start = omp_get_wtime();
			GpuCellData data(scene->scene_box, scene->d_triangles, d_rays, scene->d_tri_idx_array, *d_ray_idx_array, scene->num_triangles,
				num_primary_rays, scene->num_triangles, num_primary_rays, *h_maxts, *h_hitids);
			dacrt((void*)&data, GPU_SPATIAL_CELL, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - GPU_SPATIAL_CELL - completed in : %f seconds\n", end-start);
		}
		break;
	case CPU_GPU_TWIN_TREES:
		{
			printf("Starting CPU_GPU_TWIN_TREES\n");
			start = omp_get_wtime();
			CpuGpuTwinData data(scene->scene_box, scene->d_triangles, scene->h_triangles, d_rays, h_rays, scene->d_tri_idx_array, *d_ray_idx_array,
				scene->h_tri_idx_array, *h_ray_idx_array, *h_maxts, *h_hitids, scene->num_triangles, num_primary_rays, scene->num_triangles, num_primary_rays);
			dacrt((void*)&data, CPU_GPU_TWIN_TREES, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - CPU_GPU_TWIN_TREES - completed in : %f seconds\n", end-start);
		}
		break;
	case GPU_SPATIAL_FULLY_PARALLEL:
		{
			printf("Starting GPU_SPATIAL_FULLY_PARALLEL\n");
			start = omp_get_wtime();
			GpuFullyParallelData data(scene->scene_box, scene->d_triangles, d_rays, scene->d_tri_idx_array, *d_ray_idx_array, scene->num_triangles,
				num_primary_rays, scene->num_triangles, num_primary_rays, *h_maxts, *h_hitids);
			dacrt((void*)&data, GPU_SPATIAL_FULLY_PARALLEL, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - GPU_SPATIAL_FULLY_PARALLEL - completed in : %f seconds\n", end-start);
		}
		break;
	case GPU_SPATIAL_FULLY_PARALLEL_MODIFIED:
		{
			printf("Starting GPU_SPATIAL_FULLY_PARALLEL_MODIFIED\n");
			start = omp_get_wtime();
			
			GpuFullyParallelData data(scene->scene_box, scene->d_triangles, d_rays, scene->d_tri_idx_array, *d_ray_idx_array, scene->num_triangles,
				num_primary_rays, scene->num_triangles, num_primary_rays, *d_maxts, *d_hitids);
			dacrt((void*)&data, GPU_SPATIAL_FULLY_PARALLEL_MODIFIED, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - GPU_SPATIAL_FULLY_PARALLEL_MODIFIED - completed in : %f seconds\n", end-start);
		}
		break;
	case GPU_DACRT_FULLY_CUDA:
		{
			printf("Starting GPU_DACRT_FULLY_CUDA\n");
			start = omp_get_wtime();
			
			GpuFullyParallelData data(scene->scene_box, scene->d_triangles, d_rays, scene->d_tri_idx_array, *d_ray_idx_array, scene->num_triangles,
				num_primary_rays, scene->num_triangles, num_primary_rays, *d_maxts, *d_hitids);
			dacrt((void*)&data, GPU_DACRT_FULLY_CUDA, rtparams, primary_ray_ctr, *logfile);
			
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - GPU_DACRT_FULLY_CUDA - completed in : %f seconds\n", end-start);
		}
		break;
	}

	printStats(primary_ray_ctr);
	
	checkCuda(cudaMemcpy(*h_maxts, *d_maxts, sizeof(float) * num_primary_rays, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(*h_hitids, *d_hitids, sizeof(int) * num_primary_rays, cudaMemcpyDeviceToHost));

	//TODO : Change the way we do this here.!! Looks ugly
	// compute primary ray shading here
	if(!enable_shadows) {
		PointLight pl = {make_float3(0.0f, 0.8f, 0.0f), point_light_pos};
		buffer = new float3[num_primary_rays];
		cpuShade(pl, h_rays, scene->width, scene->height, num_primary_rays, scene->h_triangles, scene->num_triangles, 
			*h_maxts, *h_hitids, buffer, NULL, 0, NULL, NULL, primary_mcodes, false, use_morton_codes);
	}
}

void Renderer::printStats(Counters& ctr) {

	// Print the statistics
	printf("Timing and Stats for Frame No              : %d \n", frame_no);
	printf("------------------------------------------------\n");
	printf("Ray Filter Time                            : %f ms\n", ctr.ray_filter_time);
	printf("Tri Filter Time                            : %f ms\n", ctr.tri_filter_time);
	printf("Total Triangle Sorts                       : %d \n", ctr.trifilter_sort_cnt);
	printf("Total Ray Sorts                            : %d \n", ctr.rayfilter_sort_cnt);
	printf("Tri Sort By Key Time                       : %f ms\n", ctr.trisortbykey_time);
	printf("Tri Reduction Time                         : %f ms\n", ctr.trireduction_time);
	printf("Ray Sort By Key Time                       : %f ms\n", ctr.raysortbykey_time);
	printf("Ray Reduction Time                         : %f ms\n", ctr.rayreduction_time);
	printf("Brute Force Kernel Time                    : %f ms\n", ctr.brute_force_time);
	printf("Sort By Key Time                           : %f ms\n", ctr.seg_sort_time);
	printf("Reduce By Key Time                         : %f ms\n", ctr.reduction_time);
	printf("Update Min Time                            : %f ms\n", ctr.update_min_time);
	printf("Memcpy time                                : %f ms\n", ctr.mem_cpy_time);
	printf("Misc time                                  : %f ms\n", ctr.misc_time);
	printf("Other time 1                               : %f ms\n", ctr.other_time1);
	printf("Total Ray Box Intersections                : %d \n", ctr.raybox);
	printf("Total Tri Box Intersections                : %d \n", ctr.tribox);
	printf("Total Ray Tri Intersections                : %d \n", ctr.raytri);
	printf("------------------------------------------------\n");

	// print the detailed per sort operations in log file
	std::ofstream logfile("log.log");
	logfile<<"Tri Sort Data\n";
	for(unsigned int i = 0; i < ctr.tri_sort_times.size(); i++)
		logfile<<ctr.tri_sort_times[i].first<<"\t"<<ctr.tri_sort_times[i].second<<"\n";
	logfile<<"-------------------------------\nRay Sort Data\n";
	for(unsigned int i = 0; i < ctr.ray_sort_times.size(); i++)
		logfile<<ctr.ray_sort_times[i].first<<"\t"<<ctr.ray_sort_times[i].second<<"\n";
	logfile.close();
}

/// NOTE: I dont need the h_pray_idx_array at all!!!!!
void Renderer::renderShadowRaySoa(Method method, DacrtRunTimeParameters& rtparams, RayArray& h_primary_rays, RayArray& d_primary_rays, 
	int* h_pray_idx_array, int* d_pray_idx_array, int* h_hitids, float* h_maxts, int* d_hitids, float * d_maxts, int num_shadow_samples) {
	
	printf("\nShadow Ray Pass\n");
	double start, end;
	RayArray h_srays;
	RayArray d_srays;
	int* h_sray_idx_array  = NULL;
	int* d_sray_idx_array  = NULL;
	int* h_spray_idx_array = NULL;		// host shadow-primary ray association array
	int* d_spray_idx_array = NULL;		// host shadow_primary ray association array
	int* h_shitids		   = NULL;
	float* h_smaxts		   = NULL;
	int* d_shitids		   = NULL;
	float* d_smaxts		   = NULL;
	bool* h_shadows		   = NULL;
	bool* d_shadows		   = NULL;
	int num_candidates	   = 0;
	int num_primary_rays   = scene->width * scene->height * scene->spp;
	int num_shadow_rays	   = 0;
	Counters shadow_ray_ctr;

	/// NOTE: Assuming same point light for both shading and shadow calculations
	PointLight pl = {make_float3(0.0f, 0.0f, 0.0f), point_light_pos};

	if(!use_cuda) {
		start = omp_get_wtime();
		/// TODO: write code to use different methods for shadow rays. We have to collect timings.
		/// Right now, we are just computing using the spatial methods.
		std::vector<int> temp;
		for(int i = 0; i < num_primary_rays; i++) {
			if(h_hitids[i] != -1) {
				temp.push_back(i);
			}
		}
	
		num_candidates	   = temp.size();
		num_shadow_rays    = num_candidates;
		h_sray_idx_array   = new int[num_candidates * num_shadow_samples];
		h_spray_idx_array  = new int[num_candidates * num_shadow_samples];
		h_srays.d		   = new float3[num_candidates * num_shadow_samples];
		h_srays.o		   = new float3[num_candidates * num_shadow_samples];
		h_shadows		   = new bool[num_candidates * num_shadow_samples];			// we need only num_candidates to determine the shadows
		h_shitids		   = new int[num_candidates];
		h_smaxts		   = new float[num_candidates];

	
		checkCuda(cudaMalloc((void**)&d_sray_idx_array, sizeof(int) * num_candidates * num_shadow_samples));
		checkCuda(cudaMalloc((void**)&d_spray_idx_array, sizeof(int) * num_candidates * num_shadow_samples));
		checkCuda(cudaMalloc((void**)&d_srays.d, sizeof(float3) * num_candidates * num_shadow_samples));
		checkCuda(cudaMalloc((void**)&d_srays.o, sizeof(float3) * num_candidates * num_shadow_samples));
		checkCuda(cudaMalloc((void**)&d_shadows, sizeof(bool) * num_candidates * num_shadow_samples));
		checkCuda(cudaMalloc((void**)&d_shitids, sizeof(int) * num_candidates * num_shadow_samples));
		checkCuda(cudaMalloc((void**)&d_smaxts, sizeof(float) * num_candidates * num_shadow_samples));

		printf("\nGenerating Shadow Rays : %d\n", temp.size());
		/// NOTE: in case of soft shadows, we have to loop over still n_shadow_samples still.
		for(unsigned int i = 0; i < temp.size(); i++) {
			h_sray_idx_array[i] = i;
			h_spray_idx_array[i] = temp[i];
			h_shadows[i] = false;	
			float3 xt_pt = Ray(h_primary_rays.o[temp[i]], h_primary_rays.d[temp[i]])(h_maxts[temp[i]]);
			float3 dir = normalize(pl.position - xt_pt);			// incase of soft shadows, i have to make use of samples
			h_srays.o[i] = xt_pt + (0.00001F * dir);
			h_srays.d[i] = dir;
			h_shitids[i] = -1;
			h_smaxts[i] = FLT_MAX;
		}
		/*
		// generate one reflected ray bounce.
		for(unsigned int i = temp.size(); i < 2 * temp.size(); i++) {
			// we generate reflection rays. // use normal, then generate the opposite rays using snell's law
			h_sray_idx_array[i] = i;
			h_spray_idx_array[i] = temp[i-temp.size()];			//use the same association
			float3 xt_pt = Ray(h_primary_rays.o[temp[i-temp.size()]], h_primary_rays.d[temp[i-temp.size()]])(h_maxts[temp[i-temp.size()]]);
			// compute normal
			float3 normal = computeNormal(Triangle(scene->h_triangles.v0[h_hitids[temp[i]]], scene->h_triangles.v1[h_hitids[temp[i]]], scene->h_triangles.v2[h_hitids[temp[i]]]));
			float3 V = h_primary_rays.d[temp[i]];
			float fdot = -dot(normal, V);
			// reflected ray is given by V + (2 * N * fdot)
			float3 dir = normalize(V + (2 * normal * fdot));
			h_srays.o[i] = xt_pt + (0.00001F * dir);
			h_srays.d[i] = dir;
			h_shitids[i] = -1;
			h_smaxts[i] = FLT_MAX;
		}
		*/

		// copy to device
		checkCuda(cudaMemcpy(d_sray_idx_array, h_sray_idx_array, sizeof(int) * num_candidates * num_shadow_samples, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_spray_idx_array, h_spray_idx_array, sizeof(int) * num_candidates * num_shadow_samples, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_srays.o, h_srays.o, sizeof(float3) * num_candidates * num_shadow_samples, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_srays.d, h_srays.d, sizeof(float3) * num_candidates * num_shadow_samples, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_shadows, h_shadows, sizeof(bool) * num_candidates * num_shadow_samples, cudaMemcpyHostToDevice));

		end = omp_get_wtime();
		printf("Shadow Rays created in : %f seconds\n", end-start);
	} else {
		// use cuda device shadow ray initialization, and stuff
		setupShadowRaysCuda(d_primary_rays, d_srays, d_hitids, d_maxts, d_pray_idx_array, &d_sray_idx_array, &d_spray_idx_array, num_shadow_rays);
		// allocate memory
		checkCuda(cudaMalloc((void**)&d_smaxts, sizeof(float) * num_shadow_rays));
		checkCuda(cudaMalloc((void**)&d_shitids, sizeof(int) * num_shadow_rays));
		h_shitids		   = new int[num_shadow_rays];
		h_smaxts		   = new float[num_shadow_rays];
		h_shadows		   = new bool[num_shadow_rays];
		h_sray_idx_array   = new int[num_shadow_rays];
		h_spray_idx_array  = new int[num_shadow_rays];
	}
	// call the appropriate pass
	/// NOTE: the bug was a typo. I had been passing h_pray while it should have been h_spray.!! I should definitely change the naming now.!!
	switch(method) {
	case CPU_SPATIAL_SHADOW:
		{
			printf("Starting Shadow Pass\n");
			start = omp_get_wtime();
			CpuShadowPassData data(scene->scene_box, scene->h_triangles, scene->h_tri_idx_array, scene->num_triangles, scene->num_triangles,
				h_srays, h_sray_idx_array, h_spray_idx_array, num_candidates, num_candidates, h_shadows);
			dacrt((void*)&data, CPU_SPATIAL_SHADOW, rtparams, shadow_ray_ctr, *logfile);
			end = omp_get_wtime();
		}
		break;
	case GPU_DACRT_FULLY_CUDA_SHADOW:
		{
			printf("Starting GPU_DACRT_FULLY_CUDA_SHADOW \n");
			start = omp_get_wtime();
			GpuFullyParallelData data(scene->scene_box, scene->d_triangles, d_srays, scene->d_tri_idx_array, d_sray_idx_array, scene->num_triangles,
				num_shadow_rays, scene->num_triangles, num_shadow_rays, d_smaxts, d_shitids);
			dacrt((void**)&data, GPU_SPATIAL_FULLY_PARALLEL_MODIFIED, rtparams, shadow_ray_ctr, *logfile);
			// now fill the bool array correctly.
			// if we have -1 in the hitids, we have no intersection and hence no shadows. else shadow pixel
			end = omp_get_wtime();
			checkCuda(cudaMemcpy(h_shitids, d_shitids, sizeof(int) * num_shadow_rays, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(h_smaxts, d_smaxts, sizeof(float) * num_shadow_rays, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(h_spray_idx_array, d_spray_idx_array, sizeof(int) * num_shadow_rays, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(h_sray_idx_array, d_sray_idx_array, sizeof(int) * num_shadow_rays, cudaMemcpyDeviceToHost));
			unsigned int debug = 0;
			for(int i = 0; i < num_shadow_rays; i++) {
				if(h_shitids[i] != -1) {
					// check if distance to nearest hit is less than that of light
					/*
					float3 xt_pt = Ray(h_srays.o[i], h_srays.d[i])(h_smaxts[i]);
					float3 orgn = h_srays.o[i];
					float light_distance = getDistance(xt_pt, orgn);
					float nearest_distance = getDistance(orgn, pl.position);
					if(nearest_distance < light_distance) {*/
						h_shadows[i] = true;
						debug++;
					//}
				}
			}
			printf("GPU_DACRT_FULLY_CUDA_SHADOW completed in : %f seconds\n", end-start);
			printf("DEBUG : %d of shadow rays : %d\n", debug, num_shadow_rays);
			
		}
		break;
	}

	printStats(shadow_ray_ctr);

	// update the buffer accordingly now.!
	/// TODO: make the update of the buffer totally independent of the shadow pass.!
	buffer = new float3[num_primary_rays];
	if(enable_shadows) {
		cpuShade(pl, h_primary_rays, scene->width, scene->height, num_primary_rays, scene->h_triangles, scene->num_triangles, h_maxts, h_hitids, buffer, h_shadows, num_candidates,
			h_sray_idx_array, h_spray_idx_array, primary_mcodes, enable_shadows, use_morton_codes);
	}

	// free up the memory. We can safely do this because this data is not used anywhere else.
	SAFE_RELEASE(h_shitids);
	SAFE_RELEASE(h_smaxts);
	SAFE_RELEASE(h_sray_idx_array);
	SAFE_RELEASE(h_spray_idx_array);
	SAFE_RELEASE(h_shadows);
	SAFE_RELEASE(h_srays.d);
	SAFE_RELEASE(h_srays.o);
	CUDA_SAFE_RELEASE(d_sray_idx_array);
	CUDA_SAFE_RELEASE(d_spray_idx_array);
	CUDA_SAFE_RELEASE(d_srays.d);
	CUDA_SAFE_RELEASE(d_srays.o);
	CUDA_SAFE_RELEASE(d_shadows);
	CUDA_SAFE_RELEASE(d_shitids);
	CUDA_SAFE_RELEASE(d_smaxts);
	
}

void Renderer::renderSecondaryRaysSoa(Method method, DacrtRunTimeParameters& rtparams, RayArray& h_primary_rays, RayArray& d_primary_rays, 
	int* h_pray_idx_array, int* d_pray_idx_array, int* h_hitids, float* h_maxts, int* d_hitids, float * d_maxts, int max_depth, int samples) {

	// we do cuda ray generation 
	// first determine number of secondary rays that are to be generated.
	size_t num_candidates     = 0;
	size_t num_primary_rays   = scene->width * scene->height * scene->spp;
	size_t num_secondary_rays = 0;
	int* h_sray_idx_array     = NULL;
	int* d_sray_idx_array     = NULL;
	int* h_spray_idx_array    = NULL;		// host shadow-primary ray association array
	int* d_spray_idx_array    = NULL;		// host shadow_primary ray association array
	int* h_shitids            = NULL;
	float* h_smaxts           = NULL;
	int* d_shitids            = NULL;
	float* d_smaxts           = NULL;
	RayArray h_srays;
	RayArray d_srays;
	PointLight pl = {make_float3(0.0f, 0.0f, 0.0f), point_light_pos};
	std::vector<float> aoctr;					// we store hits for all ambient occluded rays here.!
		
	// we generate ambient occlusion for the rays 
	// 32 ambient occulsion rays per pixel
	printf("Secondary Ray Pass\n");
	double start, end;
	if(!use_cuda) {

		start = omp_get_wtime();
		std::vector<int> temp;
		for(size_t i = 0; i < num_primary_rays; i++) {
			if(h_hitids[i] != -1) {
				temp.push_back(i);
			}
		}
	
		num_candidates     = temp.size();
		num_secondary_rays = num_candidates * samples;
		h_sray_idx_array   = new int[num_secondary_rays];
		h_spray_idx_array  = new int[num_secondary_rays];
		h_srays.d          = new float3[num_secondary_rays];
		h_srays.o          = new float3[num_secondary_rays];
		h_shitids          = new int[num_secondary_rays];
		h_smaxts           = new float[num_secondary_rays];

		// allocate device memory
		checkCuda(cudaMalloc((void**)&d_sray_idx_array, sizeof(int) * num_secondary_rays));
		checkCuda(cudaMalloc((void**)&d_spray_idx_array, sizeof(int) * num_secondary_rays));
		checkCuda(cudaMalloc((void**)&d_srays.d, sizeof(float3) * num_secondary_rays));
		checkCuda(cudaMalloc((void**)&d_srays.o, sizeof(float3) * num_secondary_rays));
		checkCuda(cudaMalloc((void**)&d_shitids, sizeof(int) * num_secondary_rays));
		checkCuda(cudaMalloc((void**)&d_smaxts, sizeof(float) * num_secondary_rays));

		printf("Generating Secondary Rays : Num Samples per pixel : %d ; Total Rays : %d\n", samples, temp.size() * samples);
		for(size_t i = 0; i < temp.size(); i++) {
			float3 xt_pt = Ray(h_primary_rays.o[temp[i]], h_primary_rays.d[temp[i]])(h_maxts[temp[i]]);
			float3 normal = computeNormal(Triangle(scene->h_triangles.v0[h_hitids[temp[i]]], scene->h_triangles.v1[h_hitids[temp[i]]], scene->h_triangles.v2[h_hitids[temp[i]]]));
			if(dot(normal, h_primary_rays.d[temp[i]]) > 0.f)
				normal = -normal;			// backward facing triangles.
			/*
			float3 na = fabs(normal);
			float nm = fmaxf(fmaxf(na.x, na.y), na.z);
			float3 perp = make_float3(normal.y, -normal.x, 0);
			if (nm == na.z)
				perp = make_float3(0.0f, normal.z, -normal.y);
			else if (nm == na.x)
				perp = make_float3(-normal.z, 0.0f, normal.x);

			perp = normalize(perp);
			float3 biperp = cross(normal, perp);

			// assume always a multiple of 4 random directions
			//size_t num_per_quadrant = samples/4;
			for(size_t j = 0; j < 4; j++) {
				float u1 = unifRand();
				float v = unifRand();
				float u = M_PI/2 * (u1 + j);
				float v2 = v * v;
				float3 t1 = perp * cos(u) * v;
				float3 t2 = biperp * sin(u) * v;
				v = sqrtf(1-v2);
				float3 norm = normal * v;
				float3 dir = t1 + t2 + norm;
				dir = normalize(dir);
				h_srays.o[i * samples + j] = xt_pt + (0.00001F * dir);
				h_srays.d[i * samples + j] = dir;
				h_shitids[i * samples + j] = -1;
				h_smaxts[i * samples + j]  = FLT_MAX;
			}*/



			
			for(size_t j = 0; j < samples; j++) {
				h_sray_idx_array[i * samples + j]  = i * samples + j;
				h_spray_idx_array[i * samples + j] = temp[i];
				// compute perpendicular vectors
				// generate a random direction here.!
				float3 dir;
				//do {
				//	dir.x = unifRand();
				//	dir.y = unifRand();
				//	dir.z = unifRand();
				//	dir = normalize(dir);
				//} while(dot(normal, dir) < 0.0f);
				float u1 = unifRand();
				float u2 = unifRand();
				float r = sqrtf(u1);
				float theta = 2 * M_PI * u2;
				float x = r * cos(theta);
				float y = r * sin(theta);
				dir = normalize(make_float3(x, y, sqrtf(max(0.0f, 1-u1))));
				if(dot(dir, normal) < 0.0f) dir = -dir;
				h_srays.o[i * samples + j] = xt_pt + (0.00001F * dir);
				h_srays.d[i * samples + j] = dir;
				h_shitids[i * samples + j] = -1;
				h_smaxts[i * samples + j]  = FLT_MAX;
				
			}
		}

		checkCuda(cudaMemcpy(d_sray_idx_array, h_sray_idx_array, sizeof(int) * num_secondary_rays, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_spray_idx_array, h_spray_idx_array, sizeof(int) * num_secondary_rays, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_srays.o, h_srays.o, sizeof(float3) * num_secondary_rays, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(d_srays.d, h_srays.d, sizeof(float3) * num_secondary_rays, cudaMemcpyHostToDevice));
		

		end = omp_get_wtime();
		end = omp_get_wtime();
		printf("Secondary Rays generated in : %f seconds\n", end-start);
	} else {
		printf("Cuda Setup not yet complete\n");
		return;
	}
	Counters secondary_ctr;
	switch(method) {
	case CPU_SPATIAL_PRIMARY:
		{
			printf("Starting AO in CPU_SPATIAL_PRIMARY");
			start = omp_get_wtime();
			CpuPrimaryPassRenderData data(scene->scene_box, scene->h_triangles, scene->h_tri_idx_array, scene->h_aabbs, scene->num_triangles,
				h_srays, h_sray_idx_array, num_secondary_rays, h_smaxts, h_shitids);
			dacrt((void*)&data, CPU_SPATIAL_PRIMARY, rtparams, secondary_ctr, *logfile);
			end = omp_get_wtime();
			
			printf("AO Pass in CPU_SPATIAL_PRIMARY completed in : %f seconds", end-start);
		}
		break;
	case GPU_DACRT_FULLY_CUDA_SECONDARY:
		{
			printf("Starting GPU_DACRT_FULLY_CUDA_SECONDARY \n");
			start = omp_get_wtime();
			GpuFullyParallelData data(scene->scene_box, scene->d_triangles, d_srays, scene->d_tri_idx_array, d_sray_idx_array, scene->num_triangles,
				num_secondary_rays, scene->num_triangles, num_secondary_rays, d_smaxts, d_shitids);
			dacrt((void**)&data, GPU_SPATIAL_FULLY_PARALLEL_MODIFIED, rtparams, secondary_ctr, *logfile);
			end = omp_get_wtime();
			printf("GPU_DACRT_FULLY_CUDA_SECONDARY completed in : %f seconds\n", end-start);
			checkCuda(cudaMemcpy(h_shitids, d_shitids, sizeof(int) * num_secondary_rays, cudaMemcpyDeviceToHost));
			checkCuda(cudaMemcpy(h_smaxts, d_smaxts, sizeof(float) * num_secondary_rays, cudaMemcpyDeviceToHost));
		}
		break;
	}
	printStats(secondary_ctr);
	
	// shade ambient occlusion here.!
	for(size_t i = 0; i < num_candidates; i++) {
		int cnt = 0;
		for(size_t j = 0; j < samples; j++) {
			if(h_shitids[i*samples + j] != -1) cnt++;				// distance factor
		}
		aoctr.push_back((float)cnt/(float)samples);
	}

	float* aocontrib = new float[aoctr.size()];
	std::copy(aoctr.begin(), aoctr.end(), aocontrib);
	// we write our own custom ambient occlusion writing method
	float3* aobuffer = new float3[num_primary_rays];
	cpuShadeAmbientOcclusion(scene->width, scene->height, num_primary_rays, h_hitids, aobuffer, num_candidates, samples, aocontrib,
		h_sray_idx_array, h_spray_idx_array);

	writeToJpeg(scene->width, scene->height, aobuffer, ao_output);
	
	// safe release
	SAFE_RELEASE(aobuffer);
	SAFE_RELEASE(aocontrib);
	SAFE_RELEASE(h_shitids);
	SAFE_RELEASE(h_smaxts);
	SAFE_RELEASE(h_sray_idx_array);
	SAFE_RELEASE(h_srays.o);
	SAFE_RELEASE(h_srays.d);
	SAFE_RELEASE(h_srays.original_id_array);
	CUDA_SAFE_RELEASE(d_shitids);
	CUDA_SAFE_RELEASE(d_smaxts);
	CUDA_SAFE_RELEASE(d_sray_idx_array);
	CUDA_SAFE_RELEASE(d_srays.o);
	CUDA_SAFE_RELEASE(d_srays.d);
	CUDA_SAFE_RELEASE(d_srays.original_id_array);
	
}

// AoS data format methods
void Renderer::renderPrimaryRaysAos(Method method, DacrtRunTimeParameters& rtparams, RayArrayAos& h_rays, RayArrayAos& d_rays, int** h_ray_idx_array, int** d_ray_idx_array, int** h_hitids, float** h_maxts) {

	// we can generate rays
	double start, end;
	int num_primary_rays = scene->width * scene->height * scene->spp;
	
	printf("Generating Rays\n");
	start = omp_get_wtime();

	h_rays.rays			= new Ray4[num_primary_rays];
	*h_ray_idx_array	= new int[num_primary_rays];
	*h_hitids			= new int[num_primary_rays];
	*h_maxts			= new float[num_primary_rays];

	checkCuda(cudaMalloc((void**)&d_rays.rays, sizeof(Ray4) * num_primary_rays));
	checkCuda(cudaMalloc((void**)&(*d_ray_idx_array), sizeof(int) * num_primary_rays));

	if(use_morton_codes) {
		// check for 2d, and 5d morton codes
		primary_mcodes = new MortonCode[num_primary_rays];
#pragma omp parallel for
		for(int y = 0; y < scene->height; y++) {
			for(int x = 0; x < scene->width; x++) {
				int idx = x + y * scene->width;
				primary_mcodes[idx] = MortonCode(MortonCode2(x, y), x, y);
			}
		}
		
		// sort the codes
		std::sort(primary_mcodes, primary_mcodes + num_primary_rays, MortonCompare());

		// generate the rays based on the sorted values
#pragma omp parallel for
		for(int i = 0; i < num_primary_rays; i++) {
			h_rays.rays[i] = Ray4(make_float4(scene->eye, 0.0f), make_float4(normalize(scene->ef->genRayDir(primary_mcodes[i].x, primary_mcodes[i].y)), 0.0f));
			(*h_ray_idx_array)[i] = i;
			(*h_hitids)[i] = -1;
			(*h_maxts)[i] = FLT_MAX;
		}
	} else {
#pragma omp parallel for
		for(int y = 0; y < scene->height; y++) {
			for(int x = 0; x < scene->width; x++) {
				int idx = y * scene->width + x;
				h_rays.rays[idx] = Ray4(make_float4(scene->eye, 0.0f), make_float4(normalize(scene->ef->genRayDir(x, y)), 0.0f));
				(*h_ray_idx_array)[idx] = idx;
				(*h_hitids)[idx] = -1;
				(*h_maxts)[idx] = FLT_MAX;
			}
		}
	}

	// copy data to gpu
	checkCuda(cudaMemcpy(d_rays.rays, h_rays.rays, sizeof(Ray4) * num_primary_rays, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(*d_ray_idx_array, *h_ray_idx_array, sizeof(int) * num_primary_rays, cudaMemcpyHostToDevice));

	//calculate total device bytes
	/// NOTE: this is correct calculation..!! Don't change.!
	total_device_bytes += 2 * sizeof(float4) * num_primary_rays + sizeof(int) * num_primary_rays;
	
	end = omp_get_wtime();
	printf("Ray Generation Time : %f seconds\n", end-start);
	printf("Total Device Memory usage so far : %u MB\n", total_device_bytes/(1024 * 1024));
	Counters primary_ray_ctr;

	// do actual rendering
	switch(method) {
	case GPU_SPATIAL_PRIMARY_AOS:
		{
			printf("Starting GPU_SPATIAL_PRIMARY_AOS\n");
			start = omp_get_wtime();
			renderCallGpuSpatialPrimaryAos(scene, d_rays, d_ray_idx_array, *h_maxts, *h_hitids, num_primary_rays, num_primary_rays, rtparams, primary_ray_ctr, *logfile);
			end = omp_get_wtime();
			printf("\nPrimary Ray Pass - GPU_SPATIAL_PRIMARY - completed in : %f seconds \n", end-start);
		}
		break;
	}

	printStats(primary_ray_ctr);
	if(!enable_shadows) {
		PointLight pl = {make_float3(0, 0.8, 0.0), make_float3(0.8f, 0.8f, 0.8f)};
		buffer = new float3[num_primary_rays];
		cpuShadeAos(pl, h_rays, scene->width, scene->height, num_primary_rays, scene->h_triangles_aos, scene->num_triangles, 
			*h_maxts, *h_hitids, buffer, NULL, 0, NULL, NULL, primary_mcodes, false, use_morton_codes);
		
	}


}


