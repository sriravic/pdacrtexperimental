#include <util/scene.h>
#include <util/util.h>
#include <util/cutimer.h>

extern unsigned int total_device_bytes;

Scene::Scene() {
	printf("Default Scene Constructor is called\n");
	camera_params_set = false;			// not yet set
}
Scene::Scene(TriangleArray& triangles, int _num_triangles, std::string& data_fmt, bool _morton_tris) {
	printf("Scene Construction Beginning\n");
	data_format = data_fmt;
	camera_params_set = false;			// not yet set
	num_triangles		= _num_triangles;
	
	// host memory
	h_triangles			= triangles;
	h_tri_idx_array		= new int[num_triangles];
	h_aabbs.bmax		= new float3[num_triangles];
	h_aabbs.bmin		= new float3[num_triangles];
	h_aabbs.centroid	= new float3[num_triangles];
	// set all other unnecessary pointers to NULL
	d_tri_centroids		= NULL;
	morton_tris			= _morton_tris;
	h_materials			= NULL;
	d_materials			= NULL;
	h_mat_idx			= NULL;
	d_mat_idx			= NULL;
		
	// allocate memory on the gpu
	checkCuda(cudaMalloc((void**)&d_triangles.v0, sizeof(float3) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_triangles.v1, sizeof(float3) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_triangles.v2, sizeof(float3) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_tri_idx_array, sizeof(int) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_aabbs.bmax, sizeof(float3) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_aabbs.bmin, sizeof(float3) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_aabbs.centroid, sizeof(float3) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_morton_codes, sizeof(unsigned int) * num_triangles));

	total_device_bytes += 6 * sizeof(float3) * num_triangles + sizeof(int) * num_triangles;
	printf("Model Data size : %f MB\n", (float)total_device_bytes/(float)(1024 * 1024));
	
	camera				= NULL;
	ef					= NULL;
	film				= NULL;
}

Scene::Scene(TriangleArrayAos& triangles, int _num_triangles, std::string& data_fmt, bool _morton_tris) {
	printf("Scene Construction Beginning\n");
	data_format			= data_fmt;
	camera_params_set	= false;
	num_triangles		= _num_triangles;
	h_triangles_aos		= triangles;
	h_tri_idx_array		= new int[num_triangles];
	h_aabbs_aos.bboxes	= new AABB4[num_triangles];
	morton_tris			= _morton_tris;
	d_morton_codes		= NULL;
	h_materials			= NULL;
	d_materials			= NULL;
	h_mat_idx			= NULL;
	d_mat_idx			= NULL;
	
	// alloc device memory
	checkCuda(cudaMalloc((void**)&d_triangles_aos.triangles, sizeof(Triangle3) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_tri_idx_array, sizeof(int) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_aabbs_aos.bboxes, sizeof(AABB4) * num_triangles));
	checkCuda(cudaMalloc((void**)&d_morton_codes, sizeof(unsigned int) * num_triangles));

	total_device_bytes += sizeof(Triangle3) * num_triangles + sizeof(int) * num_triangles;
	printf("Model Data size : %f MB\n", (float)total_device_bytes/(float)(1024 * 1024));
	camera = NULL;
	ef	   = NULL;
	film   = NULL;
}


Scene::~Scene() {
	// host triangle hTriangles was passed as a reference. We'd allocated memory in the mesh class. We'll deallocate it there. not here.!!
	// BUG:!!
	//SAFE_RELEASE(h_triangles_aos.triangles);
	SAFE_RELEASE(h_tri_idx_array);
	SAFE_RELEASE(h_aabbs.bmax);
	SAFE_RELEASE(h_aabbs.bmin);
	SAFE_RELEASE(h_aabbs.centroid);
	SAFE_RELEASE(h_aabbs_aos.bboxes);
	SAFE_RELEASE(h_mat_idx);
	SAFE_RELEASE(h_materials);

	if(camera != NULL) delete camera;
	if(ef != NULL) delete ef;
	if(film != NULL) delete film;

	CUDA_SAFE_RELEASE(d_triangles.v0);
	CUDA_SAFE_RELEASE(d_triangles.v1);
	CUDA_SAFE_RELEASE(d_triangles.v2);
	CUDA_SAFE_RELEASE(d_triangles_aos.triangles);
	CUDA_SAFE_RELEASE(d_tri_idx_array);
	CUDA_SAFE_RELEASE(d_aabbs.bmax);
	CUDA_SAFE_RELEASE(d_aabbs.bmin);
	CUDA_SAFE_RELEASE(d_aabbs.centroid);
	CUDA_SAFE_RELEASE(d_aabbs_aos.bboxes);
	CUDA_SAFE_RELEASE(d_tri_centroids);
	CUDA_SAFE_RELEASE(d_morton_codes);
	CUDA_SAFE_RELEASE(d_mat_idx);
	CUDA_SAFE_RELEASE(d_materials);
}

void Scene::setCameraParams(const float3& _eye, int _width, int _height, int _spp) {
	// lookat params are alone set
	eye = _eye;
	width = _width;
	height = _height;
	spp = _spp;
	up = make_float3(0.0f, -1.0f, 0.0f);			
	camera_params_set = true;
}

void Scene::setCameraParams(int _width, int _height, int _spp) {
	// eye, lookat are computed per scene
	width = _width;
	height = _height;
	spp = _spp;
	up = make_float3(0.0f, -1.0f, 0.0f);
	camera_params_set = true;
}

void Scene::setCameraParams(const float3& _eye, const float3& _lookat, const float3& _up, int _width, int _height, int _spp) {
	eye = _eye;
	lookat = _lookat;
	up = _up;
	width = _width;
	height = _height;
	spp = _spp;
	camera_params_set = true;
}

void Scene::setCameraParams(const float3& _eye, const float3& _lookat, int _width, int _height, int _spp) {
	eye = _eye;
	lookat = _lookat;
	width = _width;
	height = _height;
	spp = _spp;
	up = make_float3(0.0f, -1.0f, 0.0f);
	camera_params_set = true;	
}

// this method gets calls the appropriate data path
void Scene::initScene(bool use_cuda) {
	if(data_format == "soa") {
		initSceneSoa(use_cuda);
	} else if(data_format == "aos") {
		initSceneAos(use_cuda);
	} else {
		std::cerr<<"SEVERE : Invalid method called!!\nExiting\n";
		exit(-1);
	}
}

// the flag indicates if we want to use cuda to compute the ids, bounding boxes, scene bounding boxes, etc.
// this will be used when we do a full gpu build
void Scene::initSceneSoa(bool use_cuda) {

	if(!use_cuda) {

		// computes all the triangle bounding boxes and ids and then scene bounding box
		double scene_start, scene_end, start, end;
		scene_start = omp_get_wtime();

		// reorder the triangles by their centroids and z curve
		printf("Computing Bounding Boxes for Triangles\n");
		start = omp_get_wtime();
#pragma omp parallel for
		for(int i = 0; i < num_triangles; i++) {
			h_tri_idx_array[i] = i;
			h_aabbs.bmin[i] = fminf(h_triangles.v0[i], fminf(h_triangles.v1[i], h_triangles.v2[i]));
			h_aabbs.bmax[i] = fmaxf(h_triangles.v0[i], fmaxf(h_triangles.v1[i], h_triangles.v2[i]));
			h_aabbs.centroid[i] = (h_aabbs.bmin[i] + h_aabbs.bmax[i]) * 0.5f;
		}
		end = omp_get_wtime();
		printf("Bounding Boxes computed for %d triangles in : %f seconds\n", num_triangles, end-start);

		start = omp_get_wtime();
		for(int i = 0; i < num_triangles; i++) {
			scene_box.bmin = fminf(scene_box.bmin, h_aabbs.bmin[i]);
			scene_box.bmax = fmaxf(scene_box.bmax, h_aabbs.bmax[i]);
		}	
		end = omp_get_wtime();
		printf("Scene Box Computed in : %f seconds\n\n", end-start);

		// update the scene box to be about 10% bigger
		float3 extents = scene_box.bmax - scene_box.bmin;
		scene_box.bmin -= extents * 0.1f;
		scene_box.bmax += extents * 0.1f;

		// check if all the camera params have been set
		if(!camera_params_set) {
			printf("WARNING !! - Camera Params not set... Setting Default Values.\n");
			width = 800;
			height = 600;
			up = make_float3(0.0f, 1.0f, 0.0f);
			eye = getEye(scene_box);
			lookat = scene_box.centroid();
		}

		film = new Film(width, height);
		camera = new Camera(eye, lookat, up);
		ef = new EyeFrustum(*camera, *film);
	
		printf("Transfering scene data over to the GPU\n");
		Timer tri_transfer_timer("Triangle Data Transfer Timer");
		tri_transfer_timer.start();
		cudaMemcpy(d_triangles.v0, h_triangles.v0, sizeof(float3) * num_triangles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_triangles.v1, h_triangles.v1, sizeof(float3) * num_triangles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_triangles.v2, h_triangles.v2, sizeof(float3) * num_triangles, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tri_idx_array, h_tri_idx_array, sizeof(int) * num_triangles, cudaMemcpyHostToDevice);
		tri_transfer_timer.stop();
		tri_transfer_timer.print();

		/// NOTE: TODO:!!
		// use cpu kernel kernel to do re arrange the data in a morton code manner.?
	
		scene_end = omp_get_wtime();
		printf("Scene Initialization with SoA data format completed in : %f \n", scene_end - scene_start);
	} else {
		// call gpu scene initialization method
		initSceneAosCuda();
	}
}

// this method will use AoS data format
void Scene::initSceneAos(bool use_cuda) {
	double start, end;
	double scene_start, scene_end;
	scene_start = omp_get_wtime();

	printf("Computing Bounding boxes for the triangles\n");
	start = omp_get_wtime();
#pragma omp parallel for
	for(int i = 0; i < num_triangles; i++) {
		h_tri_idx_array[i] = i;
		h_aabbs_aos.bboxes[i] = h_triangles_aos.triangles[i].getBounds();
	}
	end = omp_get_wtime();
	printf("Bounding Boxes computed for %d triangles in : %f seconds\n", num_triangles, end-start);
	
	// compute scene box
	start = omp_get_wtime();
	for(int i = 0; i < num_triangles; i++) {
		scene_box_aos.unionWith(h_aabbs_aos.bboxes[i]);
	}
	end = omp_get_wtime();

	float4 extents = scene_box_aos.data.bmax - scene_box_aos.data.bmin;
	scene_box_aos.data.bmin -= 0.1f * extents;
	scene_box_aos.data.bmax += 0.1f * extents;

	lookat = scene_box_aos.centroid3();
	// check if all the camera params have been set
	if(!camera_params_set) {
		printf("WARNING !! - Camera Params not set... Setting Default Values.\n");
		width = 800;
		height = 600;
		up = make_float3(0.0f, 1.0f, 0.0f);
		eye = getEye(scene_box);
	}
	
	film = new Film(width, height);
	camera = new Camera(eye, lookat, up);
	ef = new EyeFrustum(*camera, *film);
	
	printf("Transfering scene data over to the GPU\n");
	Timer tri_transfer_timer("Triangle Data Transfer Timer");
	tri_transfer_timer.start();
	checkCuda(cudaMemcpy(d_triangles_aos.triangles, h_triangles_aos.triangles, sizeof(Triangle3) * num_triangles, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_tri_idx_array, h_tri_idx_array, sizeof(int) * num_triangles, cudaMemcpyHostToDevice));
	tri_transfer_timer.stop();
	tri_transfer_timer.print();

	scene_end = omp_get_wtime();
	printf("Scene initialization with AoS data format completed in : %f seconds\n", scene_end-scene_start);
}

void Scene::setMaterial(const Material& mtl, unsigned int idx_start, unsigned int idx_end) {
	// push the material into space
	//material_list.push_back(std::make_tuple(mtl, idx_start, idx_end));
}