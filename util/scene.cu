#include <util/scene.h>
#include <util/util.h>
#include <util/cutimer.h>

void Scene::initSceneAosCuda() {

	// first transfer the contents to device memory
	double start, end;
	start = omp_get_wtime();
	printf("DEBUG : Tri count : %d\n", num_triangles);
	printf("Transfering scene data over to the GPU\n");
	Timer tri_transfer_timer("Triangle Data Transfer Timer");
	tri_transfer_timer.start();
	cudaMemcpy(d_triangles.v0, h_triangles.v0, sizeof(float3) * num_triangles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangles.v1, h_triangles.v1, sizeof(float3) * num_triangles, cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangles.v2, h_triangles.v2, sizeof(float3) * num_triangles, cudaMemcpyHostToDevice);
	tri_transfer_timer.stop();
	tri_transfer_timer.print();

	// fill idx data
	Timer seqtimer("Sequence Timer");
	seqtimer.start();
	thrust::sequence(thrust::device_ptr<int>(d_tri_idx_array), thrust::device_ptr<int>(d_tri_idx_array) + num_triangles, 0);
	seqtimer.stop();

	// compute aabbs 
	Timer aabbtimer("Aabb timer");
	aabbtimer.start();
	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<float3>(d_triangles.v0), 
													 thrust::device_ptr<float3>(d_triangles.v1), 
													 thrust::device_ptr<float3>(d_triangles.v2), 
													 thrust::device_ptr<float3>(d_aabbs.bmin), 
													 thrust::device_ptr<float3>(d_aabbs.bmax), 
													 thrust::device_ptr<float3>(d_aabbs.centroid)
													)
								 ),
		thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<float3>(d_triangles.v0) + num_triangles, 
													 thrust::device_ptr<float3>(d_triangles.v1) + num_triangles, 
													 thrust::device_ptr<float3>(d_triangles.v2) + num_triangles, 
													 thrust::device_ptr<float3>(d_aabbs.bmin) + num_triangles, 
													 thrust::device_ptr<float3>(d_aabbs.bmax) + num_triangles, 
													 thrust::device_ptr<float3>(d_aabbs.centroid) + num_triangles
													)
								 ),
		AabbFunctor());
	aabbtimer.stop();
	aabbtimer.print();

	// compute centroids for each triangle
	// reduce the bmins and bmaxs to get scenebox coords
	scene_box.bmin = thrust::reduce(thrust::device_ptr<float3>(d_aabbs.bmin), thrust::device_ptr<float3>(d_aabbs.bmin) + num_triangles, make_float3(FLT_MAX), Float3Minimum());
	scene_box.bmax = thrust::reduce(thrust::device_ptr<float3>(d_aabbs.bmax), thrust::device_ptr<float3>(d_aabbs.bmax) + num_triangles, make_float3(-FLT_MAX), Float3Maximum());
	float3 extents = scene_box.bmax - scene_box.bmin;
	scene_box.bmin -= extents * 0.1f;
	scene_box.bmax += extents * 0.1f;

	lookat = scene_box.centroid();

	MortonFunctor mf(scene_box.bmin, scene_box.bmax);

	checkCuda(cudaMalloc((void**)&d_tri_centroids, sizeof(float3) * num_triangles));
	Timer mctimer("Morton Code Timer");
	mctimer.start();
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<float3>(d_triangles.v0), 
																  thrust::device_ptr<float3>(d_triangles.v1), 
																  thrust::device_ptr<float3>(d_triangles.v2), 
																  thrust::device_ptr<float3>(d_tri_centroids), 
																  thrust::device_ptr<unsigned int>(d_morton_codes))),
		thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<float3>(d_triangles.v0) + num_triangles, 
													 thrust::device_ptr<float3>(d_triangles.v1) + num_triangles, 
													 thrust::device_ptr<float3>(d_triangles.v2) + num_triangles, 
													 thrust::device_ptr<float3>(d_tri_centroids) + num_triangles, 
													 thrust::device_ptr<unsigned int>(d_morton_codes) + num_triangles)),
		mf);
	mctimer.stop();
	mctimer.print();

	// sort the codes and then the triangle vertices and their aabbs also
	Timer sorttimer("Rearrange timer");
	sorttimer.start();
	if(morton_tris) {
		thrust::sort_by_key(thrust::device_ptr<unsigned int>(d_morton_codes), 
						thrust::device_ptr<unsigned int>(d_morton_codes) + num_triangles, 
						thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<float3>(d_triangles.v0), 
																	 thrust::device_ptr<float3>(d_triangles.v1), 
																	 thrust::device_ptr<float3>(d_triangles.v2), 
																	 thrust::device_ptr<float3>(d_aabbs.bmin), 
																	 thrust::device_ptr<float3>(d_aabbs.bmax), 
																	 thrust::device_ptr<float3>(d_aabbs.centroid)
																	 )
												 )
						   );
	}
	sorttimer.stop();
	sorttimer.print();

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

	// now do a memcpy of all data back to host 
	printf("Copying all updated data from device to host\n");
	Timer updated("Update timer");
	updated.start();
	checkCuda(cudaMemcpy(h_triangles.v0, d_triangles.v0, sizeof(float3) * num_triangles, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_triangles.v1, d_triangles.v1, sizeof(float3) * num_triangles, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_triangles.v2, d_triangles.v2, sizeof(float3) * num_triangles, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_tri_idx_array, d_tri_idx_array, sizeof(int) * num_triangles, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_aabbs.bmax, d_aabbs.bmax, sizeof(float3) * num_triangles, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_aabbs.bmin, d_aabbs.bmin, sizeof(float3) * num_triangles, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_aabbs.centroid, d_aabbs.centroid, sizeof(float3) * num_triangles, cudaMemcpyDeviceToHost));
	updated.stop();
	updated.print();
	end = omp_get_wtime();
	printf("Scene initialization with cuda completed in : %f seconds\n", end-start);

}