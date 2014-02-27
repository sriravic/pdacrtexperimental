#ifndef __SCENE_H__
#define __SCENE_H__

#pragma once
#include <global.h>
#include <util/model.h>
#include <util/camera.h>
#include <util/material.h>

/**
Improved Scene Representation
-----------------------------

A scene contains objects. Each object has its own transformation matrix. Each object also has its own materials etc. 
A scene also contains a camera. Camera defines where the scene is to be rendered.

We are going to follow the KISS principle.
Scene will contain only triangles, material info, camera params. Scene is setup for each frame of the animation.! So that makes sense to store only
the triangles and their corresponding data only. We'll generate rays on the renderer.

Rays are generated on fly in the renderer class.
*/

class Scene
{
private:
	bool camera_params_set;								// flag to indicate if camera_params are set. If not default values are taken into consideration
	void initSceneAos(bool use_cuda);
	void initSceneSoa(bool use_cuda);
	void initSceneSoaCuda();
	void initSceneAosCuda();
	void arrangeMortonCurve();
	void arrangeMortonCurveSoa();
public:
	Scene();
	Scene(TriangleArray& triangles, int num_triangles, std::string& data_format, bool _morton_tris);
	Scene(TriangleArrayAos& triangles, int num_triangles, std::string& data_format, bool _morton_tris);
	~Scene();
	
	void setCameraParams(const float3& eye, int width, int height, int spp = 1);
	void setCameraParams(const float3& eye, const float3& lookat, int width, int height, int spp = 1);
	void setCameraParams(const float3& eye, const float3& lookat, const float3& up, int width, int height, int spp = 1);
	void setCameraParams(int width, int height, int spp = 1);								// this function calculates the view point automatically
	void initScene(bool use_cuda = false);													// loads all the data
	void setMaterial(const Material& mtl, unsigned int idx_start, unsigned int idx_end);	// we allocate enough space for all triangles in the scene and set material idx appropriately.
	void createMaterialArray();																// creates the actual arrays and fills the arrays								
	
	TriangleArray	h_triangles;														// cpu data
	AabbArray		h_aabbs;															// cpu aabbs
	int*			h_tri_idx_array;

	TriangleArray	d_triangles;
	AabbArray		d_aabbs;
	int*			d_tri_idx_array;

	float3*			d_tri_centroids;
	unsigned int*   d_morton_codes;

	// AoS format
	TriangleArrayAos h_triangles_aos;
	AabbArrayAos	 h_aabbs_aos;
	
	TriangleArrayAos d_triangles_aos;
	AabbArrayAos	 d_aabbs_aos;
		
	int				num_triangles;
	AABB			scene_box;
	AABB4			scene_box_aos;
	std::string		data_format;
	bool			morton_tris;

	// material list
	//std::vector<std::tuple<Material, unsigned int, unsigned int> > material_list;
	Material*       h_materials;
	Material*       d_materials;
	unsigned int*   h_mat_idx;			// material array in host memory
	unsigned int*   d_mat_idx;			// material array in device memory

	// camera params
	float3			eye;
	float3			up;
	float3			lookat;
	int				width;
	int				height;
	int				spp;
	Camera*			camera;
	EyeFrustum*		ef;
	Film*			film;
};



#endif