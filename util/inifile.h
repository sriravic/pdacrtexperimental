#ifndef __INI_FILE_H__
#define __INI_FILE_H__

#include <string>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>

// ini file reader
// NOTE: increment when you want to increase the attributes here.!!!
static const int NUM_ATTRIBUTES = 30;
// increment the count here when you add a new method to this list
static const int NUM_METHODS = 62;
struct IniFile {
	
	static std::string iniStrings[NUM_ATTRIBUTES];
	static std::string methods[NUM_METHODS];
	std::string data_layout;
	bool shadow;
	bool morton_rays;
	bool morton_tris;
	bool gpu_setup;
	bool secondary_rays;
	bool samples;
	int num_samples;
	int width;
	int height;
	float3 eye_pt;
	float3 point_light;
	std::string primary_ray_method_str;
	std::string shadow_ray_method_str;
	std::string secondary_ray_method_str;
	int primary_ray_method;
	int shadow_ray_method;
	int secondary_ray_method;
	std::string outputFileName;
	bool look_at_set;
	bool rtparams_set;
	size_t num_rays_per_bounce;
	size_t max_depth;
	size_t grid_dim_x;
	size_t grid_dim_y;
	size_t grid_dim_z;

	// dacrtrun time parameters
	float3 look_at;
	int buffer_size;
	int max_segments;
	int parallel_tri_threshold;
	int parallel_ray_threshold;
	int num_rays_per_block;
	int shared_memory_tri_space;
	int ray_buffer_threshold;
	int tri_buffer_threshold;
	int max_segment_threshold;

	// ctor
	IniFile(){
		look_at_set  = false;
		rtparams_set = false;
		secondary_rays = false;
		samples = false;
		grid_dim_x = 1024;
		grid_dim_y = 1;
		grid_dim_z = 1;
	}
	// given an ini string, it splits the value and returns it back
	std::string getValue(std::string iniString); 
	// parse an input file and fill in the details
	bool parseIniFile(std::string file); 
	int findMethodNum(const std::string& method);
	void printValues();
};


#endif