#ifndef __RENDERER_H__
#define __RENDERER_H__

#pragma once
#include <global.h>
#include <util/util.h>
#include <dacrt/dacrt.h>
#include <util/renderdata.h>

// Forward declarations
class Scene;
struct Ray;
struct RayArray;

// declaration
void cpuShadeAmbientOcclusion(int width, int height, int raycnt, int* hitids, float3* buffer, int num_candidates, int num_samples,
	float* ao, int* ao_rays_id, int* pray_association);


class Renderer
{
private:
	static int frame_no;
	
	void printStats(Counters& ctr);
	// SoA Format methods
	void renderPrimaryRaysSoa(Method method, DacrtRunTimeParameters& rtparams, RayArray&, RayArray&, int** h_ray_idx_array, int** d_ray_idx_array, int** h_hitids, float** h_maxts, int** d_hitids, float** d_maxts);
	void renderShadowRaySoa(Method method, DacrtRunTimeParameters& rtparams, RayArray&, RayArray&, int* h_pray_idx_array, 
		int* d_pray_idx_array, int* h_hitids, float* h_maxts, int* d_hitids, float* d_maxts, int shadow_samples = 1);
	// we can determine the primary ray association using the hitids themselves
	void renderSecondaryRaysSoa(Method method, DacrtRunTimeParameters& rtparams, RayArray& h_primary_rays, RayArray& d_primary_rays, 
	int* h_pray_idx_array, int* d_pray_idx_array, int* h_hitids, float* h_maxts, int* d_hitids, float * d_maxts, int max_depth, int samples = 1);
	void setupShadowRaysCuda(RayArray& d_primary_rays, RayArray& d_shadow_rays, int* d_hitids, float* d_maxts, int* d_pray_idx_array,
							int** d_sray_idx_array, int** d_spray_idx_array, int& num_shadow_rays);

	// AoS Format Methods
	void renderPrimaryRaysAos(Method method, DacrtRunTimeParameters& rtparams, RayArrayAos&, RayArrayAos&, int**, int**, int**, float**);

	void shade();			

public:
	Renderer(Scene* scene, const std::string& data_format, const char* output, const char* ao_output, bool enable_morton, bool shadow_enable, bool secondary_enable, const float3& pt_light_pos, bool set_rtparams = false, bool use_cuda = false);
	Renderer(Scene* scene, const std::string& data_format, const char* output, const char* ao_output, bool enable_morton, bool shadow_enable, bool secondary_enable, DacrtRunTimeParameters& _rtparams, const float3& pt_light_pos, bool use_cuda = false);
	~Renderer();
	void render(Method primary_ray_method, Method shadow_ray_method, Method secondary_ray_method, int max_depth = 0, int num_shadow_samples = 1, int num_secondary_samples = 1);			// always call this method only
		
	Scene*		           scene;
	DacrtRunTimeParameters rtparams;
	std::string            data_format;
	const char*	           output;
	const char*			   ao_output;
	float3*	               buffer;
	float3*		           shadow_buffer;
	float3**	           secondary_buffer;		// one buffer for each pass.
	float3                 point_light_pos;		
	MortonCode*            primary_mcodes;			// codes for primary rays
	unsigned int           max_depth;
	unsigned int           num_rays_per_bounce;
	bool		           enable_shadows;
	bool		           enable_secondary;
	bool		           use_morton_codes;
	bool		           use_cuda;
	bool		           rtparams_set;
	Logger*                logfile;
};

#endif