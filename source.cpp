#include <global.h>
#include <util/model.h>
#include <util/scene.h>
#include <util/inifile.h>
#include <math/transform.h>
#include <cuda/cuda_app.h>
#include <renderer/renderer.h>
#include <primitives/primitives.h>
#include <primitives/primitives_aos.h>

#ifdef _WIN32
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif


/**
Set of operations to follow to render a scene
*********************************************
1. Load model/models into memory
2. Create number of scenes as the number of frames
3. Initialize renderer and pass scene by scene or create multiple copies of renderers and let each do a frame in a multithreaded fashion
*/

unsigned int total_device_bytes = 0;				

int main(int argc, char** argv) {
	if(argc != 2) {
		std::cerr<<"Invalid number of inputs. Enter a scene to load\n";
		return -1;
	}

	std::cout<<"Divide and Conquer Ray Tracing - GPU Version 2.0\n";
	std::cout<<"Using Thrust Major Version : "<<THRUST_MAJOR_VERSION<<" ; Thrust Minor Version : "<<THRUST_MINOR_VERSION<<"\n";
	
	// load the ini file to get stuff
	std::cout<<"Reading Ini File\n";
	IniFile ifile;
	ifile.parseIniFile("dacrt.ini");
	ifile.printValues();
	DacrtRunTimeParameters rtparams;
	if(ifile.rtparams_set) {
		rtparams.BUFFER_SIZE = ifile.buffer_size; rtparams.MAX_SEGMENTS = ifile.max_segments; rtparams.MAX_SEGMENT_THRESHOLD = ifile.max_segment_threshold;
		rtparams.NUM_RAYS_PER_BLOCK = ifile.num_rays_per_block; rtparams.PARALLEL_RAY_THRESHOLD = ifile.parallel_ray_threshold; rtparams.PARALLEL_TRI_THRESHOLD = ifile.parallel_tri_threshold;
		rtparams.TRI_SHARED_MEMORY_SPACE = ifile.shared_memory_tri_space;
		rtparams.GRID_DIM_X = ifile.grid_dim_x; rtparams.GRID_DIM_Y = ifile.grid_dim_y; rtparams.GRID_DIM_Z = ifile.grid_dim_z;
	}

	Cuda::printCudaDevices();
	
	Model* m = new Model(argv[1], ifile.data_layout);
	m->printStats();
	Scene* myscene = NULL;
	Renderer* renderer =  NULL;

	if(ifile.data_layout == "soa") {
		TriangleArray triangles(m->triangles[0], m->triangles[1], m->triangles[2]);
		
		// add the transform code here
		float angle = 15.0f;
		Transform roty = rotateY(angle);
		size_t NUM_FRAMES = (size_t)360.0f/angle;

#ifdef _MSC_VER
	std::string filename = "output";
	std::string aofilename = "aoutput";
	//std::string logdir = ".\\logs\\";
#else
	std::string filename = "./output";
	//std::string logdir = "./logs/";
#endif
		std::string filetype = ".jpg";
		std::string logfile = ".csv";


		for(size_t i = 0; i < 1; i++) {

			myscene = new Scene(triangles, m->num_faces, ifile.data_layout, ifile.morton_tris);
			if(!ifile.look_at_set) { 
				myscene->setCameraParams(ifile.eye_pt, ifile.width, ifile.height);
			} else {
				myscene->setCameraParams(ifile.eye_pt, ifile.look_at, ifile.width, ifile.height);
			}
			myscene->initScene(ifile.gpu_setup);

			// Create the output file
			std::stringstream fileno;
			fileno << i;
			std::string finalname = filename + fileno.str() + filetype;
			std::string aofinalname = aofilename + fileno.str() + filetype;
			//std::string logname = logdir + fileno.str() + logfile;

			// Runtime parameters for dacrt algorithm
			if(!ifile.rtparams_set) {
				//renderer = new Renderer(myscene, ifile.data_layout, ifile.outputFileName.c_str(), ifile.morton_rays, ifile.shadow, ifile.secondary_rays, ifile.point_light, false, false);		// setting secondary by default to false
				renderer = new Renderer(myscene, ifile.data_layout, finalname.c_str(), aofinalname.c_str(), ifile.morton_rays, ifile.shadow, ifile.secondary_rays, ifile.point_light, false, false);		// setting secondary by default to false
			} else {
				//renderer = new Renderer(myscene, ifile.data_layout, ifile.outputFileName.c_str(), ifile.morton_rays, ifile.shadow, ifile.secondary_rays, rtparams, ifile.point_light, ifile.gpu_setup);
				renderer = new Renderer(myscene, ifile.data_layout, finalname.c_str(), aofinalname.c_str(), ifile.morton_rays, ifile.shadow, ifile.secondary_rays, rtparams, ifile.point_light, ifile.gpu_setup);
			}

			if(ifile.samples) {
				renderer->render((Method)ifile.primary_ray_method, (Method)ifile.shadow_ray_method, (Method)ifile.secondary_ray_method, 1, 1, ifile.num_samples);
			} else {
				std::cout<<"No Samples set. Setting default value of 1\n";
				renderer->render((Method)ifile.primary_ray_method, (Method)ifile.shadow_ray_method, (Method)ifile.secondary_ray_method, 1, 1, 1);
			}

			// rotate eye point
			ifile.eye_pt = roty(ifile.eye_pt);

			// free up memory here
			if(myscene != NULL) delete myscene;
			if(renderer != NULL) delete renderer;
			//if(m != NULL) delete m;
		}


	} else if(ifile.data_layout == "aos") {

		TriangleArrayAos triangles(m->triangles_aos);
		myscene = new Scene(triangles, m->num_faces, ifile.data_layout, ifile.morton_tris);
		myscene->setCameraParams(ifile.eye_pt, ifile.width, ifile.height);
		myscene->initScene();
		renderer = new Renderer(myscene, ifile.data_layout, ifile.outputFileName.c_str(), NULL, ifile.morton_rays, ifile.shadow, false, ifile.point_light);
		renderer->render((Method)ifile.primary_ray_method, (Method)ifile.shadow_ray_method, (Method)ifile.secondary_ray_method);
	} else {
		std::cerr<<"SEVERE : Invalid data type!!\nExiting\n";
		exit(-1);
	}
	
	//if(myscene != NULL) delete myscene;
	//if(renderer != NULL) delete renderer;
	if(m != NULL) delete m;

#ifdef _WIN32
	_CrtDumpMemoryLeaks();
#endif

	return 0;
}