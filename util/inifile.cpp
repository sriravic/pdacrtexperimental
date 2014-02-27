#include <util/inifile.h>

// initialize static members
std::string IniFile::iniStrings[] = {"[WIDTH]", "[HEIGHT]", "[EYE_PT]", "[PRIMARY_METHOD]", "[SHADOW_METHOD]", "[SECONDARY_METHOD]", 
	"[OUTPUT]", "[ENABLE_SHADOWS]", "[MORTON_RAYS]", "[DATA_LAYOUT]", "[MORTON_TRIS]", "[GPU_SETUP]", "[LOOK_AT]", "[BUFFER_SIZE]",
	"[MAX_SEGMENTS]", "[PARALLEL_TRI_THRESHOLD]", "[PARALLEL_RAY_THRESHOLD]", "[NUM_RAYS_PER_BLOCK]", "[TRI_SHARED_MEMORY_SPACE]",
	"[RAY_BUFFER_THRESHOLD]", "[TRI_BUFFER_THRESHOLD]", "[MAX_SEGMENT_THRESHOLD]", "[ENABLE_SECONDARY]", "[LIGHT]", "[MAX_DEPTH]", 
	"[NUM_RAYS_BOUNCE]", "[SAMPLES]", "[GRID_DIM_X]", "[GRID_DIM_Y]", "[GRID_DIM_Z]"
};

std::string IniFile::methods[] = {"CPU_OBJECT_PRIMARY", "CPU_OBJECT_SHADOW", "CPU_OBJECT_SECONDARY",			// different passes of the scene
			 "CPU_SPATIAL_PRIMARY", "CPU_SPATIAL_SHADOW", "CPU_SPATIAL_SECONDARY",
			 // this contains the filtering code done on the cpu and brute force done on the gpu
			 "CPU_GPU_OBJECT_PRIMARY", "CPU_GPU_OBJECT_SHADOW", "CPU_GPU_OBJECT_SECONDARY",
			 "CPU_GPU_SPATIAL_PRIMARY", "CPU_GPU_SPATIAL_SHADOW", "CPU_GPU_SPATIAL_SECONDARY",
			 "CPU_GPU_SPATIAL_MODIFIED",
			 "CPU_GPU_DBUFFER_SPATIAL_PRIMARY",
			 "CPU_GPU_TWIN_TREES",
			 // these flags for all the work done on the gpu
			 "GPU_OBJECT_PRIMARY", "GPU_OBJECT_SHADOW", "GPU_OBJECT_SECONDARY",
			 "GPU_SPATIAL_PRIMARY", "GPU_SPATIAL_SHADOW", "GPU_SPATIAL_SECONDARY",
			 "GPU_SPATIAL_PRIMARY_SEGMENTED", "GPU_SPATIAL_SHADOW_SEGMENTED", "GPU_SPATIAL_SECONDARY_SEGMENTED",
			 "GPU_SPATIAL_CELL", "GPU_SPATIAL_JUMP", "GPU_SPATIAL_FULLY_PARALLEL", "GPU_SPATIAL_FULLY_PARALLEL_MODIFIED",
			 "GPU_DACRT_FULLY_CUDA", "GPU_DACRT_FULLY_CUDA_SHADOW", "GPU_DACRT_FULLY_CUDA_SECONDARY",
			 "GPU_DACRT_FULLY_CUDA_OBJECT", "GPU_DACRT_FULLY_CUDA_SHADOW_OBJECT", "GPU_DACRT_FULLY_CUDA_SECONDARY_OBJECT",
			 "GPU_CONES_CUDA", "GPU_CUDA_AO"

			 /* AoS Methods follow */
			 "CPU_OBJECT_PRIMARY_AOS", "CPU_OBJECT_SHADOW_AOS", "CPU_OBJECT_SECONDARY_AOS",
			 "CPU_SPATIAL_PRIMARY_AOS", "CPU_SPATIAL_SHADOW_AOS", "CPU_SPATIAL_SECONDARY_AOS",
			 "CPU_GPU_OBJECT_PRIMARY_AOS", "CPU_GPU_OBJECT_SHADOW_AOS", "CPU_GPU_OBJECT_SECONDARY_AOS",
			 "CPU_GPU_SPATIAL_PRIMARY_AOS", "CPU_GPU_SPATIAL_SHADOW_AOS", "CPU_GPU_SPATIAL_SECONDARY_AOS",
			 "CPU_GPU_SPATIAL_MODIFIED_AOS",
			 "CPU_GPU_DBUFFER_SPATIAL_PRIMARY_AOS",
			 "CPU_GPU_TWIN_TREES_AOS",
			 "GPU_OBJECT_PRIMARY_AOS", "GPU_OBJECT_SHADOW_AOS", "GPU_OBJECT_SECONDARY_AOS",
			 "GPU_SPATIAL_PRIMARY_AOS", "GPU_SPATIAL_SHADOW_AOS", "GPU_SPATIAL_SECONDARY_AOS",
			 "GPU_SPATIAL_PRIMARY_SEGMENTED_AOS", "GPU_SPATIAL_SHADOW_SEGMENTED_AOS", "GPU_SPATIAL_SECONDARY_SEGMENTED_AOS",
			 "GPU_SPATIAL_CELL_AOS", "GPU_SPATIAL_JUMP_AOS"};

std::string IniFile::getValue(std::string iniString) {
	size_t equals;
	equals = iniString.find("=");
	if(iniString.at(equals + 1) == ' ') 
		equals = equals + 2;
	// now copy from equals till end
	std::string retString = iniString.substr(equals);
	return retString;
}

int IniFile::findMethodNum(const std::string& method) {
	// iterate through the methods list and return the number which it satisfies
	int m = -1;
	for(int i = 0; i < NUM_METHODS; i++) {
		if(method == methods[i]) {
			m = i;
			break;
		}
	}
	return m;
}

bool IniFile::parseIniFile(std::string file) {
	std::ifstream input(file.c_str());
		if(input.good()) {
			// read off values
			// see ini file structure 
			std::string temp;
			char buffer[1024];
			float x, y, z;
			char* tempc = NULL;
			char delimiters[] = " {,}";
			char* val = NULL;
			
			while(!input.eof()) {
				std::string inputString;
				//input>>inputString;
				input.getline(buffer, 1024);
				inputString = std::string(buffer);
				size_t position = inputString.find("#");
				if(position == std::string::npos && inputString.length() > 1) {
					// not a comment
					// safely ignore all other lines

					// check if there is keyword
					size_t found;
					
					
					for(int i = 0; i < NUM_ATTRIBUTES; i++) {
						found = inputString.find(IniFile::iniStrings[i]);
						if(found != std::string::npos) {
							switch(i) {
							case 0:
								this->width = atoi(getValue(inputString).c_str());
								break;
							case 1:
								this->height = atoi(getValue(inputString).c_str());
								break;
							case 2:
								// compute the eye point
								// format:{x, y, z}
								temp = getValue(inputString);
								tempc = new char[temp.length()];
								temp.copy(tempc, temp.length());
								val = strtok(tempc, delimiters);
								// do it three times
								if(val != NULL) {
									x = atof(val);
									val = strtok(NULL, delimiters);
								}
								
								if(val != NULL) {
									y = atof(val);
									val = strtok(NULL, delimiters);
								}

								if(val != NULL) {
									z = atof(val);
								}
								this->eye_pt = make_float3(x, y, z);
								delete[] tempc;
								break;
							case 3:
								this->primary_ray_method_str = getValue(inputString);
								primary_ray_method = findMethodNum(primary_ray_method_str);
								break;
							case 4:
								this->shadow_ray_method_str = getValue(inputString);
								shadow_ray_method = findMethodNum(shadow_ray_method_str);
								break;
							case 5:
								this->secondary_ray_method_str = getValue(inputString);
								secondary_ray_method = findMethodNum(secondary_ray_method_str);
								break;
							case 6:
								this->outputFileName = getValue(inputString);
								break;
							case 7:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->shadow = true;
								else this->shadow = false;
								break;
							case 8:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->morton_rays = true;
								else this->morton_rays = false;
								break;
							case 9:
								this->data_layout = getValue(inputString);
								break;
							case 10:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->morton_tris = true;
								else this->morton_tris = false;
								break;
							case 11:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->gpu_setup = true;
								else this->gpu_setup = false;
								break;
							case 12:
								// look at params
								temp = getValue(inputString);
								tempc = new char[temp.length()];
								temp.copy(tempc, temp.length());
								val = strtok(tempc, delimiters);
								// do it three times
								if(val != NULL) {
									x = atof(val);
									val = strtok(NULL, delimiters);
								}
								
								if(val != NULL) {
									y = atof(val);
									val = strtok(NULL, delimiters);
								}

								if(val != NULL) {
									z = atof(val);
								}
								this->look_at = make_float3(x, y, z);
								look_at_set = true;
								delete[] tempc;

								break;
							case 13:
								// buffer size
								buffer_size = atoi(getValue(inputString).c_str());
								rtparams_set = true;
								break;
							case 14:
								// max segments
								max_segments = atoi(getValue(inputString).c_str());
								break;
							case 15:
								// parallel tri threshold
								parallel_tri_threshold = atoi(getValue(inputString).c_str());
								break;
							case 16:
								// parallel ray threshold
								parallel_ray_threshold = atoi(getValue(inputString).c_str());
								break;
							case 17:
								// num rays per block
								num_rays_per_block = atoi(getValue(inputString).c_str());
								break;
							case 18:
								// shared memory size
								shared_memory_tri_space = atoi(getValue(inputString).c_str());
								break;
							case 19:
								// ray buffer threshold
								ray_buffer_threshold = atoi(getValue(inputString).c_str());
								break;
							case 20:
								// tri buffer threshold
								tri_buffer_threshold = atoi(getValue(inputString).c_str());
								break;
							case 21:
								// segment threshold
								max_segment_threshold = atoi(getValue(inputString).c_str());
								break;
							case 22:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->secondary_rays = true;
								else this->secondary_rays = false;
								break;
							case 23:
								temp = getValue(inputString);
								tempc = new char[temp.length()];
								temp.copy(tempc, temp.length());
								val = strtok(tempc, delimiters);
								// do it three times
								if(val != NULL) {
									x = atof(val);
									val = strtok(NULL, delimiters);
								}
								
								if(val != NULL) {
									y = atof(val);
									val = strtok(NULL, delimiters);
								}

								if(val != NULL) {
									z = atof(val);
								}
								this->point_light = make_float3(x, y, z);
								delete[] tempc;
								break;
							case 24:
								max_depth = atoi(getValue(inputString).c_str());
								break;
							case 25:
								num_rays_per_bounce = atoi(getValue(inputString).c_str());
								break;
							case 26:
								num_samples = atoi(getValue(inputString).c_str());
								samples = true;
								break;
							case 27:
								grid_dim_x = (size_t)(atoi(getValue(inputString).c_str()));
								break;
							case 28:
								grid_dim_y = (size_t)(atoi(getValue(inputString).c_str()));
								break;
							case 29:
								grid_dim_z = (size_t)(atoi(getValue(inputString).c_str()));
								break;
							};
						}
					}
					
				} else continue;
			}
		} else {
			std::cerr<<"\n Invalid ini file";
			return false;
		}
		return true;
}

// This prints out all values as denoted by the inifile
void IniFile::printValues() {

	printf("Initialization values\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("Width                  : %d\n", width);
	printf("Height                 : %d\n", height);
	printf("eye pt                 : {%f, %f, %f}\n", eye_pt.x, eye_pt.y, eye_pt.z);
	if(look_at_set)
		printf("look at                : {%f, %f, %f}\n", look_at.x, look_at.y, look_at.z);
	printf("Primary Method         : %s\n", primary_ray_method_str.c_str());
	printf("Primary Method value   : %d\n", primary_ray_method);
	printf("Secondary Method       : %s\n", secondary_ray_method_str.c_str());
	printf("secondary Method value : %d\n", secondary_ray_method);
	printf("Shadow method          : %s\n", shadow_ray_method_str.c_str());
	printf("Shadow Method value    : %d\n", shadow_ray_method);
	printf("Enable shadows         : %s\n", shadow ? "Yes":"No");
	printf("Enable Secondary Rays  : %s\n", secondary_rays ? "Yes":"No");
	printf("Enable morton rays     : %s\n", morton_rays ? "Yes" : "No");
	printf("Enable morton tris     : %s\n", morton_tris ? "Yes" : "No");
	printf("Use gpu scene setup    : %s\n", gpu_setup ? "Yes" : "No");
	printf("Data Layout            : %s\n", data_layout.c_str());
	printf("=====================================\n\n");
}

