#ifndef __MODEL_H__
#define __MODEL_H__

#pragma once
#include <global.h>

// forward declaration
struct Triangle3;

class Model
{
public:
	Model():num_vertices(0), num_faces(0){}
	Model(const char* f, std::string& data_layout):filename(f) {
		printf("Loading Model : %s\n", f);
		
		triangles[0] = triangles[1] = triangles[2] = NULL;
		triangles_aos = NULL;
		faces[0] = faces[1] = faces[2] = NULL;
		
		double start, end;
		start = omp_get_wtime();
		loadVerticesAndFaces();
		if(data_layout == "soa")
			loadModelSoa();
		else if(data_layout == "aos")
			loadModelAos();
		else {
			std::cerr<<"SEVERE : Invalid data method!!\n Exiting\n";
			exit(-1);
		}
		end = omp_get_wtime();
		printf("Model : %s loaded in %lf seconds\n", f, end-start);
	}

	void printStats() const { 
		printf("Mesh Statistics : \nTriangle Count : %d\nVertex Count : %d\n", num_faces, num_vertices);
	}

	~Model();

	std::string filename;
	float3* triangles[3];
	float3* vertices;
	Triangle3* triangles_aos;
	int* faces[3];
	size_t num_vertices;
	size_t num_faces;
	
private:
	// this method loads vertices and faces which can be used to construct the triangles themselves
	void loadVerticesAndFaces();
	void loadModelSoa();
	void loadModelAos();
};



#endif