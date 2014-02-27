#include <primitives/primitives_aos.h>
#include <util/model.h>

void Model::loadModelAos() {
	
	int v0, v1, v2;
	triangles_aos = new Triangle3[num_faces];
	for(size_t i = 0; i < num_faces; i++) {
		v0 = faces[0][i];
		v1 = faces[1][i];
		v2 = faces[2][i];
		triangles_aos[i] = Triangle3(vertices[v0], vertices[v1], vertices[v2]);
	}
}

Model::~Model() {
	SAFE_RELEASE(vertices);
	SAFE_RELEASE(faces[0]);
	SAFE_RELEASE(faces[1]);
	SAFE_RELEASE(faces[2]);
	SAFE_RELEASE(triangles[0]);
	SAFE_RELEASE(triangles[1]);
	SAFE_RELEASE(triangles[2]);
	SAFE_RELEASE(triangles_aos);
}