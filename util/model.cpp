#include <util/model.h>
#include <TriMesh.h>

// this method loads the vertices
void Model::loadVerticesAndFaces() {
	TriMesh *mesh = TriMesh::read(filename.c_str());
	num_vertices = mesh->vertices.size();
	num_faces = mesh->faces.size();
	vertices = new float3[num_vertices];
	faces[0] = new int[num_faces];
	faces[1] = new int[num_faces];
	faces[2] = new int[num_faces];

	for(size_t i = 0; i < num_vertices; i++) {
		float3 vertex;
		vertex = make_float3(mesh->vertices[i][0], mesh->vertices[i][1], mesh->vertices[i][2]);
		vertices[i] = vertex;
	}
	
	for(size_t i = 0; i < num_faces; i++) {
		faces[0][i] = mesh->faces[i][0];
		faces[1][i] = mesh->faces[i][1];
		faces[2][i] = mesh->faces[i][2];
	}
	delete mesh;
}

void Model::loadModelSoa() {
	triangles[0] = new float3[num_faces];
	triangles[1] = new float3[num_faces];
	triangles[2] = new float3[num_faces];
	int v0, v1, v2;
	for(size_t i = 0; i < num_faces; i++) {
		v0 = faces[0][i];
		v1 = faces[1][i];
		v2 = faces[2][i];
		triangles[0][i] = vertices[v0];
		triangles[1][i] = vertices[v1];
		triangles[2][i] = vertices[v2];
	}
}

