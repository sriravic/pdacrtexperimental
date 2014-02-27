#include <math/transform.h>

Transform rotateX(float angle) {
    float sin_t = sinf(Radians(angle));
    float cos_t = cosf(Radians(angle));
    Matrix4x4 m(1,     0,      0, 0,
                0, cos_t, -sin_t, 0,
                0, sin_t,  cos_t, 0,
                0,     0,      0, 1);
    return Transform(m, Transpose(m));
}


Transform rotateY(float angle) {
    float sin_t = sinf(Radians(angle));
    float cos_t = cosf(Radians(angle));
    Matrix4x4 m( cos_t,   0,  sin_t, 0,
                 0,   1,      0, 0,
                -sin_t,   0,  cos_t, 0,
                 0,   0,   0,    1);
    return Transform(m, Transpose(m));
}



Transform rotateZ(float angle) {
    float sin_t = sinf(Radians(angle));
    float cos_t = cosf(Radians(angle));
    Matrix4x4 m(cos_t, -sin_t, 0, 0,
                sin_t,  cos_t, 0, 0,
                0,      0, 1, 0,
                0,      0, 0, 1);
    return Transform(m, Transpose(m));
}

Transform translate(const float3 &delta) {
    Matrix4x4 m(1, 0, 0, delta.x,
                0, 1, 0, delta.y,
                0, 0, 1, delta.z,
                0, 0, 0,       1);
    Matrix4x4 minv(1, 0, 0, -delta.x,
                   0, 1, 0, -delta.y,
                   0, 0, 1, -delta.z,
                   0, 0, 0,        1);
    return Transform(m, minv);
}
