#ifndef NDARRAY_H
#define NDARRAY_H

typedef struct {
    int length;
    int dim;
    int* size;
    float* data;
} ndarray_t;

ndarray_t* create_ndarray(int length, int dim, int* size, float* data);
ndarray_t* create_empty_ndarray(int dim, int* size);
void delete_ndarray(ndarray_t* param);

void product_ewise(ndarray_t* output, const ndarray_t* a, const ndarray_t* b);

#endif