#include "ndarray.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

ndarray_t* create_ndarray(int length, int dim, int* size, float* data) {
    ndarray_t* array = (ndarray_t*)malloc(sizeof(ndarray_t));
    array->length = length;
    array->dim = dim;
    array->size = size;
    array->data = data;
    
    return array;
}

ndarray_t* create_empty_ndarray(int dim, int* size) {
    ndarray_t* array = (ndarray_t*)malloc(sizeof(ndarray_t));
    int length = 1;
    for (int i = 0; i < dim; i++) length *= size[i];
    array->length = length;
    array->dim = dim;
    array->size = size;
    array->data = (float*)malloc(sizeof(float)*length);
    for (int i = 0; i < length; i++) {
        array->data[i] = 0;
    }
    return array;
}

void delete_ndarray(ndarray_t* array) {
    free(array->size);
    free(array->data);
    free(array);
}

void product_ewise(ndarray_t* output, const ndarray_t* a, const ndarray_t* b) {
    
}
    







































