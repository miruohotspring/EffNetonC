#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "ndarray.h"

ndarray_t* create_ndarray(int length, int dim, int* size, double* data) {
    ndarray_t* array = (ndarray_t*)malloc(sizeof(ndarray_t));
    array->length = length;
    array->dim = dim;
    array->size = size;
    array->data = data;
    
    return array;
}


void delete_ndarray(ndarray_t* array) {
    free(array->size);
    free(array->data);
    free(array);
}

void product_ewise(ndarray_t* output, const ndarray_t* a, const ndarray_t* b) {
    
}
    







































