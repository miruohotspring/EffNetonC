/*
 * ndarray.h
 *
 *  Created on: Mar 17, 2021
 *      Author: shirakawa
 */

#ifndef NDARRAY_HPP
#define NDARRAY_HPP
#define MEM_ALIGNMENT 4096

typedef struct {
    int length;
    int dim;
    int* size;
    float* data;
} ndarray_t;

ndarray_t* create_ndarray(
	int length,
	int dim,
	int* size,
	float* data
);

ndarray_t* create_empty_ndarray(
	int dim,
	int* size
);

void delete_ndarray(
	ndarray_t* param
);

void product_ewise(
	ndarray_t* output,
	const ndarray_t* a,
	const ndarray_t* b
);

void add_ndarray(
	ndarray_t* x,
	const ndarray_t* y
);

void print_size(
	ndarray_t* a
);

#endif /* SRC_NDARRAY_H_ */