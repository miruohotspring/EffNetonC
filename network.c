#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "ndarray.h"
#include "network.h"

conv2d_layer_t* Conv2d(
    int in_channels,
    int out_channels,
    int groups,
    int kernel_size,
    int stride,
    int padding_top,
    int padding_bottom,
    int padding_right,
    int padding_left,
    ndarray_t* weight,
    ndarray_t* bias
){
    conv2d_layer_t* layer = (conv2d_layer_t*)malloc(sizeof(conv2d_layer_t));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->groups = groups;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding_top = padding_top;
    layer->padding_bottom = padding_bottom;
    layer->padding_right = padding_right;
    layer->padding_left = padding_left;
    layer->weight = weight;
    layer->bias = bias;
    
    return layer;
}
    
ndarray_t* create_param_from_name(char* name) {
    ndarray_t* param = (ndarray_t*)malloc(sizeof(ndarray_t));
    int length = 1;
    int dim;
    int* size;
    double* data;
    
    FILE* f;
    char path[100] = "./data/";
    char c;
    char s[128];
    int j = 0;
    strcat(path, name);
    strcat(path, ".txt");
    if ((f = fopen(path, "r")) == NULL) {
        printf("file open error");
        exit(EXIT_FAILURE);
    }
    if ((dim = fgetc(f) - '0') == 0) return NULL;
    while ((fgetc(f)) != ',');
    
    size = (int*)malloc(sizeof(int)*dim);
    for (int i = 0; i < dim; i++) {
        for (j = 0; (c = fgetc(f)), (c != ',' && c != EOF); j++) s[j] = c;
        s[j] = 0;
        size[i] = atoi(s);
        length *= size[i];
    }
    
    data = (double*)malloc(sizeof(double)*length);
    for (int i = 0; i < length; i++) {
        for (j = 0; (c = fgetc(f)), (c != ',' && c != EOF); j++) s[j] = c;
        s[j] = 0;
        data[i] = atof(s);
    }
    
    param->length = length;
    param->dim = dim;
    param->size = size;
    param->data = data;
    
    fclose(f);
    return param;
}

void load_params(char*** names_p, ndarray_t** params_p) {
    *params_p = (ndarray_t*)malloc(213 * sizeof(ndarray_t));
    *names_p = (char**)malloc(213 * sizeof(char*));
    for (int i = 0; i < 213; i++) {
        (*names_p)[i] = (char*)malloc(100 * sizeof(char));
    }
    
    FILE* f;
    char readline[100];
    
    if ((f = fopen("./data/module_list.txt", "r")) == NULL) {
        printf("file open error: cannot open list");
        exit(EXIT_FAILURE);
    }
    int i = 0;
    while (fgets(readline, 100, f) != NULL) {
        readline[strcspn(readline, "\n")] = 0;
        ndarray_t* param = create_param_from_name(readline);
        if (param == NULL) continue;
        strcpy((*names_p)[i], readline);
        (*params_p)[i] = *param;
        i++;
    }    
    fclose(f);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-1 * x));
}

void swish(ndarray_t* output, const ndarray_t* input) {
    assert(output->size == input->size);
    for (int i = 0; i < output->length; i++) {
        output->data[i] = sigmoid(input->data[i]);
    }
}

void conv2d_forward(
    ndarray_t* output,
    const ndarray_t* input,
    const conv2d_layer_t* layer
    )
{
    assert(output->dim == input->dim);
    
    int hw_max = output->size[1] * output->size[2];
    if (layer->kernel_size == 1) {
        for (int cout = 0; cout < output->size[0]; cout++) {
            for (int hw = 0; hw < hw_max; hw++) {
                for (int cin = 0; cin < layer->in_channels; cin++) {
                    output->data[cout*hw_max + hw] +=\
                    layer->weight->data[cout*layer->in_channels + cin] *\
                    input->data[cin*hw_max + hw];
                }    
            }
        }
    }
}

void batchnorm_forward(
    ndarray_t* output,
    const ndarray_t* input,
    const batchnorm_layer_t* layer
    )
{
    assert(output->size == input->size);
    
    int i = 0;
    int hw_max = output->size[1] * output->size[2];
    for (int c = 0; c < output->size[0]; c++) {
        for (int hw = 0; hw < hw_max; hw++) {
            output->data[i] =\
                ((input->data[i] - layer->running_mean->data[c]) /\
                (sqrt(layer->running_var->data[c]) + 0.00001)) *\
                layer->weight->data[c] + layer->bias->data[c];
            i++;
        }
    }
}

/*
zero padding
example with padding of (top, bottom, right, left) = (1, 1, 1, 1):
                   |0 0 0 0|
|a b|              |0 a b 0|
|c d|      ->      |0 c d 0|
                   |0 0 0 0|
*/
void zero_padding(ndarray_t* output, ndarray_t* input, int top, int bottom, int right, int left) {
    assert(output->dim == 3);
    assert(input->dim == 3);
    
    int i_in = 0;
    int i_out = 0;
    for (int c = 0; c < output->size[0]; c++) {
        for (int h = 0; h < output->size[1]; h++) {
            for (int w = 0; w < output->size[2]; w++) {
                if ((h >= top) && (h < output->size[1] - bottom) && (w >= left) && (w < output->size[2] - right)) {
                    output->data[i_out] = input->data[i_in];
                    i_in++;
                } else {
                    output->data[i_out] = 0;
                }
                i_out++;
            }
        }
    }
}











































