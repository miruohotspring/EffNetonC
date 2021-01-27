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
    ndarray_t* weight
) {
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
    layer->bias = NULL;
    
    return layer;
}

conv2d_layer_t* Conv2d_bias(
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
) {
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

batchnorm_layer_t* Batchnorm(
    ndarray_t* running_mean,
    ndarray_t* running_var,
    ndarray_t* weight,
    ndarray_t* bias
) {
    batchnorm_layer_t* layer = (batchnorm_layer_t*)malloc(sizeof(batchnorm_layer_t));
    layer->running_mean = running_mean;
    layer->running_var = running_var;
    layer->weight = weight;
    layer->bias = bias;
    
    return layer;
}
    
    
ndarray_t* create_param_from_name(char* name) {
    ndarray_t* param = (ndarray_t*)malloc(sizeof(ndarray_t));
    int length = 1;
    int dim;
    int* size;
    float* data;
    
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
    
    data = (float*)malloc(sizeof(float)*length);
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

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-1.0 * x));
}

void sigmoid_multiply(ndarray_t* output, const ndarray_t* input, const ndarray_t* multiplier) {
    assert(output->dim == multiplier->dim);
    int image_size = output->size[2] * output->size[3];
    for (int c = 0; c < output->size[1]; c++) {
        for (int p = 0; p < image_size; p++) {
            output->data[c*image_size + p] = sigmoid(input->data[c]) * multiplier->data[c*image_size + p];
        }
    }
}

void swish(ndarray_t* output, const ndarray_t* input) {
    assert(output->dim == input->dim);
    for (int i = 0; i < output->length; i++) {
        output->data[i] = input->data[i] * sigmoid(input->data[i]);
    }
}

void conv2d_forward(
    ndarray_t* output,
    const ndarray_t* input,
    const conv2d_layer_t* layer
    )
{
    assert(output->dim == input->dim);
    int image_size = output->size[2] * output->size[3];
    int input_image_size = input->size[2] * input->size[3];
    if (layer->kernel_size == 1 && layer->bias == NULL) {
        for (int cout = 0; cout < output->size[1]; cout++) {
            for (int p = 0; p < image_size; p++) {
                for (int cin = 0; cin < layer->in_channels; cin++) {
                    output->data[cout*image_size + p] +=\
                    layer->weight->data[cout*layer->in_channels + cin] *\
                    input->data[cin*image_size + p];
                }    
            }
        }
    }
    else if (layer->kernel_size == 1) {
        for (int cout = 0; cout < output->size[1]; cout++) {
            for (int p = 0; p < image_size; p++) {
                for (int cin = 0; cin < layer->in_channels; cin++) {
                    output->data[cout*image_size + p] +=\
                    layer->weight->data[cout*layer->in_channels + cin] *\
                    input->data[cin*image_size + p]; 
                }    
                output->data[cout*image_size + p] += layer->bias->data[cout];
            }
        }
    }
    else if (layer->kernel_size == 3) {
        for (int cout = 0; cout < output->size[1]; cout++) {
            for (int row = 0; row < output->size[2]; row++) {
            for (int col = 0; col < output->size[3]; col++) {
                for (int cin = 0; cin < 1; cin++) {
                    int out_i = cout*image_size + row*output->size[3] + col;
                    int weight_i = cout*9;
                    int in_i = cout*input_image_size + row*input->size[3]*2 + col*2;
                    int w = input->size[3];
                    
                    output->data[out_i] += layer->weight->data[weight_i]     * input->data[in_i];
                    output->data[out_i] += layer->weight->data[weight_i + 1] * input->data[in_i + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 2] * input->data[in_i + 2];
                    output->data[out_i] += layer->weight->data[weight_i + 3] * input->data[in_i + w];
                    output->data[out_i] += layer->weight->data[weight_i + 4] * input->data[in_i + w + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 5] * input->data[in_i + w + 2];
                    output->data[out_i] += layer->weight->data[weight_i + 6] * input->data[in_i + w*2];
                    output->data[out_i] += layer->weight->data[weight_i + 7] * input->data[in_i + w*2 + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 8] * input->data[in_i + w*2 + 2];
                    
                }
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
    assert(output->dim == input->dim);
    int image_size = output->size[2] * output->size[3];
    for (int c = 0; c < output->size[1]; c++) {
        for (int p = 0; p < image_size; p++) {
            output->data[c*image_size + p] =\
                ((input->data[c*image_size + p] - layer->running_mean->data[c]) /\
                (sqrt(layer->running_var->data[c]) + 0.00001)) *\
                layer->weight->data[c] + layer->bias->data[c];
/*
            if (output->size[2] == 56) {
                printf("%f, %f, %f, %f, %f, %f\n", input->data[c*image_size + p], layer->running_mean->data[c], layer->running_var->data[c], layer->weight->data[c], layer->bias->data[c], output->data[c*image_size + p]);
                exit(0);
            }
*/
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
void zero_padding(ndarray_t* output, const ndarray_t* input, int top, int bottom, int right, int left) {
    int i_in = 0;
    int i_out = 0;
    for (int c = 0; c < output->size[1]; c++) {
        for (int h = 0; h < output->size[2]; h++) {
            for (int w = 0; w < output->size[3]; w++) {
                if ((h >= top) && (h < output->size[2] - bottom) && (w >= left) && (w < output->size[3] - right)) {
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

void average_pooling(ndarray_t* output, const ndarray_t* input) {
    int image_size = input->size[2] * input->size[3];
    for (int c = 0; c < output->size[1]; c++) {
        float sum = 0;
        for (int p = 0; p < image_size; p++) {
            sum += input->data[c*image_size + p];
        }
        output->data[c] = sum / image_size;
    }
}











































