#include "kernel.h"

void kernel_conv2d_forward(
    float* output, const float* input, const float* weight,
    int filter_size, int stride, int groups,
    int in_channels, int out_channels, int in_size, int out_size
) {
    int cout, cin, row, col, out_i, in_i, weight_i;
    for (cout = 0; cout < out_channels; cout++) {
        for (row = 0; row < out_size; row++) {
        for (col = 0; col < out_size; col++) {
            for (cin = 0; cin < in_channels; cin++) {
                out_i = cout*out_size*out_size + row*out_size + col;
                weight_i = cout*in_channels*filter_size*filter_size + cin*filter_size;
                in_i = cin*in_size*in_size + row*in_size*stride + col*stride;
                
                output[out_i] += weight[weight_i]     * input[in_i];
                output[out_i] += weight[weight_i + 1] * input[in_i + 1];
                output[out_i] += weight[weight_i + 2] * input[in_i + 2];
                output[out_i] += weight[weight_i + 3] * input[in_i + in_size];
                output[out_i] += weight[weight_i + 4] * input[in_i + in_size + 1];
                output[out_i] += weight[weight_i + 5] * input[in_i + in_size + 2];
                output[out_i] += weight[weight_i + 6] * input[in_i + in_size*2];
                output[out_i] += weight[weight_i + 7] * input[in_i + in_size*2 + 1];
                output[out_i] += weight[weight_i + 8] * input[in_i + in_size*2 + 2];
                
            }
        }
        }
    }
}

