#ifndef NETWORK_H
#define NETWORK_H

#include <stdbool.h>

static int param_num = 309;
extern char** names;
extern ndarray_t* params;

typedef char* name;
typedef struct {
    int in_channels;
    int out_channels;
    int groups;
    int kernel_size;
    int stride;
    int padding_top;
    int padding_bottom;
    int padding_right;
    int padding_left;
    ndarray_t* weight;
    ndarray_t* bias;
} conv2d_layer_t;

typedef struct {
    ndarray_t* running_mean;
    ndarray_t* running_var;
    ndarray_t* weight;
    ndarray_t* bias;
} batchnorm_layer_t;

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    ndarray_t* weight;
    ndarray_t* bias;
} conv2d_t;

typedef struct {
    ndarray_t* input;
    ndarray_t* output;
    ndarray_t* running_mean;
    ndarray_t* running_var;
    ndarray_t* weight;
    ndarray_t* bias;
} batchnorm_t;

typedef struct {
    ndarray_t* input;
    ndarray_t* output;
} swish_t;

typedef struct {
    ndarray_t* input;
    ndarray_t* output;
} sigmoid_mul_t;

typedef struct {
    int top;
    int right;
    int left;
    int bottom;
    ndarray_t* input;
    ndarray_t* output;
} padding_t;

typedef struct {
    conv2d_layer_t* expand_conv;
    conv2d_layer_t* depthwise_conv;
    conv2d_layer_t* se_reduce;
    conv2d_layer_t* se_expand;
    conv2d_layer_t* project_conv;
    batchnorm_layer_t* bn0;
    batchnorm_layer_t* bn1;
    batchnorm_layer_t* bn2;
    int in_channels;
    int out_channels;
    int dw_pad_top;
    int dw_pad_bottom;
    int dw_pad_right;
    int dw_pad_left;
} mb_conv_block_t;

typedef struct {
    ndarray_t* expand_conv_out;
    ndarray_t* depthwise_conv_out;
    ndarray_t* se_reduce_out;
    ndarray_t* se_expand_out;
    ndarray_t* project_conv_out;
    ndarray_t* bn0_out;
    ndarray_t* bn1_out;
    ndarray_t* swish0_out;
    ndarray_t* swish1_out;
    ndarray_t* swish2_out;
    ndarray_t* avgpool_out;
    ndarray_t* sigmoid_out;
    ndarray_t* pad0_out;
    ndarray_t* final_out;
} mb_conv_outputs_t;

// create mbconv block
mb_conv_block_t* MBConvBlock6(
    mb_conv_outputs_t** output,
    const int expand_rate,
    const int block_num,
    const int in_channels,
    const int out_channels,
    const int input_size,
    const int dw_kernel,
    const int dw_stride,
    const int dw_pad_top,
    const int dw_pad_bottom,
    const int dw_pad_right,
    const int dw_pad_left
);

// create layer
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
);
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
);
batchnorm_layer_t* Batchnorm(
    ndarray_t* running_mean,
    ndarray_t* running_var,
    ndarray_t* weight,
    ndarray_t* bias
);

// create conv2d
// 入力行列の大きさは事前に分かっていないといけない
conv2d_t* create_conv2d(
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride
);
ndarray_t* create_conv2d_output(
    int out_channels,
    int kernel_size,
    int stride,
    int ih,
    int iw
);

// create param
ndarray_t* get_param_from_name(char* name);
ndarray_t* create_param_from_name(char* name);
void load_params(char*** names_p, ndarray_t** params_p);

// forward function
void mbconv6_forward(mb_conv_outputs_t* outputs, const ndarray_t* input, const mb_conv_block_t* mb);
void mbconv1_forward(mb_conv_outputs_t* outputs, const ndarray_t* input, const mb_conv_block_t* mb);
void conv2d_forward(ndarray_t* output, const ndarray_t* input, const conv2d_layer_t* layer);
void batchnorm_forward(ndarray_t* output, const ndarray_t* input, const batchnorm_layer_t* layer);

// utility function
float sigmoid(float x);
void sigmoid_multiply(ndarray_t* output, const ndarray_t* input, const ndarray_t* multiplier);
void zero_padding(ndarray_t* output, const ndarray_t* input, int top, int bottom, int right, int left);
void swish(ndarray_t* output, const ndarray_t* input);
void average_pooling(ndarray_t* output, const ndarray_t* input);










































#endif