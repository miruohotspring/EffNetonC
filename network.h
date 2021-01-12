#ifndef NETWORK_H
#define NETWORK_H

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
    ndarray_t* weight;
    ndarray_t* bias;
} fc_layer_t;

typedef struct {
    conv2d_layer_t conv_layers[5];
    batchnorm_layer_t bn_layers[3];
} mb_conv1_block_t;

typedef struct {
    conv2d_layer_t conv_layers[4];
    batchnorm_layer_t bn_layers[2];
} mb_conv6_block_t;

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
batchnorm_layer_t* Batchnorm(
    ndarray_t* running_mean,
    ndarray_t* running_var,
    ndarray_t* weight,
    ndarray_t* bias
);
fc_layer_t* Fc(
    ndarray_t* weight,
    ndarray_t* bias
);    

// create param
ndarray_t* create_param_from_name(char* name);
void load_params(char*** names_p, ndarray_t** params_p);

// forward function
void conv2d_forward(ndarray_t* output, const ndarray_t* input, const conv2d_layer_t* layer);
void batchnorm_forward(ndarray_t* output, const ndarray_t* input, const batchnorm_layer_t* layer);
void fc_forward(ndarray_t* output, const ndarray_t* input, const fc_layer_t* layer);

// utility function
double sigmoid(double x);
void zero_padding(ndarray_t* output, ndarray_t* input, int top, int bottom, int right, int left);
void swish(ndarray_t* output, const ndarray_t* input);
void average_pooling(ndarray_t* output, const ndarray_t* input);










































#endif