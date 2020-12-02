#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    int in_channels;
    int out_channels;
    int groups;
    int kernel_size;
    int stride;
    ndarray_t* weight;
    ndarray_t* bias;
} conv2d_layer_t;

typedef struct {
    ndarray_t* running_mean;
    ndarray_t* running_var;
} batchnorm_layer_t;

typedef struct {
    ndarray_t* weight;
    ndarray_t* bias;
} fc_layer_t;

typedef struct {
    int expand_ratio;
    conv2d_layer_t* conv_layers;
    batchnorm_layer_t* bn_layers;
} mb_conv_block_t;

// create layer
conv2d_layer_t* conv2d(
    int in_channels,
    int out_channels,
    int groups,
    int kernel_size,
    int stride,
    ndarray_t* weight,
    ndarray_t* bias
);
batchnorm_layer_t* batchnorm(
    ndarray_t* running_mean,
    ndarray_t* running_var
);
fc_layer_t* fc(
    ndarray_t* weight,
    ndarray_t* bias
);    

// create param
ndarray_t* create_param_from_name(char* name);
ndarray_t* load_params();

// forward function
void conv2d_forward(ndarray_t* output, const ndarray_t* input, conv2d_layer_t* layer);
void batchnorm_forward(ndarray_t* output, const ndarray_t* input, batchnorm_layer_t* layer);
void fc_forward(ndarray_t* output, const ndarray_t* input, fc_layer_t* layer);

// utility function
double sigmoid(double x);
void swish(ndarray_t* output, const ndarray_t* input);
void average_pooling(ndarray_t* output, const ndarray_t* input);

#endif