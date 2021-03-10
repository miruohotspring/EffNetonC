#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ndarray.h"
#include "network.h"

float calc_loss(ndarray_t* x, ndarray_t* target) {
    float loss = 0;
    for (int i = 0; i < target->length; i++) {
        loss += (target->data[i] - x->data[i])*(target->data[i] - x->data[i]);
    }
    return loss;
}

int main(int argc, char* argv[]) {
    printf("loading params...\n");
    load_params(&names, &params);
    
    //ndarray_t* input = create_param_from_name("expand_conv_in");
    ndarray_t* inputs = create_param_from_name("inputs");
    int* pad_size = (int*)malloc(sizeof(int)*4);
    pad_size[0] = 1;
    pad_size[1] = 3;
    pad_size[2] = 225;
    pad_size[3] = 225;
    ndarray_t* conv_stem_in = create_empty_ndarray(4, pad_size);
    ndarray_t* conv_stem_out = create_conv2d_output(32, 3, 2, 225, 225); 
    conv2d_layer_t* conv_stem = Conv2d(3, 32, 0, 3, 2, 0, 0, 0, 0, get_param_from_name("_conv_stem.weight"));
    
    batchnorm_layer_t* bn_stem = Batchnorm(
        get_param_from_name("_bn0.running_mean"),
        get_param_from_name("_bn0.running_var"),
        get_param_from_name("_bn0.weight"),
        get_param_from_name("_bn0.bias")
    );    
    ndarray_t* bn_stem_out = create_empty_ndarray(4, conv_stem_out->size);
    ndarray_t* swish_stem_out = create_empty_ndarray(4, conv_stem_out->size);
    
    mb_conv_block_t* blocks[16];
    mb_conv_outputs_t* outputs[16];
    blocks[0]  = MBConvBlock6( &outputs[0], 1,  0,  32,  16, 112, 3, 1, 1, 1, 1, 1);
    blocks[1]  = MBConvBlock6( &outputs[1], 6,  1,  16,  24, 112, 3, 2, 0, 1, 1, 0);
    blocks[2]  = MBConvBlock6( &outputs[2], 6,  2,  24,  24,  56, 3, 1, 1, 1, 1, 1);
    blocks[3]  = MBConvBlock6( &outputs[3], 6,  3,  24,  40,  56, 5, 2, 1, 2, 2, 1);
    blocks[4]  = MBConvBlock6( &outputs[4], 6,  4,  40,  40,  28, 5, 1, 2, 2, 2, 2);
    blocks[5]  = MBConvBlock6( &outputs[5], 6,  5,  40,  80,  28, 3, 2, 0, 1, 1, 0);
    blocks[6]  = MBConvBlock6( &outputs[6], 6,  6,  80,  80,  14, 3, 1, 1, 1, 1, 1);
    blocks[7]  = MBConvBlock6( &outputs[7], 6,  7,  80,  80,  14, 3, 1, 1, 1, 1, 1);
    blocks[8]  = MBConvBlock6( &outputs[8], 6,  8,  80, 112,  14, 5, 1, 2, 2, 2, 2);
    blocks[9]  = MBConvBlock6( &outputs[9], 6,  9, 112, 112,  14, 5, 1, 2, 2, 2, 2);
    blocks[10] = MBConvBlock6(&outputs[10], 6, 10, 112, 112,  14, 5, 1, 2, 2, 2, 2);
    blocks[11] = MBConvBlock6(&outputs[11], 6, 11, 112, 192,  14, 5, 2, 1, 2, 2, 1);
    blocks[12] = MBConvBlock6(&outputs[12], 6, 12, 192, 192,   7, 5, 1, 2, 2, 2, 2);
    blocks[13] = MBConvBlock6(&outputs[13], 6, 13, 192, 192,   7, 5, 1, 2, 2, 2, 2);
    blocks[14] = MBConvBlock6(&outputs[14], 6, 14, 192, 192,   7, 5, 1, 2, 2, 2, 2);
    blocks[15] = MBConvBlock6(&outputs[15], 6, 15, 192, 320,   7, 3, 1, 1, 1, 1, 1);
    
    zero_padding(conv_stem_in, inputs, 0, 1, 1, 0);
    conv2d_forward(conv_stem_out, conv_stem_in, conv_stem);
    batchnorm_forward(bn_stem_out, conv_stem_out, bn_stem);
    swish(swish_stem_out, bn_stem_out);
    mbconv1_forward(outputs[0], swish_stem_out, blocks[0]);
    for (int i = 1; i < 16; i++) {
        mbconv6_forward(outputs[i], outputs[i-1]->final_out, blocks[i]);
    }
    
    ndarray_t* bn2_target = create_param_from_name("bn2_out");
    printf("done. (loss: %f)\n", calc_loss(outputs[1]->final_out, bn2_target));
    return 0;
}
