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
    
    ndarray_t* input = create_param_from_name("expand_conv_in");
    
    /*
    // expand_conv
    int expand_conv_out_size[4] = {1, 96, 112, 112};
    conv2d_layer_t* expand_conv = Conv2d(16, 96, 0, 1, 1, 0, 0, 0, 0, get_param_from_name("_blocks.1._expand_conv.weight"));
    ndarray_t* expand_conv_out = create_empty_ndarray(4, expand_conv_out_size);
    
    // batch_norm0
    batchnorm_layer_t* bn0 = Batchnorm(
        get_param_from_name("_blocks.1._bn0.running_mean"),
        get_param_from_name("_blocks.1._bn0.running_var"),
        get_param_from_name("_blocks.1._bn0.weight"),
        get_param_from_name("_blocks.1._bn0.bias")
    );
    ndarray_t* bn0_out = create_empty_ndarray(4, expand_conv_out_size);
    
    // swish
    ndarray_t* swish0_out = create_empty_ndarray(4, expand_conv_out_size);
    
    // padding
    int pad0_size[4] = {1, 96, 113, 113};
    ndarray_t* pad0_out = create_empty_ndarray(4, pad0_size);
    
    // depthwise_conv
    int depthwise_conv_out_size[4] = {1, 96, 56, 56};
    conv2d_layer_t* depthwise_conv = Conv2d(96, 96, 0, 3, 2, 1, 1, 1, 1, get_param_from_name("_blocks.1._depthwise_conv.weight"));
    ndarray_t* depthwise_conv_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // batch_norm1
    batchnorm_layer_t* bn1 = Batchnorm(
        get_param_from_name("_blocks.1._bn1.running_mean"),
        get_param_from_name("_blocks.1._bn1.running_var"),
        get_param_from_name("_blocks.1._bn1.weight"),
        get_param_from_name("_blocks.1._bn1.bias")
    );
    ndarray_t* bn1_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // swish
    ndarray_t* swish1_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // average pooling
    int avgpool_size[4] = {1, 96, 1, 1};
    ndarray_t* avgpool_out = create_empty_ndarray(4, avgpool_size);
    
    // se_reduce
    int se_reduce_out_size[4] = {1, 4, 1, 1};
    conv2d_layer_t* se_reduce = Conv2d_bias(96, 4, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name("_blocks.1._se_reduce.weight"),
        get_param_from_name("_blocks.1._se_reduce.bias"));
    ndarray_t* se_reduce_out = create_empty_ndarray(4, se_reduce_out_size);
    
    // swish
    ndarray_t* swish2_out = create_empty_ndarray(4, se_reduce_out_size);
    
    // se_expand
    int se_expand_out_size[4] = {1, 96, 1, 1};
    conv2d_layer_t* se_expand = Conv2d_bias(4, 96, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name("_blocks.1._se_expand.weight"),
        get_param_from_name("_blocks.1._se_expand.bias"));
    ndarray_t* se_expand_out = create_empty_ndarray(4, se_expand_out_size);
    
    // sigmoid multiply
    ndarray_t* sigmoid_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // project conv
    int project_conv_out_size[4] = {1, 24, 56, 56};
    conv2d_layer_t* project_conv = Conv2d(96, 24, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name("_blocks.1._project_conv.weight"));
    ndarray_t* project_conv_out = create_empty_ndarray(4, project_conv_out_size);
    
    // batch_norm2
    batchnorm_layer_t* bn2 = Batchnorm(
        get_param_from_name("_blocks.1._bn2.running_mean"),
        get_param_from_name("_blocks.1._bn2.running_var"),
        get_param_from_name("_blocks.1._bn2.weight"),
        get_param_from_name("_blocks.1._bn2.bias")
    );
    ndarray_t* bn2_out = create_empty_ndarray(4, project_conv_out_size);
    
    // forward
    
    conv2d_forward(expand_conv_out, expand_conv_in, expand_conv);
    batchnorm_forward(bn0_out, expand_conv_out, bn0);
    swish(swish0_out, bn0_out);
    zero_padding(pad0_out, swish0_out, 0, 1, 1, 0);
    conv2d_forward(depthwise_conv_out, pad0_out, depthwise_conv);
    batchnorm_forward(bn1_out, depthwise_conv_out, bn1);
    swish(swish1_out, bn1_out);
    average_pooling(avgpool_out, swish1_out);
    conv2d_forward(se_reduce_out, avgpool_out, se_reduce);
    swish(swish2_out, se_reduce_out);
    conv2d_forward(se_expand_out, swish2_out, se_expand);
    sigmoid_multiply(sigmoid_out, se_expand_out, swish1_out);
    conv2d_forward(project_conv_out, sigmoid_out, project_conv);
    batchnorm_forward(bn2_out, project_conv_out, bn2);
    */
    
    mb_conv_block_t* block1 = (mb_conv_block_t*)malloc(sizeof(mb_conv_block_t));
    mb_conv_outputs_t* out1 = (mb_conv_outputs_t*)malloc(sizeof(mb_conv_outputs_t));
    MBConvBlock6(block1, out1, 1, 16, 24, 112, 3, 2, 0, 1, 1, 0);
    mbconv6_forward(out1, input, block1);
    ndarray_t* bn2_target = create_param_from_name("bn2_out");
    printf("done. (loss: %f)\n", calc_loss(out1->final_out, bn2_target));
    return 0;
}
