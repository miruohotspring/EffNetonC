#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ndarray.h"
#include "network.h"

static int param_num = 19;
char** names;
ndarray_t* params;

ndarray_t* get_param_from_name(char* name) {
    int index = 1000;
    for (int i = 0; i < param_num; i++) {
        if (strcmp(name, names[i]) == 0) index = i;
    }
    if (index == 1000) {
        printf("name not found\n");
        exit(1);
    }
    return &params[index];
}

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
    printf("\tdone.\n");
    
    // expand_conv
    printf("forwarding: expand_conv...\n");
    int expand_conv_out_size[4] = {1, 96, 112, 112};
    conv2d_layer_t* expand_conv = Conv2d(16, 96, 0, 1, 1, 0, 0, 0, 0, get_param_from_name("_blocks.1._expand_conv.weight"));
    ndarray_t* expand_conv_in = create_param_from_name("expand_conv_in");
    ndarray_t* expand_conv_out = create_empty_ndarray(4, expand_conv_out_size);
    conv2d_forward(expand_conv_out, expand_conv_in, expand_conv);
    ndarray_t* expand_conv_target = create_param_from_name("expand_conv_out");
    printf("\tdone. (loss: %f)\n", calc_loss(expand_conv_out, expand_conv_target));
    
    // batch_norm0
    printf("forwarding: batch_norm0...\n");
    batchnorm_layer_t* bn0 = Batchnorm(
        get_param_from_name("_blocks.1._bn0.running_mean"),
        get_param_from_name("_blocks.1._bn0.running_var"),
        get_param_from_name("_blocks.1._bn0.weight"),
        get_param_from_name("_blocks.1._bn0.bias")
    );
    ndarray_t* bn0_out = create_empty_ndarray(4, expand_conv_out_size);
    batchnorm_forward(bn0_out, expand_conv_out, bn0);
    ndarray_t* bn0_target = create_param_from_name("bn0_out");
    printf("\tdone. (loss: %f)\n", calc_loss(bn0_out, bn0_target));
    
    // swish
    printf("forwarding: swish0...\n");
    ndarray_t* swish0_out = create_empty_ndarray(4, expand_conv_out_size);
    swish(swish0_out, bn0_out);
    ndarray_t* swish0_target = create_param_from_name("swish0_out");
    printf("\tdone. (loss: %f)\n", calc_loss(swish0_out, swish0_target));
    
    // padding
    printf("padding...\n");
    int pad0_size[4] = {1, 96, 113, 113};
    ndarray_t* pad0_out = create_empty_ndarray(4, pad0_size);
    zero_padding(pad0_out, swish0_out, 0, 1, 1, 0);
    printf("\tdone.\n");
    
    // depthwise_conv
    printf("forwarding: depthwise_conv...\n");
    int depthwise_conv_out_size[4] = {1, 96, 56, 56};
    conv2d_layer_t* depthwise_conv = Conv2d(96, 96, 0, 3, 2, 1, 1, 1, 1, get_param_from_name("_blocks.1._depthwise_conv.weight"));
    ndarray_t* depthwise_conv_out = create_empty_ndarray(4, depthwise_conv_out_size);
    conv2d_forward(depthwise_conv_out, pad0_out, depthwise_conv);
    ndarray_t* depthwise_conv_target = create_param_from_name("depthwise_conv_out");
    printf("\tdone. (loss: %f)\n", calc_loss(depthwise_conv_out, depthwise_conv_target));
    
    // batch_norm1
    printf("forwarding: batch_norm1...\n");
    batchnorm_layer_t* bn1 = Batchnorm(
        get_param_from_name("_blocks.1._bn1.running_mean"),
        get_param_from_name("_blocks.1._bn1.running_var"),
        get_param_from_name("_blocks.1._bn1.weight"),
        get_param_from_name("_blocks.1._bn1.bias")
    );
    ndarray_t* bn1_out = create_empty_ndarray(4, depthwise_conv_out_size);
    batchnorm_forward(bn1_out, depthwise_conv_out, bn1);
    ndarray_t* bn1_target = create_param_from_name("bn1_out");
    printf("\tdone. (loss: %f)\n", calc_loss(bn1_out, bn1_target));
    //for (int i = 0; i < 10; i++) printf("%f, ", bn0_out->data[i]);
    //printf("\n");
    //for (int i = 0; i < 10; i++) printf("%f, ", bn0_target->data[i]);
    
    // swish
    printf("forwarding: swish1...\n");
    ndarray_t* swish1_out = create_empty_ndarray(4, depthwise_conv_out_size);
    swish(swish1_out, bn1_out);
    ndarray_t* swish1_target = create_param_from_name("swish1_out");
    printf("\tdone. (loss: %f)\n", calc_loss(swish1_out, swish1_target));
    
    // average pooling
    printf("forwarding: average_pooling...\n");
    int avgpool_size[4] = {1, 96, 1, 1};
    ndarray_t* avgpool_out = create_empty_ndarray(4, avgpool_size);
    average_pooling(avgpool_out, swish1_out);
    ndarray_t* avgpool_target = create_param_from_name("avgpool_out");
    printf("\tdone. (loss: %f)\n", calc_loss(avgpool_out, avgpool_target));
    
    // se_reduce
    printf("forwarding: se_reduce...\n");
    int se_reduce_out_size[4] = {1, 4, 1, 1};
    conv2d_layer_t* se_reduce = Conv2d_bias(96, 4, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name("_blocks.1._se_reduce.weight"),
        get_param_from_name("_blocks.1._se_reduce.bias"));
    ndarray_t* se_reduce_out = create_empty_ndarray(4, se_reduce_out_size);
    conv2d_forward(se_reduce_out, avgpool_out, se_reduce);
    ndarray_t* se_reduce_target = create_param_from_name("se_reduce_out");
    printf("\tdone. (loss: %f)\n", calc_loss(se_reduce_out, se_reduce_target));
    
    // swish
    printf("forwarding: swish2...\n");
    ndarray_t* swish2_out = create_empty_ndarray(4, se_reduce_out_size);
    swish(swish2_out, se_reduce_out);
    ndarray_t* swish2_target = create_param_from_name("swish2_out");
    printf("\tdone. (loss: %f)\n", calc_loss(swish2_out, swish2_target));
    
    // se_expand
    printf("forwarding: se_expand...\n");
    int se_expand_out_size[4] = {1, 96, 1, 1};
    conv2d_layer_t* se_expand = Conv2d_bias(4, 96, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name("_blocks.1._se_expand.weight"),
        get_param_from_name("_blocks.1._se_expand.bias"));
    ndarray_t* se_expand_out = create_empty_ndarray(4, se_expand_out_size);
    conv2d_forward(se_expand_out, swish2_out, se_expand);
    ndarray_t* se_expand_target = create_param_from_name("se_expand_out");
    printf("\tdone. (loss: %f)\n", calc_loss(se_expand_out, se_expand_target));
    
    // sigmoid multiply
    printf("forwarding: sigmoid multiply...\n");
    ndarray_t* sigmoid_out = create_empty_ndarray(4, depthwise_conv_out_size);
    sigmoid_multiply(sigmoid_out, se_expand_out, swish1_out);
    ndarray_t* sigmoid_target = create_param_from_name("sigmoid_out");
    printf("\tdone. (loss: %f)\n", calc_loss(sigmoid_out, sigmoid_target));
    
    // project conv
    printf("forwarding: project_conv...\n");
    int project_conv_out_size[4] = {1, 24, 56, 56};
    conv2d_layer_t* project_conv = Conv2d(96, 24, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name("_blocks.1._project_conv.weight"));
    ndarray_t* project_conv_out = create_empty_ndarray(4, project_conv_out_size);
    conv2d_forward(project_conv_out, sigmoid_out, project_conv);
    ndarray_t* project_conv_target = create_param_from_name("project_conv_out");
    printf("\tdone. (loss: %f)\n", calc_loss(project_conv_out, project_conv_target));
    
    // batch_norm2
    printf("forwarding: batch_norm2...\n");
    batchnorm_layer_t* bn2 = Batchnorm(
        get_param_from_name("_blocks.1._bn2.running_mean"),
        get_param_from_name("_blocks.1._bn2.running_var"),
        get_param_from_name("_blocks.1._bn2.weight"),
        get_param_from_name("_blocks.1._bn2.bias")
    );
    ndarray_t* bn2_out = create_empty_ndarray(4, project_conv_out_size);
    batchnorm_forward(bn2_out, project_conv_out, bn2);
    ndarray_t* bn2_target = create_param_from_name("bn2_out");
    printf("\tdone. (loss: %f)\n", calc_loss(bn2_out, bn2_target));
    for (int i = 0; i < 5; i++) printf("%f, ", bn2_out->data[i]);
    return 0;
}
