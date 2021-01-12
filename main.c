#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ndarray.h"
#include "network.h"

static int param_num = 13;
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
        loss += target->data[i] - x->data[i];
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
    
    
    
    return 0;
}
