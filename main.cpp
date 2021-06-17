#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "ndarray.hpp"
#include "network.hpp"
#include "kernel.hpp"
#include "util.hpp"

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
    
    // block arguments
    const int expand_rate = 6;
    const int block_num = 1;
    const int in_channels = 16;
    const int out_channels = 24;
    const int input_size = 112;
    const int dw_kernel = 3;
    const int dw_stride = 2;
    const int dw_pad_top = 0;
    const int dw_pad_bottom = 1;
    const int dw_pad_right = 1;
    const int dw_pad_left = 0;
    
    // create outputs
    mb_conv_outputs_t* krnl_out;
    mb_conv_outputs_t* cpu_out;
    
    // create blocks
    mb_conv_block_t* krnl_block = MBConvBlock6(
        &krnl_out,
        expand_rate,
        block_num,
        in_channels,
        out_channels,
        input_size,
        dw_kernel,
        dw_stride,
        dw_pad_top,
        dw_pad_bottom,
        dw_pad_right,
        dw_pad_left
    );
    mb_conv_block_t* cpu_block = MBConvBlock6(
        &cpu_out,
        expand_rate,
        block_num,
        in_channels,
        out_channels,
        input_size,
        dw_kernel,
        dw_stride,
        dw_pad_top,
        dw_pad_bottom,
        dw_pad_right,
        dw_pad_left
    );
    
    // CPU inference
    mbconv6_forward(cpu_out, input, cpu_block);
    
    // Layer Fusion 1.
    // create a new bias array
    ndarray_t* fused_conv1_bias;
    int* fu1_bias_size = (int*)aligned_alloc(MEM_ALIGNMENT, sizeof(int));
    fu1_bias_size[0] = krnl_block->expand_conv->weight->size[0];
    fused_conv1_bias = create_empty_ndarray(1, fu1_bias_size);
    for (int c = 0; c < krnl_block->expand_conv->weight->size[0]; c++) {
        fused_conv1_bias->data[c] =\
            -1 * krnl_block->bn0->running_mean->data[c]\
            * (1 / sqrt(krnl_block->bn0->running_var->data[c] + 0.001))\
            * krnl_block->bn0->weight->data[c]\
            + krnl_block->bn0->bias->data[c];
    }
    
    int weight_hw = krnl_block->expand_conv->weight->size[2] * krnl_block->expand_conv->weight->size[3];
    for (int c = 0; c < krnl_block->expand_conv->weight->size[0]; c++) {
        for (int cin = 0; cin < krnl_block->expand_conv->weight->size[1]; cin++) {
            krnl_block->expand_conv->weight->data[c*weight_hw*16 + cin] =\
                krnl_block->expand_conv->weight->data[c*weight_hw*16 + cin]\
                * (1 / sqrt(krnl_block->bn0->running_var->data[c] + 0.001))\
                * krnl_block->bn0->weight->data[c];
        }
    }
    
    // Layer Fusion 2.
    // create a new bias array
    ndarray_t* fused_conv2_bias;
    int* fu2_bias_size = (int*)aligned_alloc(MEM_ALIGNMENT, sizeof(int));
    fu2_bias_size[0] = 96;
    fused_conv2_bias = create_empty_ndarray(1, fu2_bias_size);
    for (int c = 0; c < 96; c++) {
        fused_conv2_bias->data[c] =\
            -1 * krnl_block->bn1->running_mean->data[c]\
            * (1 / sqrt(krnl_block->bn1->running_var->data[c] + 0.001))\
            * krnl_block->bn1->weight->data[c]\
            + krnl_block->bn1->bias->data[c];
    }
    
    for (int c = 0; c < 96; c++) {
        for (int p = 0; p < 9; p++) {
            krnl_block->depthwise_conv->weight->data[c*9 + p] =\
                krnl_block->depthwise_conv->weight->data[c*9 + p]\
                * (1 / sqrt(krnl_block->bn1->running_var->data[c] + 0.001))\
                * krnl_block->bn1->weight->data[c];
        }
    }
    
    // Layer Fusion 3.
    // create a new bias array
    ndarray_t* fused_conv3_bias;
    int* fu3_bias_size = (int*)aligned_alloc(MEM_ALIGNMENT, sizeof(int));
    fu3_bias_size[0] = 24;
    fused_conv3_bias = create_empty_ndarray(1, fu3_bias_size);
    for (int c = 0; c < 24; c++) {
        fused_conv3_bias->data[c] =\
            -1 * krnl_block->bn2->running_mean->data[c]\
            * (1 / sqrt(krnl_block->bn2->running_var->data[c] + 0.001))\
            * krnl_block->bn2->weight->data[c]\
            + krnl_block->bn2->bias->data[c];
    }
    
    for (int cout = 0; cout < 24; cout++) {
        for (int cin = 0; cin < 96; cin++) {
            krnl_block->project_conv->weight->data[cout*96 + cin] =\
                krnl_block->project_conv->weight->data[cout*96 + cin]\
                * (1 / sqrt(krnl_block->bn2->running_var->data[cout] + 0.001))\
                * krnl_block->bn2->weight->data[cout];
        }
    }
    
    std::cout << "1: " << fused_conv1_bias->length << std::endl;
    std::cout << "2: " << fused_conv2_bias->length << std::endl;
    std::cout << "3: " << fused_conv3_bias->length << std::endl;
    
    krnl(
        input->data,
        krnl_out->final_out->data,
        krnl_block->expand_conv->weight->data,
        krnl_block->depthwise_conv->weight->data,
        krnl_block->se_reduce->weight->data,
        krnl_block->se_reduce->bias->data,
        krnl_block->se_expand->weight->data,
        krnl_block->se_expand->bias->data,
        krnl_block->project_conv->weight->data,
        in_channels,
        input_size,
        fused_conv1_bias->data,
        fused_conv2_bias->data,
        fused_conv3_bias->data
    );
    
    /*
    std::cout << "LENGTH" << std::endl;
    std::cout << krnl_block->expand_conv->weight->length << std::endl;
    std::cout << krnl_block->depthwise_conv->weight->length << std::endl;
    std::cout << krnl_block->se_reduce->weight->length << std::endl;
    std::cout << krnl_block->se_reduce->bias->length << std::endl;
    std::cout << krnl_block->se_expand->weight->length << std::endl;
    std::cout << krnl_block->se_expand->bias->length << std::endl;
    std::cout << krnl_block->project_conv->weight->length << std::endl;
    */
    
    print_total_readnum();
    print_total_writenum();
        
    for (int i = 0; i < 10; i++) {
        std::cout << "cpu_output\t[" << i << "] = " << cpu_out->final_out->data[i] << std::endl;
        std::cout << "kernel_output\t[" << i << "] = " << krnl_out->final_out->data[i] << std::endl;
    }
    
    /*
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
    */
    return 0;
}
