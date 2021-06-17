/*
 * kernel.c
 *
 *  Created on: Mar 24, 2021
 *      Author: shirakawa
 */


// TODO: Implementation for any filter_size.
#include <math.h>
#include <stdio.h>
#include "kernel.hpp"

void krnl(
	const float* input,
    float* final_out,
    
    // trained params
	const float* exconv_weight,
	const float* dwconv_weight,
    const float* se_reduce_weight,
    const float* se_reduce_bias,
    const float* se_expand_weight,
    const float* se_expand_bias,
	const float* project_conv_weight,
    
    // hyper params
    int in_channels,
	int in_size,
    
    const float* fused_conv1_bias,
    const float* fused_conv2_bias,
    const float* fused_conv3_bias
) {

printf("kernel started\n");

float pad0_out[MAX_PAD0] = {0};
float dwconv_out[MAX_DWCONV] = {0};
float avgpool_out[MAX_AVGPOOL] = {0};
float se_reduce_out[MAX_SEREDUCE] = {0};
float se_expand_out[MAX_SEEXPAND] = {0};
float sigmoid_out[MAX_SIGMOID] = {0};

float step1_weight[16] = {0};

#pragma HLS RESOURCE variable=pad0_out core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=dwconv_out core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=avgpool_out core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=se_reduce_out core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=se_expand_out core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=sigmoid_out core=XPM_MEMORY uram

// Step 1. Expand Convolution
    int image_size = in_size * in_size;
    int pad_size = (in_size+1) * (in_size+1);
    // a, b. Fused Convolution
    int cout, cin, p, row, col;
    for (cout = 0; cout < in_channels * 6; cout++) {
        for (cin = 0; cin < in_channels; cin++) {
            step1_weight[cin] = readmem(exconv_weight, cout*in_channels + cin);
        }
        for (row = 0; row < in_size+1; row++) {
        for (col = 0; col < in_size+1; col++) {
            float step1_out = 0;
            // padding
            if ((row < in_size) && (col < in_size)) {
                for (cin = 0; cin < in_channels; cin++) {
                    step1_out +=\
                    step1_weight[cin] *\
                    readmem(input, cin*image_size + row*in_size + col);
                }
                step1_out += readmem(fused_conv1_bias, cout);
                step1_out = step1_out * sigmoid(step1_out);
            }
            pad0_out[cout*pad_size + row*(in_size+1) + col] = step1_out;
        }
        }
    }
    
    print_readnum();
    print_writenum();

// Step 2. Depthwise Convolution
    // a. Convolution
    int out_size, out_i, weight_i, in_i;
    in_size = 113;
    out_size = 56;
    for (cout = 0; cout < in_channels*6; cout++) {
        for (row = 0; row < out_size; row++) {
        for (col = 0; col < out_size; col++) {
            for (int cin = 0; cin < 1; cin++) {
                float step2_out = 0;
                out_i = cout*out_size*out_size + row*out_size + col;
                weight_i = cout*9;
                in_i = cout*in_size*in_size + row*in_size*2 + col*2;

                step2_out += readmem(dwconv_weight, weight_i    ) * readmem(pad0_out, in_i                );
                step2_out += readmem(dwconv_weight, weight_i + 1) * readmem(pad0_out, in_i + 1            );
                step2_out += readmem(dwconv_weight, weight_i + 2) * readmem(pad0_out, in_i + 2            );
                step2_out += readmem(dwconv_weight, weight_i + 3) * readmem(pad0_out, in_i + in_size      );
                step2_out += readmem(dwconv_weight, weight_i + 4) * readmem(pad0_out, in_i + in_size + 1  );
                step2_out += readmem(dwconv_weight, weight_i + 5) * readmem(pad0_out, in_i + in_size + 2  );
                step2_out += readmem(dwconv_weight, weight_i + 6) * readmem(pad0_out, in_i + in_size*2    );
                step2_out += readmem(dwconv_weight, weight_i + 7) * readmem(pad0_out, in_i + in_size*2 + 1);
                step2_out += readmem(dwconv_weight, weight_i + 8) * readmem(pad0_out, in_i + in_size*2 + 2);
                
                step2_out += readmem(fused_conv2_bias, cout);
                step2_out = step2_out * sigmoid(step2_out);
                
                dwconv_out[out_i] = step2_out;
            }
        }
        }
    }

    print_readnum();
    print_writenum();
    
// Step 3. Squeeze-Excitation
    // a. Average Pooling
    float sum;
    image_size = out_size * out_size;
    for (cout = 0; cout < in_channels*6; cout++) {
        sum = 0;
        for (p = 0; p < image_size; p++) {
            sum += readmem(dwconv_out, cout*image_size + p);
        }
        avgpool_out[cout] = sum / image_size;
    }
    
    // b. Convolution (Squeeze)
    for (cout = 0; cout < in_channels/4; cout++) {
        float step3_out1 = 0;
        for (cin = 0; cin < in_channels*6; cin++) {
            step3_out1 +=\
            readmem(se_reduce_weight, cout*in_channels*6 + cin) *\
            readmem(avgpool_out, cin);
        }
        step3_out1 += readmem(se_reduce_bias, cout);
        step3_out1 = step3_out1 * sigmoid(step3_out1);
        se_reduce_out[cout] = step3_out1;
    }
    
    // d. Convolution (Excite)
    for (cout = 0; cout < in_channels*6; cout++) {
        float step3_out2 = 0;
        for (cin = 0; cin < in_channels/4; cin++) {
            step3_out2 += readmem(se_expand_weight, cout*in_channels/4 + cin) * readmem(se_reduce_out, cin);
        }
        step3_out2 += readmem(se_expand_bias, cout);
        se_expand_out[cout] = step3_out2;
    }
    
    // e. Sigmoid & f. Multiplication
    image_size = 56 * 56;
    for (cout = 0; cout < in_channels*6; cout++) {
        for (p = 0; p < image_size; p++) {
            sigmoid_out[cout*image_size + p] = sigmoid(readmem(se_expand_out, cout)) * readmem(dwconv_out, cout*image_size + p);
        }
    }
    
    print_readnum();
    print_writenum();
    
// Step 4. Pointwise Convolution
    // a. Convolution
    for (cout = 0; cout < 24; cout++) {
        for (p = 0; p < image_size; p++) {
            float step4_out = 0;
            for (cin = 0; cin < in_channels*6; cin++) {
                step4_out += readmem(project_conv_weight, cout*in_channels*6 + cin) * readmem(sigmoid_out, cin*image_size + p);
            }
            writemem(final_out, cout*image_size + p, step4_out + readmem(fused_conv3_bias, cout));
        }
    }
    
    print_readnum();
    print_writenum();
    
}
