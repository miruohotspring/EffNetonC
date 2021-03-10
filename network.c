#include "ndarray.h"
#include "network.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

char** names;
ndarray_t* params;


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
) {
    char str[100];
    char str2[100];
    char str3[100];
    char str4[100];
    char prefix[100] = "_blocks.";
    char block_num_char = '0' + block_num;
    sprintf(prefix, "%s%d", prefix, block_num);
    
    mb_conv_block_t* block = (mb_conv_block_t*)malloc(sizeof(mb_conv_block_t));
    *output = (mb_conv_outputs_t*)malloc(sizeof(mb_conv_outputs_t));
    
    // expand_conv
    int* expand_conv_out_size;
    conv2d_layer_t* expand_conv;
    ndarray_t* expand_conv_out;
    batchnorm_layer_t* bn0;
    ndarray_t* bn0_out;
    ndarray_t* swish0_out;
    if (expand_rate != 1) {
        expand_conv_out_size = (int*)malloc(sizeof(int)*4);
        expand_conv_out_size[0] = 1;
        expand_conv_out_size[1] = in_channels * expand_rate;
        expand_conv_out_size[2] = input_size;
        expand_conv_out_size[3] = input_size;
        sprintf(str, "%s%s", prefix, "._expand_conv.weight");
        expand_conv = Conv2d(in_channels, in_channels * expand_rate, 0, 1, 1, 0, 0, 0, 0, get_param_from_name(str));
        expand_conv_out = create_empty_ndarray(4, expand_conv_out_size);
        
        // batch_norm0
        sprintf(str, "%s%s", prefix, "._bn0.running_mean");
        sprintf(str2, "%s%s", prefix, "._bn0.running_var");
        sprintf(str3, "%s%s", prefix, "._bn0.weight");
        sprintf(str4, "%s%s", prefix, "._bn0.bias");
        bn0 = Batchnorm(
            get_param_from_name(str),
            get_param_from_name(str2),
            get_param_from_name(str3),
            get_param_from_name(str4)
        );
        bn0_out = create_empty_ndarray(4, expand_conv_out_size);
        
        // swish
        swish0_out = create_empty_ndarray(4, expand_conv_out_size);
    }
    
    // padding
    int ih = input_size + dw_pad_top + dw_pad_bottom;
    int iw = input_size + dw_pad_left + dw_pad_right;
    
    int* pad0_size = (int*)malloc(sizeof(int)*4);
    pad0_size[0] = 1;
    pad0_size[1] = in_channels * expand_rate;
    pad0_size[2] = ih;
    pad0_size[3] = iw;
    ndarray_t* pad0_out = create_empty_ndarray(4, pad0_size);
    
    // depthwise_conv
    int* depthwise_conv_out_size = (int*)malloc(sizeof(int)*4);
    depthwise_conv_out_size[0] = 1;
    depthwise_conv_out_size[1] = in_channels * expand_rate;
    depthwise_conv_out_size[2] = (ih-dw_kernel)/dw_stride + 1;
    depthwise_conv_out_size[3] = (iw-dw_kernel)/dw_stride + 1;
    sprintf(str, "%s%s", prefix, "._depthwise_conv.weight");
    conv2d_layer_t* depthwise_conv = Conv2d(in_channels * expand_rate, in_channels * expand_rate, in_channels * expand_rate, dw_kernel, dw_stride, dw_pad_top, dw_pad_bottom, dw_pad_right, dw_pad_left, get_param_from_name(str));
    ndarray_t* depthwise_conv_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // batch_norm1
    sprintf(str, "%s%s", prefix, "._bn1.running_mean");
    sprintf(str2, "%s%s", prefix, "._bn1.running_var");
    sprintf(str3, "%s%s", prefix, "._bn1.weight");
    sprintf(str4, "%s%s", prefix, "._bn1.bias");
    batchnorm_layer_t* bn1 = Batchnorm(
        get_param_from_name(str),
        get_param_from_name(str2),
        get_param_from_name(str3),
        get_param_from_name(str4)
    );
    ndarray_t* bn1_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // swish
    ndarray_t* swish1_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // average pooling
    int* avgpool_size = (int*)malloc(sizeof(int)*4);
    avgpool_size[0] = 1;
    avgpool_size[1] = in_channels * expand_rate;
    avgpool_size[2] = 1;
    avgpool_size[3] = 1;
    ndarray_t* avgpool_out = create_empty_ndarray(4, avgpool_size);
    
    // se_reduce
    int* se_reduce_out_size = (int*)malloc(sizeof(int)*4);
    se_reduce_out_size[0] = 1;
    se_reduce_out_size[1] = in_channels/4;
    se_reduce_out_size[2] = 1;
    se_reduce_out_size[3] = 1;
    sprintf(str, "%s%s", prefix, "._se_reduce.weight");
    sprintf(str2, "%s%s", prefix, "._se_reduce.bias");
    conv2d_layer_t* se_reduce = Conv2d_bias(in_channels * expand_rate, in_channels/4, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name(str),
        get_param_from_name(str2));
    ndarray_t* se_reduce_out = create_empty_ndarray(4, se_reduce_out_size);
    
    // swish
    ndarray_t* swish2_out = create_empty_ndarray(4, se_reduce_out_size);
    
    // se_expand
    int* se_expand_out_size = (int*)malloc(sizeof(int)*4);
    se_expand_out_size[0] = 1;
    se_expand_out_size[1] = in_channels * expand_rate;
    se_expand_out_size[2] = 1;
    se_expand_out_size[3] = 1;
    sprintf(str, "%s%s", prefix, "._se_expand.weight");
    sprintf(str2, "%s%s", prefix, "._se_expand.bias");
    conv2d_layer_t* se_expand = Conv2d_bias(in_channels/4, in_channels * expand_rate, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name(str),
        get_param_from_name(str2));
    ndarray_t* se_expand_out = create_empty_ndarray(4, se_expand_out_size);
    
    // sigmoid multiply
    ndarray_t* sigmoid_out = create_empty_ndarray(4, depthwise_conv_out_size);
    
    // project conv
    int* project_conv_out_size = (int*)malloc(sizeof(int)*4);
    project_conv_out_size[0] = 1;
    project_conv_out_size[1] = out_channels;
    project_conv_out_size[2] = (ih-dw_kernel)/dw_stride + 1;
    project_conv_out_size[3] = (iw-dw_kernel)/dw_stride + 1;
    sprintf(str, "%s%s", prefix, "._project_conv.weight");
    conv2d_layer_t* project_conv = Conv2d(in_channels * expand_rate, out_channels, 0, 1, 1, 0, 0, 0, 0, 
        get_param_from_name(str));
    ndarray_t* project_conv_out = create_empty_ndarray(4, project_conv_out_size);
    
    // batch_norm2
    sprintf(str, "%s%s", prefix, "._bn2.running_mean");
    sprintf(str2, "%s%s", prefix, "._bn2.running_var");
    sprintf(str3, "%s%s", prefix, "._bn2.weight");
    sprintf(str4, "%s%s", prefix, "._bn2.bias");
    batchnorm_layer_t* bn2 = Batchnorm(
        get_param_from_name(str),
        get_param_from_name(str2),
        get_param_from_name(str3),
        get_param_from_name(str4)
    );
    ndarray_t* bn2_out = create_empty_ndarray(4, project_conv_out_size);
    
    block->expand_conv = expand_conv;
    block->depthwise_conv = depthwise_conv;
    block->se_reduce = se_reduce;
    block->se_expand = se_expand;
    block->project_conv = project_conv;
    block->bn0 = bn0;
    block->bn1 = bn1;
    block->bn2 = bn2;
    block->dw_pad_top = dw_pad_top;
    block->dw_pad_bottom = dw_pad_bottom;
    block->dw_pad_right = dw_pad_right;
    block->dw_pad_left = dw_pad_left;
    block->in_channels = in_channels;
    block->out_channels = out_channels;
    
    (*output)->expand_conv_out = expand_conv_out;
    (*output)->depthwise_conv_out = depthwise_conv_out;
    (*output)->se_reduce_out = se_reduce_out;
    (*output)->se_expand_out = se_expand_out;
    (*output)->project_conv_out = project_conv_out;
    (*output)->bn0_out = bn0_out;
    (*output)->bn1_out = bn1_out;
    (*output)->swish0_out = swish0_out;
    (*output)->swish1_out = swish1_out;
    (*output)->swish2_out = swish2_out;
    (*output)->avgpool_out = avgpool_out;
    (*output)->sigmoid_out = sigmoid_out;
    (*output)->pad0_out = pad0_out;
    (*output)->final_out = bn2_out;
    
    return block;
}

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
) {
    conv2d_layer_t* layer = (conv2d_layer_t*)malloc(sizeof(conv2d_layer_t));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->groups = groups;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding_top = padding_top;
    layer->padding_bottom = padding_bottom;
    layer->padding_right = padding_right;
    layer->padding_left = padding_left;
    layer->weight = weight;
    layer->bias = NULL;
    
    return layer;
}

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
) {
    conv2d_layer_t* layer = (conv2d_layer_t*)malloc(sizeof(conv2d_layer_t));
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->groups = groups;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding_top = padding_top;
    layer->padding_bottom = padding_bottom;
    layer->padding_right = padding_right;
    layer->padding_left = padding_left;
    layer->weight = weight;
    layer->bias = bias;
    
    return layer;
}

batchnorm_layer_t* Batchnorm(
    ndarray_t* running_mean,
    ndarray_t* running_var,
    ndarray_t* weight,
    ndarray_t* bias
) {
    batchnorm_layer_t* layer = (batchnorm_layer_t*)malloc(sizeof(batchnorm_layer_t));
    layer->running_mean = running_mean;
    layer->running_var = running_var;
    layer->weight = weight;
    layer->bias = bias;
    
    return layer;
}

conv2d_t* create_conv2d(
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride
) {
    conv2d_t* layer = (conv2d_t*)malloc(sizeof(conv2d_t));
    int* output_size = (int*)malloc(sizeof(int)*4);
    return layer;
}

// conv2dの出力(入れ物)を作成する
ndarray_t* create_conv2d_output(
    int out_channels,
    int kernel_size,
    int stride,
    int ih,
    int iw
) {
    int* size = (int*)malloc(sizeof(int)*4);
    size[0] = 1;
    size[1] = out_channels;
    size[2] = (ih - kernel_size) / stride + 1;
    size[3] = (iw - kernel_size) / stride + 1;
    
    return create_empty_ndarray(4, size);
}

// create param
ndarray_t* get_param_from_name(char* name) {
    int index = 1000;
    for (int i = 0; i < param_num; i++) {
        if (strcmp(name, names[i]) == 0) index = i;
    }
    if (index == 1000) {
        printf("%s, %s\n", "name not found", name);
        exit(1);
    }
    return &params[index];
}
    
ndarray_t* create_param_from_name(char* name) {
    ndarray_t* param = (ndarray_t*)malloc(sizeof(ndarray_t));
    int length = 1;
    int dim;
    int* size;
    float* data;
    
    FILE* f;
    char path[100] = "./data/";
    char c;
    char s[128];
    int j = 0;
    strcat(path, name);
    strcat(path, ".txt");
    if ((f = fopen(path, "r")) == NULL) {
        printf("file open error");
        exit(EXIT_FAILURE);
    }
    if ((dim = fgetc(f) - '0') == 0) return NULL;
    while ((fgetc(f)) != ',');
    
    size = (int*)malloc(sizeof(int)*dim);
    for (int i = 0; i < dim; i++) {
        for (j = 0; (c = fgetc(f)), (c != ',' && c != EOF); j++) s[j] = c;
        s[j] = 0;
        size[i] = atoi(s);
        length *= size[i];
    }
    
    data = (float*)malloc(sizeof(float)*length);
    for (int i = 0; i < length; i++) {
        for (j = 0; (c = fgetc(f)), (c != ',' && c != EOF); j++) s[j] = c;
        s[j] = 0;
        data[i] = atof(s);
    }
    
    param->length = length;
    param->dim = dim;
    param->size = size;
    param->data = data;
    
    fclose(f);
    return param;
}

void load_params(char*** names_p, ndarray_t** params_p) {
    *params_p = (ndarray_t*)malloc(param_num * sizeof(ndarray_t));
    *names_p = (char**)malloc(param_num * sizeof(char*));
    for (int i = 0; i < param_num; i++) {
        (*names_p)[i] = (char*)malloc(100 * sizeof(char));
    }
    
    FILE* f;
    char readline[100];
    
    if ((f = fopen("./data/module_list_all.txt", "r")) == NULL) {
        printf("file open error: cannot open list");
        exit(EXIT_FAILURE);
    }
    int i = 0;
    while (fgets(readline, 100, f) != NULL) {
        readline[strcspn(readline, "\n")] = 0;
        ndarray_t* param = create_param_from_name(readline);
        if (param == NULL) continue;
        strcpy((*names_p)[i], readline);
        (*params_p)[i] = *param;
        i++;
    }    
    fclose(f);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-1.0 * x));
}

void sigmoid_multiply(ndarray_t* output, const ndarray_t* input, const ndarray_t* multiplier) {
    assert(output->dim == multiplier->dim);
    int image_size = output->size[2] * output->size[3];
    for (int c = 0; c < output->size[1]; c++) {
        for (int p = 0; p < image_size; p++) {
            output->data[c*image_size + p] = sigmoid(input->data[c]) * multiplier->data[c*image_size + p];
        }
    }
}

void swish(ndarray_t* output, const ndarray_t* input) {
    assert(output->dim == input->dim);
    for (int i = 0; i < output->length; i++) {
        output->data[i] = input->data[i] * sigmoid(input->data[i]);
    }
}

void mbconv6_forward(
    mb_conv_outputs_t* o,
    const ndarray_t* input,
    const mb_conv_block_t* b
    )
{
    conv2d_forward(o->expand_conv_out, input, b->expand_conv);
    batchnorm_forward(o->bn0_out, o->expand_conv_out, b->bn0);
    swish(o->swish0_out, o->bn0_out);
    zero_padding(o->pad0_out, o->swish0_out, b->dw_pad_top, b->dw_pad_bottom, b->dw_pad_right, b->dw_pad_left);
    conv2d_forward(o->depthwise_conv_out, o->pad0_out, b->depthwise_conv);
    batchnorm_forward(o->bn1_out, o->depthwise_conv_out, b->bn1);
    swish(o->swish1_out, o->bn1_out);
    average_pooling(o->avgpool_out, o->swish1_out);
    conv2d_forward(o->se_reduce_out, o->avgpool_out, b->se_reduce);
    swish(o->swish2_out, o->se_reduce_out);
    conv2d_forward(o->se_expand_out, o->swish2_out, b->se_expand);
    sigmoid_multiply(o->sigmoid_out, o->se_expand_out, o->swish1_out);
    conv2d_forward(o->project_conv_out, o->sigmoid_out, b->project_conv);
    batchnorm_forward(o->final_out, o->project_conv_out, b->bn2);
    if (b->in_channels == b->out_channels) add_ndarray(o->final_out, input);
}

void mbconv1_forward(
    mb_conv_outputs_t* o,
    const ndarray_t* input,
    const mb_conv_block_t* b
    )
{
    zero_padding(o->pad0_out, input, b->dw_pad_top, b->dw_pad_bottom, b->dw_pad_right, b->dw_pad_left);
    conv2d_forward(o->depthwise_conv_out, o->pad0_out, b->depthwise_conv);
    batchnorm_forward(o->bn1_out, o->depthwise_conv_out, b->bn1);
    swish(o->swish1_out, o->bn1_out);
    average_pooling(o->avgpool_out, o->swish1_out);
    conv2d_forward(o->se_reduce_out, o->avgpool_out, b->se_reduce);
    swish(o->swish2_out, o->se_reduce_out);
    conv2d_forward(o->se_expand_out, o->swish2_out, b->se_expand);
    sigmoid_multiply(o->sigmoid_out, o->se_expand_out, o->swish1_out);
    conv2d_forward(o->project_conv_out, o->sigmoid_out, b->project_conv);
    batchnorm_forward(o->final_out, o->project_conv_out, b->bn2);
    if (b->in_channels == b->out_channels) add_ndarray(o->final_out, input);
}


void conv2d_forward(
    ndarray_t* output,
    const ndarray_t* input,
    const conv2d_layer_t* layer
    )
{
    assert(output->dim == input->dim);
    int image_size = output->size[2] * output->size[3];
    int input_image_size = input->size[2] * input->size[3];
    if (layer->kernel_size == 1 && layer->bias == NULL) {
        for (int cout = 0; cout < output->size[1]; cout++) {
            for (int p = 0; p < image_size; p++) {
                for (int cin = 0; cin < layer->in_channels; cin++) {
                    output->data[cout*image_size + p] +=\
                    layer->weight->data[cout*layer->in_channels + cin] *\
                    input->data[cin*image_size + p];
                }    
            }
        }
    }
    else if (layer->kernel_size == 1) {
        for (int cout = 0; cout < output->size[1]; cout++) {
            for (int p = 0; p < image_size; p++) {
                for (int cin = 0; cin < layer->in_channels; cin++) {
                    output->data[cout*image_size + p] +=\
                    layer->weight->data[cout*layer->in_channels + cin] *\
                    input->data[cin*image_size + p]; 
                }    
                output->data[cout*image_size + p] += layer->bias->data[cout];
            }
        }
    }
    else if (layer->kernel_size == 3) {
        if (layer->groups != 0) {
            for (int cout = 0; cout < output->size[1]; cout++) {
                for (int row = 0; row < output->size[2]; row++) {
                for (int col = 0; col < output->size[3]; col++) {
                    for (int cin = 0; cin < 1; cin++) {
                        int out_i = cout*image_size + row*output->size[3] + col;
                        int weight_i = cout*9;
                        int in_i = cout*input_image_size + row*input->size[3]*layer->stride + col*layer->stride;
                        int w = input->size[3];
                        
                        output->data[out_i] += layer->weight->data[weight_i]     * input->data[in_i];
                        output->data[out_i] += layer->weight->data[weight_i + 1] * input->data[in_i + 1];
                        output->data[out_i] += layer->weight->data[weight_i + 2] * input->data[in_i + 2];
                        output->data[out_i] += layer->weight->data[weight_i + 3] * input->data[in_i + w];
                        output->data[out_i] += layer->weight->data[weight_i + 4] * input->data[in_i + w + 1];
                        output->data[out_i] += layer->weight->data[weight_i + 5] * input->data[in_i + w + 2];
                        output->data[out_i] += layer->weight->data[weight_i + 6] * input->data[in_i + w*2];
                        output->data[out_i] += layer->weight->data[weight_i + 7] * input->data[in_i + w*2 + 1];
                        output->data[out_i] += layer->weight->data[weight_i + 8] * input->data[in_i + w*2 + 2];
                        
                    }
                }
                }
            }
        }
        else {
            for (int cout = 0; cout < output->size[1]; cout++) {
                for (int row = 0; row < output->size[2]; row++) {
                for (int col = 0; col < output->size[3]; col++) {
                    for (int cin = 0; cin < input->size[1]; cin++) {
                        int out_i = cout*image_size + row*output->size[3] + col;
                        int weight_i = cout*input->size[1]*9 + cin*9;
                        int in_i = cin*input_image_size + row*input->size[3]*layer->stride + col*layer->stride;
                        int w = input->size[3];
                        
                        output->data[out_i] += layer->weight->data[weight_i]     * input->data[in_i];
                        output->data[out_i] += layer->weight->data[weight_i + 1] * input->data[in_i + 1];
                        output->data[out_i] += layer->weight->data[weight_i + 2] * input->data[in_i + 2];
                        output->data[out_i] += layer->weight->data[weight_i + 3] * input->data[in_i + w];
                        output->data[out_i] += layer->weight->data[weight_i + 4] * input->data[in_i + w + 1];
                        output->data[out_i] += layer->weight->data[weight_i + 5] * input->data[in_i + w + 2];
                        output->data[out_i] += layer->weight->data[weight_i + 6] * input->data[in_i + w*2];
                        output->data[out_i] += layer->weight->data[weight_i + 7] * input->data[in_i + w*2 + 1];
                        output->data[out_i] += layer->weight->data[weight_i + 8] * input->data[in_i + w*2 + 2];
                        
                    }
                }
                }
            }
        }
    }
    else if (layer->kernel_size == 5) {
        for (int cout = 0; cout < output->size[1]; cout++) {
            for (int row = 0; row < output->size[2]; row++) {
            for (int col = 0; col < output->size[3]; col++) {
                for (int cin = 0; cin < 1; cin++) {
                    int out_i = cout*image_size + row*output->size[3] + col;
                    int weight_i = cout*25;
                    int in_i = cout*input_image_size + row*input->size[3]*layer->stride + col*layer->stride;
                    int w = input->size[3];
                    
                    output->data[out_i] += layer->weight->data[weight_i]        * input->data[in_i];
                    output->data[out_i] += layer->weight->data[weight_i + 1]    * input->data[in_i + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 2]    * input->data[in_i + 2];
                    output->data[out_i] += layer->weight->data[weight_i + 3]    * input->data[in_i + 3];
                    output->data[out_i] += layer->weight->data[weight_i + 4]    * input->data[in_i + 4];
                    output->data[out_i] += layer->weight->data[weight_i + 5]    * input->data[in_i + w];
                    output->data[out_i] += layer->weight->data[weight_i + 6]    * input->data[in_i + w + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 7]    * input->data[in_i + w + 2];
                    output->data[out_i] += layer->weight->data[weight_i + 8]    * input->data[in_i + w + 3];
                    output->data[out_i] += layer->weight->data[weight_i + 9]    * input->data[in_i + w + 4];
                    output->data[out_i] += layer->weight->data[weight_i + 10]   * input->data[in_i + w*2];
                    output->data[out_i] += layer->weight->data[weight_i + 11]   * input->data[in_i + w*2 + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 12]   * input->data[in_i + w*2 + 2];
                    output->data[out_i] += layer->weight->data[weight_i + 13]   * input->data[in_i + w*2 + 3];
                    output->data[out_i] += layer->weight->data[weight_i + 14]   * input->data[in_i + w*2 + 4];
                    output->data[out_i] += layer->weight->data[weight_i + 15]   * input->data[in_i + w*3];
                    output->data[out_i] += layer->weight->data[weight_i + 16]   * input->data[in_i + w*3 + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 17]   * input->data[in_i + w*3 + 2];
                    output->data[out_i] += layer->weight->data[weight_i + 18]   * input->data[in_i + w*3 + 3];
                    output->data[out_i] += layer->weight->data[weight_i + 19]   * input->data[in_i + w*3 + 4];
                    output->data[out_i] += layer->weight->data[weight_i + 20]   * input->data[in_i + w*4];
                    output->data[out_i] += layer->weight->data[weight_i + 21]   * input->data[in_i + w*4 + 1];
                    output->data[out_i] += layer->weight->data[weight_i + 22]   * input->data[in_i + w*4 + 2];
                    output->data[out_i] += layer->weight->data[weight_i + 23]   * input->data[in_i + w*4 + 3];
                    output->data[out_i] += layer->weight->data[weight_i + 24]   * input->data[in_i + w*4 + 4];
                    
                }
            }
            }
        }
    }
}

void batchnorm_forward(
    ndarray_t* output,
    const ndarray_t* input,
    const batchnorm_layer_t* layer
    )
{
    assert(output->dim == input->dim);
    int image_size = output->size[2] * output->size[3];
    for (int c = 0; c < output->size[1]; c++) {
        for (int p = 0; p < image_size; p++) {
            output->data[c*image_size + p] =\
                (input->data[c*image_size + p] - layer->running_mean->data[c])\
                * (1 / sqrt(layer->running_var->data[c] + 0.001))\
                * layer->weight->data[c]\
                + layer->bias->data[c];
        }
    }
}

/*
zero padding
example with padding of (top, bottom, right, left) = (1, 1, 1, 1):
                   |0 0 0 0|
|a b|              |0 a b 0|
|c d|      ->      |0 c d 0|
                   |0 0 0 0|
*/
void zero_padding(ndarray_t* output, const ndarray_t* input, int top, int bottom, int right, int left) {
    int i_in = 0;
    int i_out = 0;
    for (int c = 0; c < output->size[1]; c++) {
        for (int h = 0; h < output->size[2]; h++) {
            for (int w = 0; w < output->size[3]; w++) {
                if ((h >= top) && (h < output->size[2] - bottom) && (w >= left) && (w < output->size[3] - right)) {
                    output->data[i_out] = input->data[i_in];
                    i_in++;
                } else {
                    output->data[i_out] = 0;
                }
                i_out++;
            }
        }
    }
}

void average_pooling(ndarray_t* output, const ndarray_t* input) {
    int image_size = input->size[2] * input->size[3];
    for (int c = 0; c < output->size[1]; c++) {
        float sum = 0;
        for (int p = 0; p < image_size; p++) {
            sum += input->data[c*image_size + p];
        }
        output->data[c] = sum / image_size;
    }
}











































