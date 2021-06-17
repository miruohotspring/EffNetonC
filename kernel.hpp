#define MAX_INPUT 200704
#define MAX_BN0 1204224
#define MAX_PAD0 1225824
#define MAX_DWCONV 301056
#define MAX_AVGPOOL 96
#define MAX_SEREDUCE 4
#define MAX_SEEXPAND 96
#define MAX_SIGMOID 301056
#define MAX_OUTPUT 75264

#include "util.hpp"

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
);
