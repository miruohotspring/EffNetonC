#ifndef KERNEL_H
#define KERNEL_H

// for kernel
void kernel_conv2d_forward(
    float* output, const float* input, const float* weight,
    int filter_size, int stride, int groups,
    int in_channels, int out_channels, int in_size, int out_size
);

#endif